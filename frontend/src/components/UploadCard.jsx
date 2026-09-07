import React, { useCallback, useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, FileAudio, FileImage, FileVideo, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { AnalysisProgress } from '@/components/AnalysisProgress';

const typeConfig = {
  image: {
    icon: FileImage,
    title: 'Image Forensics',
    description: 'Face swaps, GAN synthesis, and micro-texture boundaries',
    accept: 'image/*',
    color: 'text-blue-400',
    borderGlow: 'hover:border-blue-500/60 hover:shadow-blue-500/10',
    iconBg: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
  },
  video: {
    icon: FileVideo,
    title: 'Video Forensics',
    description: 'MINTIME multi-face tracking & inter-frame jitter',
    accept: 'video/*',
    color: 'text-cyan-400',
    borderGlow: 'hover:border-cyan-500/60 hover:shadow-cyan-500/10',
    iconBg: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400',
  },
  audio: {
    icon: FileAudio,
    title: 'Acoustic Forensics',
    description: '107-feature DSP and synthetic speech clone verification',
    accept: 'audio/*',
    color: 'text-indigo-400',
    borderGlow: 'hover:border-indigo-500/60 hover:shadow-indigo-500/10',
    iconBg: 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400',
  },
};

export const UploadCard = ({ type, onAnalysisComplete }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [error, setError] = useState(null);
  
  // 1. ADDED: A reference to target the hidden file input directly
  const fileInputRef = useRef(null);
  
  const config = typeConfig[type] || typeConfig.image;
  const Icon = config.icon;

const transformBackendResponse = (backendResult) => {
    const isDeepfake = backendResult.label === "FAKE";
    const fileType = backendResult.file_type || 'image';
    const confidence = backendResult.confidence;
    
    let details = [
      {
        label: 'Detection Verdict',
        value: isDeepfake ? 'FORGED / SYNTHETIC' : 'NATURAL / AUTHENTIC',
        status: isDeepfake ? 'fake' : 'authentic'
      },
      {
        label: 'Forensic Certainty',
        value: `${confidence}%`,
        status: confidence > 70 ? (isDeepfake ? 'fake' : 'authentic') : 'warning'
      }
    ];

    if (fileType === 'image') {
      details.push(
        {
          label: 'Spatial CNN Stream',
          value: backendResult.spatial_score !== undefined ? `${Math.round(backendResult.spatial_score * 100)}% anomaly (Seams/Texture)` : 'Active',
          status: 'info'
        },
        {
          label: 'Frequency Spectrum',
          value: backendResult.frequency_score !== undefined ? `${Math.round(backendResult.frequency_score * 100)}% anomaly (2D FFT / DCT)` : 'Active',
          status: 'info'
        },
        {
          label: 'Attention Enhancement',
          value: backendResult.attention_score !== undefined ? `${Math.round(backendResult.attention_score * 100)}% focus (Ocular/Perioral)` : 'Active',
          status: 'info'
        },
        {
          label: 'Vision Transformer',
          value: backendResult.vit_score !== undefined ? `${Math.round(backendResult.vit_score * 100)}% patch probability` : 'Active',
          status: 'info'
        }
      );
      if (backendResult.fallback_triggered && backendResult.fallback_notes && backendResult.fallback_notes.length > 0) {
        details.push({
          label: 'Rule-Based Fallback',
          value: backendResult.fallback_notes[0],
          status: 'warning'
        });
      }
    } else if (fileType === 'video') {
      details.push(
        {
          label: 'MINTIME Face Tracker',
          value: 'Size-Invariant Multi-Identity Tracking',
          status: 'info'
        },
        {
          label: 'Spatio-Temporal Sequence',
          value: backendResult.temporal_score !== undefined ? `${Math.round(backendResult.temporal_score * 100)}% inter-frame motion jitter` : 'Active',
          status: 'info'
        },
        {
          label: 'Spatial & Frequency',
          value: (backendResult.spatial_score !== undefined && backendResult.frequency_score !== undefined) ? `Spatial: ${Math.round(backendResult.spatial_score * 100)}% | Freq: ${Math.round(backendResult.frequency_score * 100)}%` : 'Active',
          status: 'info'
        }
      );
      if (backendResult.audio_score !== null && backendResult.audio_score !== undefined) {
        details.push({
          label: 'Multimodal Audio Forensics',
          value: `${Math.round(backendResult.audio_score * 100)}% voice synthetic anomaly`,
          status: 'info'
        });
      }
    } else if (fileType === 'audio') {
      details.push(
        {
          label: 'Acoustic Architecture',
          value: '107-Feature DSP + Spectral Waveform Model',
          status: 'info'
        },
        {
          label: 'Synthetic Voice Probability',
          value: `${Math.round(backendResult.probability * 100)}%`,
          status: isDeepfake ? 'fake' : 'authentic'
        }
      );
    }
    
    return {
      fileName: backendResult.filename,
      fileType: fileType,
      overallScore: backendResult.confidence,
      isDeepfake: isDeepfake,
      confidence: backendResult.confidence, 
      certainty: confidence,
      probability: backendResult.probability,
      segments: [
        {
          start: 0,
          end: 100,
          isFake: isDeepfake,
          confidence: backendResult.confidence,
          probability: backendResult.probability,
          reason: isDeepfake ? 'Multi-branch deepfake artifacts detected' : 'Authentic photographic patterns confirmed'
        }
      ],
      details: details,
      suspicious_frames: backendResult.suspicious_frames || []
    };
  };

  const analyzeFile = async (file) => {
    setIsAnalyzing(true);
    setError(null);
    setProgress(0);
    setStage('[PREPROCESSING] Multi-Face Detection & Normalization...');

    const previewUrl = URL.createObjectURL(file);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const backendResult = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let crawlInterval;

        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percentComplete = Math.round((event.loaded / event.total) * 35);
            setProgress(percentComplete);
            setStage(`[NETWORK] Uploading ${Math.round(event.loaded / 1024 / 1024)}MB...`);
          }
        };

        xhr.upload.onload = () => {
          setProgress(35);
          setStage('[ML] Running Spatial CNN & Frequency (FFT/DCT) Analysis...');
          
          let currentProgress = 35;
          crawlInterval = setInterval(() => {
            currentProgress += (92 - currentProgress) * 0.15;
            const p = Math.round(currentProgress);
            setProgress(p);
            if (p > 60 && p < 80) {
              setStage('[ATTN] Attention Enhancement & Spatio-Temporal Sequence...');
            } else if (p >= 80) {
              setStage('[FUSION] Multi-Branch Feature Fusion & Safety Fallback...');
            }
          }, 450);
        };

        xhr.onload = () => {
          clearInterval(crawlInterval);
          if (xhr.status >= 200 && xhr.status < 300) {
            setProgress(100);
            setStage('[DONE] Analysis Complete');
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error(`Server Error: ${xhr.statusText}`));
          }
        };

        xhr.onerror = () => {
          clearInterval(crawlInterval);
          reject(new Error('Network connection failed'));
        };

        xhr.open('POST', 'http://127.0.0.1:8000/analyze');
        xhr.send(formData);
      });

      const transformedResult = transformBackendResponse(backendResult);
      // ATTACH THE PREVIEW URL TO THE RESULT
      transformedResult.previewUrl = previewUrl; 
      
      setTimeout(() => {
        if (onAnalysisComplete) {
          onAnalysisComplete(transformedResult);
        }
        setIsAnalyzing(false);
        setProgress(0);
        setStage('');
      }, 800);

    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message);
      setIsAnalyzing(false);
      setProgress(0);
      setStage('');
    }
  };

  // 2. ADDED: Programmatic click handler for the whole card
  const handleCardClick = () => {
    if (!isAnalyzing && !uploadedFile && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setUploadedFile(file);
      analyzeFile(file);
    }
  }, []);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      analyzeFile(file);
    }
  }, []);

  const handleRemoveFile = useCallback((e) => {
    e.stopPropagation(); // Prevents clicking "X" from opening the file dialog again
    setUploadedFile(null);
    setError(null);
    
    // 3. ADDED: Clear the input value so the same file can be uploaded again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <Card
      onClick={handleCardClick}
      className={cn(
        'group relative overflow-hidden transition-all duration-300 rounded-2xl cursor-pointer',
        'bg-slate-900/70 backdrop-blur-md border border-slate-800 shadow-xl',
        'hover:bg-slate-900 hover:shadow-2xl hover:-translate-y-1',
        config.borderGlow,
        isDragging && 'border-blue-500 bg-slate-900/90 shadow-blue-500/20 scale-[1.02]',
        uploadedFile && 'border-slate-700 bg-slate-900/90 cursor-default',
        isAnalyzing && 'opacity-95 pointer-events-none'
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        ref={fileInputRef}
        accept={config.accept}
        onChange={handleFileSelect}
        className="hidden"
        disabled={isAnalyzing}
      />
      
      {/* Ambient Top Glow on Hover */}
      <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
      
      <CardHeader className="text-center pb-2 relative z-20">
        <div className={cn(
          'mx-auto mb-3 p-4 rounded-xl border transition-all duration-300 group-hover:scale-105 shadow-inner',
          config.iconBg
        )}>
          <Icon className="w-7 h-7" />
        </div>
        <CardTitle className="text-base md:text-lg font-bold text-white tracking-tight">
          {config.title}
        </CardTitle>
        <CardDescription className="text-slate-400 text-xs mt-1.5 leading-relaxed">
          {config.description}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="text-center relative z-20 pt-2 pb-6">
        {isAnalyzing ? (
          <AnalysisProgress progress={progress} stage={stage} />
        ) : uploadedFile ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2 px-3 py-2 bg-slate-800/80 rounded-lg border border-slate-700 backdrop-blur-sm">
              <span className="text-xs text-slate-200 font-mono truncate max-w-[200px]">
                {uploadedFile.name}
              </span>
              <button
                onClick={handleRemoveFile}
                className="p-1 hover:bg-red-500/20 rounded-md transition-colors z-30 relative text-slate-400 hover:text-red-400"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {error && (
              <p className="text-xs text-red-400 font-mono mt-2 bg-red-950/40 p-2.5 rounded-lg border border-red-500/30 text-left">
                {error}
              </p>
            )}
          </div>
        ) : (
          <div className="mt-2 py-3 px-4 rounded-xl border border-dashed border-slate-800 bg-slate-950/40 group-hover:border-slate-700 transition-colors flex items-center justify-center gap-2 text-slate-400 text-xs font-mono group-hover:text-blue-400">
            <Upload className="w-3.5 h-3.5" />
            <span>Drop file or click to browse</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};