import React, { useState, useCallback } from 'react';
import { HeroSection } from '@/components/HeroSection';
import { UploadCard } from '@/components/UploadCard';
import { AnalysisResult } from '@/components/AnalysisResult';
import { Button } from '@/components/ui/button';
import { RotateCcw, ShieldCheck, Activity, Cpu, Layers, Radio, Award } from 'lucide-react';

const Index = () => {
  const [analysisState, setAnalysisState] = useState('idle');
  const [result, setResult] = useState(null);

  const handleAnalysisComplete = useCallback((analysisResult) => {
    console.log('Analysis complete:', analysisResult);
    setResult(analysisResult);
    setAnalysisState('complete');
  }, []);

  const handleReset = useCallback(() => {
    setAnalysisState('idle');
    setResult(null);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-blue-600 selection:text-white flex flex-col font-sans">
      
      {/* Top University Institutional Banner */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md px-4 lg:px-8 py-3 shadow-lg">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          
          {/* Institution Logo & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-md shadow-blue-500/20 ring-1 ring-white/10">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 uppercase tracking-wider font-mono">
                  PCE B.Tech Capstone
                </span>
                <span className="text-[11px] text-slate-400 hidden md:inline font-medium">
                  Academic Year 2026–27
                </span>
              </div>
              <h1 className="text-sm md:text-base font-bold text-slate-100 tracking-tight">
                Pillai College of Engineering (Autonomous)
                <span className="text-slate-400 font-normal hidden sm:inline"> • Dept. of Computer Engineering</span>
              </h1>
            </div>
          </div>

          {/* Guide & Student Contributors */}
          <div className="text-xs text-slate-300 font-mono flex flex-col sm:items-end text-center sm:text-right">
            <div>
              <span className="text-slate-400">Supervisor:</span> <span className="text-blue-300 font-semibold">Prof. Kirti Rana</span>
            </div>
            <div className="text-[11px] text-slate-400">
              Team: Pranav N. • Navaneeth M. • Vishagh N. • Tejas A.
            </div>
          </div>

        </div>
      </header>

      {/* Main Container */}
      <div className="relative flex-grow flex flex-col">
        
        {/* Subtle Engineering Grid Background Pattern */}
        <div 
          className="absolute inset-0 pointer-events-none opacity-[0.03] z-0" 
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, #38bdf8 1px, transparent 0)`,
            backgroundSize: '28px 28px'
          }}
        />

        {/* Ambient Top Glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] bg-blue-600/10 blur-[130px] rounded-full pointer-events-none z-0" />

        <main className="relative z-10 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 flex-grow">
          
          {analysisState === 'idle' && (
            <div className="space-y-12 animate-fade-in">
              
              {/* Modernized Hero Section */}
              <HeroSection />

              {/* Upload Workspace Card Container */}
              <section className="space-y-6">
                <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-slate-800/80 pb-4 gap-2">
                  <div>
                    <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                      <Radio className="w-5 h-5 text-blue-400 animate-pulse" />
                      Multimodal Forensic Inspection Console
                    </h2>
                    <p className="text-sm text-slate-400 mt-1">
                      Select an evaluation modality to execute multi-branch deepfake feature analysis.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                    <span>Backend ML Engine: <strong className="text-emerald-400">Online</strong></span>
                  </div>
                </div>
                
                {/* 3 Modality Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <UploadCard type="image" onAnalysisComplete={handleAnalysisComplete} />
                  <UploadCard type="video" onAnalysisComplete={handleAnalysisComplete} />
                  <UploadCard type="audio" onAnalysisComplete={handleAnalysisComplete} />
                </div>
              </section>

              {/* Architectural Highlights Banner */}
              <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-4">
                
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-blue-500/40 transition-all flex items-start gap-3.5">
                  <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 shrink-0">
                    <Layers className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono">Spatial Boundary CNN</h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      Outer 15% blending seam edge density, CLAHE Laplacian variance, and lighting asymmetry analysis.
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-cyan-500/40 transition-all flex items-start gap-3.5">
                  <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 shrink-0">
                    <Activity className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono">2D FFT & 8×8 DCT</h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      Azimuthal power spectrum decomposition capturing GAN upsampling artifacts and compression grids.
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-indigo-500/40 transition-all flex items-start gap-3.5">
                  <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 shrink-0">
                    <Cpu className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono">Dual ViT Ensemble</h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      Self-attention patch transformers for StyleGAN synthesis and FaceForensics++ face-swap detection.
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-emerald-500/40 transition-all flex items-start gap-3.5">
                  <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shrink-0">
                    <Award className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono">Dynamic Fusion & Fallback</h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                      Multi-branch weighted aggregation combined with deterministic rule-based safety validation.
                    </p>
                  </div>
                </div>

              </section>

            </div>
          )}

          {analysisState === 'complete' && result && (
            <section className="max-w-4xl mx-auto py-4 animate-fade-in space-y-6">
              
              {/* Back to Console Header */}
              <div className="flex flex-col sm:flex-row items-center justify-between p-4 rounded-xl bg-slate-900/80 border border-slate-800 gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    <span className="text-xs font-mono font-bold text-emerald-400 tracking-wider uppercase">Forensic Dossier Generated</span>
                  </div>
                  <h2 className="text-xl font-bold text-white mt-0.5">
                    Forensic Verification Report: <span className="text-blue-400 font-mono text-base">{result.fileName}</span>
                  </h2>
                </div>
                
                <Button 
                  onClick={handleReset} 
                  className="bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs px-4 py-2 rounded-lg shadow-md transition-all flex items-center gap-2"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Inspect Another Asset
                </Button>
              </div>
              
              {/* Forensic Case Dossier */}
              <AnalysisResult result={result} />
            </section>
          )}

        </main>

        {/* Professional Academic Footer */}
        <footer className="relative w-full z-10 border-t border-slate-800/80 bg-slate-950/95 py-6 mt-16 text-center text-xs text-slate-500 font-mono">
          <div className="max-w-7xl mx-auto px-4 space-y-2">
            <p className="text-slate-300 font-semibold tracking-wide">
              PILLAI COLLEGE OF ENGINEERING (PCE AUTONOMOUS), NEW PANVEL – 410 206
            </p>
            <p className="text-slate-400">
              Department of Computer Engineering • Final Year Major Project (2026–2027)
            </p>
            <p className="text-[11px] text-slate-500">
              Reference Base: MINTIME (Multi-Identity Size-Invariant Video Deepfake Detection, IEEE TIFS 2024)
            </p>
          </div>
        </footer>

      </div>
    </div>
  );
};

export default Index;