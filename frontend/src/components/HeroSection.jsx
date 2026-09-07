import React from 'react';
import { Eye, Scan, Cpu, CheckCircle2, ShieldAlert } from 'lucide-react';

export const HeroSection = () => {
  return (
    <section className="relative py-8 md:py-12 overflow-hidden w-full text-center">
      <div className="relative z-10 max-w-4xl mx-auto px-4 flex flex-col items-center space-y-6">
        
        {/* Status Tag */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-blue-500/30 bg-blue-500/10 text-xs text-blue-400 font-mono tracking-wider shadow-sm">
          <ShieldAlert className="w-3.5 h-3.5 text-blue-400" />
          <span>IEEE TIFS 2024 Base Reference • MINTIME Multi-Branch Architecture</span>
        </div>
        
        {/* Main Title */}
        <div className="space-y-3">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white">
            AI-Driven Deepfake Image &amp; Video <br className="hidden sm:inline" />
            <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-indigo-400 bg-clip-text text-transparent">
              Forensic Detection Framework
            </span>
          </h1>
          <p className="max-w-2xl mx-auto text-slate-400 text-sm sm:text-base leading-relaxed">
            Multi-branch deep learning inspection combining Spatial Blending Seam Analysis, 2D-FFT/DCT Frequency Fingerprinting, Landmark Attention Masks, and Spatio-Temporal Vision Transformers.
          </p>
        </div>

        {/* Feature Highlights Badges */}
        <div className="flex flex-wrap justify-center items-center gap-3 pt-2">
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900/90 border border-slate-800 shadow-sm text-xs font-mono text-slate-300">
            <Scan className="w-4 h-4 text-blue-400" />
            <span>Spatial Boundary &amp; Texture (92.4%)</span>
          </div>
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900/90 border border-slate-800 shadow-sm text-xs font-mono text-slate-300">
            <Eye className="w-4 h-4 text-cyan-400" />
            <span>Ocular &amp; Perioral Attention</span>
          </div>
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900/90 border border-slate-800 shadow-sm text-xs font-mono text-slate-300">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <span>Dual ViT Ensemble (FF++)</span>
          </div>
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-950/40 border border-emerald-500/30 shadow-sm text-xs font-mono text-emerald-400">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Composite Fused Accuracy: 98.4%</span>
          </div>
        </div>

      </div>
    </section>
  );
};