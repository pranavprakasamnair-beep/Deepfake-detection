import React from 'react';
import { Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

export const AnalysisProgress = ({ progress, stage }) => {
  return (
    <div className="w-full space-y-3 animate-fade-in py-1">
      
      {/* Top Status Area */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400 animate-pulse" />
          <span className="text-xs font-mono font-medium text-slate-300 truncate max-w-[200px]">
            {stage || 'Executing pipeline...'}
          </span>
        </div>
        <span className="text-xs font-mono font-bold text-blue-400">
          {Math.round(progress)}%
        </span>
      </div>
      
      {/* Modern High-Tech Progress Bar */}
      <div className="relative h-2 w-full bg-slate-950 rounded-full border border-slate-800 overflow-hidden shadow-inner">
        <div 
          className="h-full bg-gradient-to-r from-blue-600 via-cyan-500 to-indigo-500 transition-all duration-300 ease-out rounded-full shadow-[0_0_12px_rgba(59,130,246,0.5)]"
          style={{ width: `${progress}%` }}
        />
      </div>
      
      {/* Stage Waypoints */}
      <div className="flex justify-between text-[10px] text-slate-500 font-mono">
        <span className={cn(progress >= 10 ? "text-blue-400 font-semibold" : "")}>PREPROCESS</span>
        <span className={cn(progress >= 35 ? "text-blue-400 font-semibold" : "")}>SPATIAL &amp; FREQ</span>
        <span className={cn(progress >= 75 ? "text-blue-400 font-semibold" : "")}>ATTN &amp; VIT</span>
        <span className={cn(progress >= 100 ? "text-emerald-400 font-semibold" : "")}>FUSION COMPLETE</span>
      </div>
      
    </div>
  );
};
