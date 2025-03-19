
import React from "react";
import { useDeepfake } from "@/context/DeepfakeContext";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ResultDisplayProps {
  className?: string;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ className }) => {
  const { result, status, resetState } = useDeepfake();

  if (status !== "success" || !result) {
    return null;
  }

  const { isDeepfake, confidence, processingTime, metadata } = result;
  const confidencePercentage = (confidence * 100).toFixed(2);
  
  return (
    <div className={cn("mt-8 w-full max-w-2xl mx-auto animate-slide-up", className)}>
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Analysis Results</h3>
          <span className="text-xs text-muted-foreground">
            Processed in {processingTime.toFixed(2)}s
          </span>
        </div>
        
        <div className="flex items-center justify-center my-8">
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Background circle */}
            <svg className="w-full h-full absolute" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="currentColor"
                strokeWidth="8"
                className="text-muted/20"
              />
              {/* Progress circle */}
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke={isDeepfake ? "currentColor" : "currentColor"}
                strokeWidth="8"
                strokeDasharray={`${Number(confidencePercentage) * 2.83} 283`}
                strokeLinecap="round"
                className={`progress-ring ${
                  isDeepfake ? "text-destructive" : "text-primary"
                }`}
              />
            </svg>
            <div className="text-center">
              <span className="text-4xl font-bold block">
                {confidencePercentage}%
              </span>
              <span className="text-sm block mt-1">Confidence</span>
            </div>
          </div>
        </div>
        
        <div className="text-center mb-6">
          <div 
            className={cn(
              "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium",
              isDeepfake 
                ? "bg-destructive/10 text-destructive" 
                : "bg-primary/10 text-primary"
            )}
          >
            {isDeepfake ? "Deepfake Detected" : "Authentic Image"}
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            {isDeepfake
              ? "Our model has detected manipulations in this image that indicate it's likely a deepfake."
              : "Our model indicates this image appears to be authentic without significant manipulations."}
          </p>
        </div>
        
        {metadata && (
          <div className="p-3 bg-secondary/50 rounded-lg text-xs text-muted-foreground mb-4">
            <div className="flex justify-between items-center">
              <span>Model:</span>
              <span className="font-medium">{metadata.modelName}</span>
            </div>
            <div className="flex justify-between items-center mt-1">
              <span>Version:</span>
              <span className="font-medium">{metadata.version}</span>
            </div>
          </div>
        )}
        
        <div className="flex justify-center">
          <Button onClick={resetState} variant="outline">
            Analyze Another Image
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ResultDisplay;
