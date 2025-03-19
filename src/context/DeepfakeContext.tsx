
import React, { createContext, useContext, useState, useMemo, useCallback } from "react";
import { FileWithPreview, DeepfakeResult, UploadStatus } from "@/lib/types";
import { detectDeepfake } from "@/services/api";
import { toast } from "@/components/ui/use-toast";

interface DeepfakeContextType {
  file: FileWithPreview | null;
  result: DeepfakeResult | null;
  status: UploadStatus;
  error: string | null;
  setFile: (file: File | null) => void;
  analyzeImage: () => Promise<void>;
  resetState: () => void;
  isApiConnected: boolean;
  setIsApiConnected: (connected: boolean) => void;
}

const DeepfakeContext = createContext<DeepfakeContextType | undefined>(undefined);

export function DeepfakeProvider({ children }: { children: React.ReactNode }) {
  const [file, setFileState] = useState<FileWithPreview | null>(null);
  const [result, setResult] = useState<DeepfakeResult | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isApiConnected, setIsApiConnected] = useState<boolean>(false);

  const setFile = useCallback((file: File | null) => {
    if (file) {
      // Create preview URL for the file
      const preview = URL.createObjectURL(file);
      setFileState({ ...file, preview });
      setStatus("idle");
      setError(null);
      setResult(null);
    } else {
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
      setFileState(null);
    }
  }, []);

  const analyzeImage = useCallback(async () => {
    if (!file) {
      setError("Please select an image to analyze");
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please select an image to analyze",
      });
      return;
    }

    if (!isApiConnected) {
      setError("API server is not connected");
      toast({
        variant: "destructive",
        title: "Server Error",
        description: "Cannot connect to the analysis server",
      });
      return;
    }

    setStatus("uploading");
    setError(null);

    try {
      // Simulate upload delay (remove in production)
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setStatus("processing");
      
      // Call the actual API
      const result = await detectDeepfake(file);
      
      setResult(result);
      setStatus("success");
      
      toast({
        title: "Analysis Complete",
        description: "The image has been successfully analyzed",
      });
    } catch (err) {
      console.error("Analysis error:", err);
      setStatus("error");
      const errorMessage = err instanceof Error ? err.message : "Failed to analyze image";
      setError(errorMessage);
      
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: errorMessage,
      });
    }
  }, [file, isApiConnected]);

  const resetState = useCallback(() => {
    if (file?.preview) {
      URL.revokeObjectURL(file.preview);
    }
    setFileState(null);
    setResult(null);
    setStatus("idle");
    setError(null);
  }, [file]);

  // Clean up URL object when component unmounts
  React.useEffect(() => {
    return () => {
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
    };
  }, [file]);

  const value = useMemo(
    () => ({
      file,
      result,
      status,
      error,
      setFile,
      analyzeImage,
      resetState,
      isApiConnected,
      setIsApiConnected,
    }),
    [file, result, status, error, setFile, analyzeImage, resetState, isApiConnected, setIsApiConnected]
  );

  return <DeepfakeContext.Provider value={value}>{children}</DeepfakeContext.Provider>;
}

export function useDeepfake() {
  const context = useContext(DeepfakeContext);
  if (context === undefined) {
    throw new Error("useDeepfake must be used within a DeepfakeProvider");
  }
  return context;
}
