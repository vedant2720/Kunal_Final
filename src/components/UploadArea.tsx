
import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useDeepfake } from "@/context/DeepfakeContext";
import { cn } from "@/lib/utils";

interface UploadAreaProps {
  className?: string;
}

const UploadArea: React.FC<UploadAreaProps> = ({ className }) => {
  const { file, setFile, status, analyzeImage } = useDeepfake();
  const [uploadProgress, setUploadProgress] = useState(0);

  // Simulate upload progress
  React.useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    
    if (status === "uploading") {
      interval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = prev + Math.random() * 15;
          return newProgress >= 100 ? 100 : newProgress;
        });
      }, 300);
    } else if (status === "processing") {
      setUploadProgress(100);
    } else {
      setUploadProgress(0);
    }
    
    return () => clearInterval(interval);
  }, [status]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, [setFile]);

  const { 
    getRootProps, 
    getInputProps, 
    isDragActive,
    isDragAccept,
    isDragReject
  } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxSize: 10485760, // 10MB
    multiple: false,
    disabled: status === "uploading" || status === "processing"
  });

  const getDropzoneClassName = () => {
    if (isDragActive && isDragAccept) return "dropzone dropzone-active";
    if (isDragReject) return "dropzone dropzone-error";
    return "dropzone dropzone-idle";
  };

  return (
    <div className={cn("w-full max-w-2xl mx-auto", className)}>
      <div 
        {...getRootProps()} 
        className={cn(
          "p-6 flex flex-col items-center justify-center transition-all duration-300 animate-fade-in",
          getDropzoneClassName(),
          status === "uploading" || status === "processing" ? "opacity-70 pointer-events-none" : ""
        )}
      >
        <input {...getInputProps()} />
        
        <div className="w-16 h-16 mb-4 flex items-center justify-center rounded-full bg-primary/10">
          {!file ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-8 h-8 text-primary"
            >
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7" />
              <line x1="16" x2="22" y1="5" y2="5" />
              <line x1="19" x2="19" y1="2" y2="8" />
              <circle cx="9" cy="9" r="2" />
              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-8 h-8 text-primary"
            >
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7" />
              <path d="m9 11 3 3L22 4" />
            </svg>
          )}
        </div>
        
        <div className="text-center space-y-2">
          <h3 className="text-lg font-semibold">
            {isDragActive
              ? "Drop the image here"
              : file
              ? "Image ready for analysis"
              : "Upload an image"}
          </h3>
          <p className="text-sm text-muted-foreground max-w-xs">
            {file
              ? `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`
              : "Drag and drop or click to select an image (JPEG, PNG, WebP)"}
          </p>
        </div>
        
        {(status === "uploading" || status === "processing") && (
          <div className="w-full mt-4 space-y-2">
            <Progress value={uploadProgress} className="h-2" />
            <p className="text-xs text-center text-muted-foreground">
              {status === "uploading" ? "Uploading image..." : "Analyzing image..."}
            </p>
          </div>
        )}
        
        {file && status !== "uploading" && status !== "processing" && (
          <div className="mt-4 flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                setFile(null);
              }}
            >
              Remove
            </Button>
            <Button 
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                analyzeImage();
              }}
            >
              Analyze Image
            </Button>
          </div>
        )}
      </div>
      
      {file?.preview && status !== "uploading" && (
        <div className="mt-6 w-full overflow-hidden rounded-lg border animate-scale-in">
          <img
            src={file.preview}
            alt="Preview"
            className="w-full h-auto object-contain"
            style={{ maxHeight: "300px" }}
          />
        </div>
      )}
    </div>
  );
};

export default UploadArea;
