
export interface DeepfakeResult {
  isDeepfake: boolean;
  confidence: number;
  processingTime: number;
  frameResults?: {
    totalFrames: number;
    fakeFrames: number;
    frameConfidences?: number[];
  };
  metadata?: {
    modelName: string;
    version: string;
    mediaType: "image" | "video";
  };
}

export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

export interface FileWithPreview extends File {
  preview: string;
}
