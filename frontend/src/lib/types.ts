
export interface DeepfakeResult {
  isDeepfake: boolean;
  confidence: number;
  processingTime: number;
  formattedResult?: string;
  frameResults?: {
    totalFrames: number;
    fakeFrames: number;
    frameConfidences?: number[];
    fakeProbability?: number;
  };
  metadata?: {
    modelName: string;
    version: string;
    mediaType: "image" | "video";
    rawPrediction?: number;
  };
}

export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

export interface FileWithPreview extends File {
  preview: string;
}
