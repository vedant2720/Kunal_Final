
export interface DeepfakeResult {
  isDeepfake: boolean;
  confidence: number;
  processingTime: number;
  metadata?: {
    modelName: string;
    version: string;
  };
}

export type UploadStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

export interface FileWithPreview extends File {
  preview: string;
}
