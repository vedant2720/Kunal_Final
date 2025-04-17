import os
import time
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn
from werkzeug.utils import secure_filename
import tempfile
import timm
from tqdm import tqdm
from PIL import Image
import face_recognition
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global model variable
model = None

# Define allowed file extensions
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'webm', 'mkv'}

# Vision Transformer for Video
class ViTVideoClassifier(nn.Module):
    def __init__(self, num_classes=2, num_frames=6, pretrained=True, memory_efficient=True):
        super(ViTVideoClassifier, self).__init__()
        
        # Use a smaller ViT model for memory efficiency
        if memory_efficient:
            # Use ViT-Small instead of ViT-Base or larger versions
            self.vit_encoder = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
            embed_dim = 384  # ViT-Small embedding dimension
        else:
            # Use ViT-Base
            self.vit_encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            embed_dim = 768  # ViT-Base embedding dimension
        
        # Remove the classification head
        self.vit_encoder.head = nn.Identity()
        
        # Create a proper temporal encoder with transformer architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2
        )
        
        # Separate CNN feature extractor for temporal features
        self.temporal_cnn = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Classifier (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Positional encoding for frames
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame with ViT encoder
        frame_features = []
        for i in range(num_frames):
            # Extract features for the current frame
            features = self.vit_encoder(x[:, i])  # [batch_size, embed_dim]
            frame_features.append(features)
        
        # Stack frame features along the temporal dimension
        x = torch.stack(frame_features, dim=1)  # [batch_size, num_frames, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply temporal encoder (Transformer)
        x = self.temporal_encoder(x)
        
        # Global temporal pooling (mean pooling across frames)
        x = torch.mean(x, dim=1)  # [batch_size, embed_dim]
        
        # Classification (Decoder)
        x = self.decoder(x)
        
        return x

def extract_frames(video_path, num_frames=6, transform=None):
    """Extract frames from a video file."""
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Calculate sampling interval to get num_frames evenly distributed
        if total_frames <= num_frames:
            # If video has fewer frames than needed, duplicate frames
            indices = list(range(total_frames)) * (num_frames // total_frames + 1)
            indices = indices[:num_frames]
        else:
            # Sample frames evenly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if transform:
                    frame = transform(image=frame)["image"]
                frames.append(frame)
            else:
                # If frame read fails, add a blank frame
                blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                if transform:
                    blank_frame = transform(image=blank_frame)["image"]
                frames.append(blank_frame)
        
        cap.release()
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        # Create dummy frames if extraction fails
        blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(num_frames):
            if transform:
                dummy = transform(image=blank_frame)["image"]
                frames.append(dummy)
            else:
                # Convert numpy to tensor if no transform
                dummy = torch.from_numpy(blank_frame.transpose(2, 0, 1)).float() / 255.0
                frames.append(dummy)
    
    # Stack frames into a tensor
    frames = torch.stack(frames)
    return frames

def predict_video(model, video_path, num_frames=6, device='cuda'):
    """Predict if a video is real or fake."""
    # Data transformations for inference
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Extract frames
    print(f"Extracting frames from {video_path}...")
    frames = extract_frames(video_path, num_frames=num_frames, transform=transform)
    
    # Add batch dimension and move to device
    frames = frames.unsqueeze(0).to(device)  # [1, num_frames, 3, 224, 224]
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(frames)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    # Convert prediction to label
    label = "Fake" if predicted_class == 1 else "Real"
    
    return label, confidence

def load_model():
    """Load the model and return it"""
    global model
    
    NUM_FRAMES = 6
    CHECKPOINT_PATH = 'models/checkpoint_epoch_25.pth'
    
    print("Loading model checkpoint...")
    # Initialize model
    model = ViTVideoClassifier(num_classes=2, num_frames=NUM_FRAMES, pretrained=False, memory_efficient=True)
    
    try:
        # Try with add_safe_globals for numpy.dtype (preferred secure method)
        try:
            from torch.serialization import add_safe_globals
            import numpy
            
            # Add all potentially needed numpy types to safe globals
            add_safe_globals([
                numpy.dtype,
                numpy.core.multiarray.scalar,
                numpy.ndarray,
                numpy._globals._NoValue,
                numpy.bool_,
                numpy.int_,
                numpy.intc,
                numpy.intp,
                numpy.int8,
                numpy.int16,
                numpy.int32,
                numpy.int64,
                numpy.uint8,
                numpy.uint16,
                numpy.uint32,
                numpy.uint64,
                numpy.float_,
                numpy.float16,
                numpy.float32,
                numpy.float64
            ])
            
            # Load checkpoint
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model checkpoint loaded with safe globals! Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        except (ImportError, AttributeError, Exception) as e:
            print(f"Safe globals method failed: {e}")
            raise
            
    except Exception:
        # Option 2: Try with weights_only=False (less secure but reliable)
        try:
            print("Attempting to load with weights_only=False...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model checkpoint loaded with weights_only=False! Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        except Exception as e:
            # Option 3: Try with pickle module directly (last resort)
            try:
                print("Attempting to load with pickle module...")
                import pickle
                with open(CHECKPOINT_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model checkpoint loaded with pickle! Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
            except Exception as e2:
                print(f"All loading methods failed. Last error: {e2}")
                print("Cannot proceed without model weights.")
                return None
    
    # Move model to device
    model = model.to(device)
    return model

def process_video_api(video_path):
    """Process a video for the API endpoint and return results."""
    global model
    
    # Load model if not already loaded
    if model is None:
        model = load_model()
        if model is None:
            return {"error": "Failed to load model"}
    
    # Constants
    NUM_FRAMES = 6
    
    try:
        # Get prediction
        start_time = time.time()
        label, confidence = predict_video(model, video_path, NUM_FRAMES, device)
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "label": label,
            "confidence": float(confidence),
            "processingTime": float(processing_time)
        }
    except Exception as e:
        print(f"Error processing video: {e}")
        return {"error": str(e)}

# ✅ Routes
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "ok", "modelLoaded": model is not None})

@app.route('/api/detect_video', methods=['POST'])
def detect_video_route():
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_video_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = process_video_api(filepath)
    
    # Clean up
    try:
        os.remove(filepath)
    except:
        pass
        
    return jsonify(result)

# Load model on startup
if __name__ == '__main__':
    # Initialize model at startup
    load_model()
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5272)
else:
    # When imported as a module, load model
    load_model()