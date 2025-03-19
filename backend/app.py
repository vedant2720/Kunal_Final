
import os
import time
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Load model (global variables for efficiency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to saved model
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/vit_model.pth')

# Initialize model and processor
try:
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    print("Image processor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Helper functions
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def predict_image(image_path):
    """Predict if an image is real or fake."""
    start_time = time.time()
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image file"}, 400
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0, 1].item()  # Probability of being fake
            is_deepfake = confidence > 0.5
        
        processing_time = time.time() - start_time
        
        return {
            "isDeepfake": bool(is_deepfake),
            "confidence": float(confidence),
            "processingTime": float(processing_time),
            "metadata": {
                "modelName": "ViT-Base-Patch16-224",
                "version": "1.0",
                "mediaType": "image"
            }
        }
    except Exception as e:
        print(f"Error predicting image: {e}")
        return {"error": str(e)}, 500

def predict_frame(frame):
    """Predict if a single frame is real or fake."""
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])
        
        frame_tensor = transform(frame).unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(frame_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0, 1].item()  # Probability of being fake
        
        return confidence  # Probability of being fake
    except Exception as e:
        print(f"Error predicting frame: {e}")
        return None

def predict_video(video_path):
    """Predict if a video is real or fake by analyzing frames."""
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Check if video was opened successfully
        if not cap.isOpened():
            return {"error": "Could not open video file"}, 400
        
        frame_confidences = []
        total_frames = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:  # No more frames
                break
            
            total_frames += 1
            
            # Process every 5th frame for efficiency
            if total_frames % 5 == 0:
                confidence = predict_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if confidence is not None:
                    frame_confidences.append(confidence)
                    processed_frames += 1
        
        cap.release()
        
        # Handle case where no frames were processed
        if not frame_confidences:
            return {"error": "No frames could be processed in the video"}, 400
        
        # Compute average confidence and count fake frames
        avg_confidence = np.mean(frame_confidences)
        fake_frames = sum(1 for conf in frame_confidences if conf > 0.5)
        is_deepfake = avg_confidence > 0.5
        
        processing_time = time.time() - start_time
        
        return {
            "isDeepfake": bool(is_deepfake),
            "confidence": float(avg_confidence),
            "processingTime": float(processing_time),
            "frameResults": {
                "totalFrames": processed_frames,
                "fakeFrames": fake_frames,
                "frameConfidences": [float(conf) for conf in frame_confidences]
            },
            "metadata": {
                "modelName": "ViT-Base-Patch16-224",
                "version": "1.0",
                "mediaType": "video"
            }
        }
    except Exception as e:
        print(f"Error predicting video: {e}")
        return {"error": str(e)}, 500

# Routes
@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API is running."""
    return jsonify({"status": "ok", "modelLoaded": model is not None})

@app.route('/api/detect', methods=['POST'])
def detect_image():
    """Endpoint to detect deepfakes in images."""
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_image_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
        
        if isinstance(result, tuple) and len(result) == 2:
            return jsonify(result[0]), result[1]
        return jsonify(result)
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/detect_video', methods=['POST'])
def detect_video_route():
    """Endpoint to detect deepfakes in videos."""
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_video(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
        
        if isinstance(result, tuple) and len(result) == 2:
            return jsonify(result[0]), result[1]
        return jsonify(result)
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
