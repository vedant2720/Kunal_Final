import os
import time
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import tempfile
from transformers import ViTImageProcessor
from PIL import Image
from io import BytesIO


# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/cnn_vit_hybrid.pth')



try:
    from train_cnn_vit import HybridCNNViT
    model = HybridCNNViT(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Hybrid CNN-ViT model loaded successfully")
    
    # Load image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    print("Image processor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# ✅ Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# ✅ Helper functions
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def predict_frame(frame):
    frame = transform(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(frame)
        probabilities = torch.softmax(logits, dim=1)
        fake_prob = probabilities[0, 1].item()
    
    return fake_prob

def predict_image(image_path):
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image file"}, 400
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fake_prob = predict_frame(image)
    processing_time = time.time() - start_time
    
    return {
        "isDeepfake": bool(fake_prob > 0.5),
        "confidence": float(fake_prob),
        "processingTime": float(processing_time),
        "metadata": {
            "modelName": "HybridCNNViT",
            "version": "1.0",
            "mediaType": "image"
        }
    }

def predict_video(video_path):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Could not open video file"}, 400
    
    fake_probs = []
    total_frames = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        if total_frames % 5 == 0:
            fake_prob = predict_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fake_probs.append(fake_prob)
            processed_frames += 1
    
    cap.release()
    
    if not fake_probs:
        return {"error": "No frames could be processed in the video"}, 400
    
    avg_fake_prob = np.mean(fake_probs)
    is_deepfake = avg_fake_prob > 0.5
    processing_time = time.time() - start_time
    
    return {
        "isDeepfake": bool(is_deepfake),
        "confidence": float(avg_fake_prob),
        "processingTime": float(processing_time),
        "frameResults": {
            "totalFrames": processed_frames,
            "fakeFrames": sum(1 for p in fake_probs if p > 0.5),
            "fakeProbability": float(avg_fake_prob)
        },
        "metadata": {
            "modelName": "HybridCNNViT",
            "version": "1.0",
            "mediaType": "video"
        }
    }

# ✅ Routes
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({"status": "ok", "modelLoaded": model is not None})

@app.route('/api/detect', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_image_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = predict_image(filepath)
    os.remove(filepath)
    return jsonify(result)

@app.route('/api/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400
    
    file = request.files['video']
    if file.filename == '' or not allowed_video_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = predict_video(filepath)
    os.remove(filepath)
    return jsonify(result)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5272)
