
# Deepfake Detection Backend

This Flask application serves as the backend for the deepfake detection web application.

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained model file in the `models` directory:
   - The model should be named `vit_model.pth` or set the environment variable `MODEL_PATH` to specify a different path.

## Running the Server

Start the Flask development server:
```bash
python app.py
```

The server will run on http://localhost:5000

## API Endpoints

- `GET /api/status` - Check if the API server is running
- `POST /api/detect` - Analyze an image for deepfake detection
- `POST /api/detect_video` - Analyze a video for deepfake detection

## Model Information

The backend uses a ViT (Vision Transformer) model trained for deepfake detection. The model requires PyTorch and the Transformers library.
