import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# --------------------------
# ✅ Load ViT image processor for normalization
# --------------------------
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# ✅ Define image transformation (Resize, Normalize, Convert to Tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)  # Normalize
])


# --------------------------
# ✅ Video Frame Dataset
# --------------------------
class VideoFrameDataset(Dataset):
    """Custom dataset for extracting frames from videos (real/fake)."""
    
    def __init__(self, root_dir, transform=None, frame_skip=5):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip  
        self.data = []

        # ✅ Label Mapping (Real = 0, Fake = 1)
        self.class_mapping = {"real": 0, "fake": 1}

        # ✅ Load videos and extract frames
        for label in ["real", "fake"]:
            class_dir = os.path.join(root_dir, label)
            if not os.path.exists(class_dir):
                continue
            
            for video_name in os.listdir(class_dir):
                if not video_name.endswith(('.mp4', '.avi', '.mov')):  
                    continue
                
                video_path = os.path.join(class_dir, video_name)
                self.data.extend(self._extract_frames(video_path, self.class_mapping[label]))

    def _extract_frames(self, video_path, label):
        """Extract frames from video and assign label."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Warning: Cannot open {video_path}")
            return []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip == 0:  
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
                frame_pil = Image.fromarray(frame)  # Convert NumPy array to PIL Image

                try:
                    if self.transform:
                        frame_tensor = self.transform(frame_pil)  # ✅ Apply transformation
                        frames.append((frame_tensor, torch.tensor(label, dtype=torch.long)))  # ✅ Ensure label is tensor

                except Exception as e:
                    print(f"⚠️ Error processing frame from {video_path}: {e}")

        cap.release()
        return frames

    def __len__(self):
        return len(self.data)  # ✅ Return dataset size

    def __getitem__(self, idx):
        return self.data[idx]  # ✅ Ensure it returns (frame_tensor, label)


# --------------------------
# ✅ CNN Feature Extractor (ResNet-18)
# --------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.shape[0], -1)  


# --------------------------
# ✅ Hybrid CNN + ViT Model
# --------------------------
class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridCNNViT, self).__init__()
        
        self.cnn_extractor = ResNetFeatureExtractor()
        cnn_feature_size = 512  

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_feature_size = self.vit.config.hidden_size  # 768 for ViT

        self.fc = nn.Linear(cnn_feature_size + vit_feature_size, num_classes)

    def forward(self, x):
        # CNN Feature Extraction
        cnn_features = self.cnn_extractor(x)  

        # ViT Feature Extraction
        vit_outputs = self.vit(x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]  # ✅ Extract CLS Token

        # Combine CNN & ViT Features
        combined_features = torch.cat((cnn_features, vit_features), dim=1)  
        logits = self.fc(combined_features)  
        return logits


# --------------------------
# ✅ Training Pipeline
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load Dataset
    data_dir = "dataset/"  
    train_dataset = VideoFrameDataset(root_dir=data_dir, transform=transform)  
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"✅ Loaded {len(train_dataset)} frames from videos in '{data_dir}'")
    print(f"ℹ️ Classes: {train_dataset.class_mapping}")
    # ✅ Ensure dataset is not empty
    if len(train_dataset) == 0:
        raise ValueError("❌ No frames extracted! Check dataset structure or OpenCV video reading.")

    # ✅ Initialize Model
    model = HybridCNNViT(num_classes=2).to(device)

    # ✅ Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # ✅ Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"🔁 Epoch [{epoch+1}/{num_epochs}]")
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"✅ Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

    # ✅ Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_vit_hybrid.pth")
    print("✅ Hybrid Model saved!")
