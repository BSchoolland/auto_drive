import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import os
import json
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from server import start_server_thread

# Global variables for tracking
training_data = []
last_save_time = 0

class CULaneDataset(Dataset):
    def __init__(self, data_root, list_file, transform=None, img_size=(96, 256)):
        self.data_root = data_root
        self.transform = transform
        self.img_size = img_size
        
        # Read image paths from list file
        with open(list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Filter to only include images we have both image and mask for
        self.valid_pairs = []
        for img_path in self.image_paths:
            # Convert jpg path to png mask path
            mask_path = img_path.replace('.jpg', '.png')
            mask_full_path = os.path.join(data_root, 'laneseg_label_w16' + mask_path)
            img_full_path = os.path.join(data_root, img_path.lstrip('/'))
            
            if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
                self.valid_pairs.append((img_full_path, mask_full_path))
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize both to same size (much smaller for speed)
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            # Apply transform to image
            image = self.transform(image)
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class LaneDetectionCNN(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
        # Simpler, smaller encoder (downsampling)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Center
        self.center = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.final = nn.Conv2d(32, num_classes, 1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        # Center
        center = self.center(enc2_pool)
        
        # Decoder with skip connections
        dec2 = self.up2(center)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        output = self.final(dec1)
        return output
    
    def training_step(self, batch, batch_idx):
        global training_data, last_save_time
        
        images, masks = batch
        outputs = self.forward(images)
        
        # Compute loss
        loss = F.cross_entropy(outputs, masks)
        
        # Track training data
        current_time = time.time()
        training_data.append({
            "epoch": self.current_epoch,
            "batch": batch_idx,
            "loss": loss.item(),
            "timestamp": current_time
        })
        
        # Save periodically
        if current_time - last_save_time >= 1.0:
            with open("outputs/training_data.json", "w") as f:
                json.dump(training_data, f)
            last_save_time = current_time
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # Higher learning rate for faster convergence

def create_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_and_visualize(model, image_path, mask_path=None):
    """Make prediction and create visualization"""
    model.eval()
    img_size = (96, 256)  # Same size as training
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (img_size[1], img_size[0]))
    
    # Convert to tensor and normalize
    input_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    
    # Resize prediction back to original size
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                   (original_size[1], original_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    
    # Create colored visualization
    colors = {
        0: [0, 0, 0],      # Background - black
        1: [255, 0, 0],    # Lane 1 - red
        2: [0, 255, 0],    # Lane 2 - green  
        3: [0, 0, 255],    # Lane 3 - blue
        4: [255, 255, 0]   # Lane 4 - yellow
    }
    
    colored_mask = np.zeros((*pred_mask_resized.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_mask[pred_mask_resized == class_id] = color
    
    # Overlay on original image
    overlay = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
    
    return overlay, pred_mask_resized, colored_mask

if __name__ == "__main__":
    # Start server for live training visualization
    server_thread = start_server_thread(training_data, port=8001)
    print("Training visualization server started. Visit http://localhost:8001 to see live training progress")
    
    # Create dataset and dataloader
    transform = create_transform()
    dataset = CULaneDataset('culane_dataset', 'culane_dataset/list/train.txt', transform=transform)
    
    # Use a very small subset for quick testing
    subset_size = min(100, len(dataset))  # Much smaller for fast testing
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=True, num_workers=0)  # Larger batch, no multiprocessing
    
    # Create model
    model = LaneDetectionCNN(num_classes=5)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=100,  # A small number of epochs for 10 minute test
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator='auto'
    )
    
    print(f"Training on {len(subset_dataset)} samples with image size 96x256")
    
    # Train the model
    trainer.fit(model, dataloader)
    
    # Save model
    torch.save(model.state_dict(), 'outputs/models/lane_detection_model.pth')
    
    # Final save of training data
    with open("outputs/training_data.json", "w") as f:
        json.dump(training_data, f)
    
    print("Training complete! Making test predictions...")
    
    # Test on a few samples
    test_samples = dataset.valid_pairs[:3]
    for i, (img_path, mask_path) in enumerate(test_samples):
        overlay, pred_mask, colored_mask = predict_and_visualize(model, img_path, mask_path)
        
        # Save visualization
        cv2.imwrite(f'outputs/prediction_{i}.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'outputs/prediction_mask_{i}.jpg', colored_mask)
        print(f"Saved prediction_{i}.jpg and prediction_mask_{i}.jpg")
    
    print("Training and predictions complete!")
    print("Server will continue running. Visit http://localhost:8001 to see training progress")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...") 