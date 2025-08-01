import torch
import cv2
import numpy as np
import os
import argparse
from lane_detection import LaneDetectionCNN, create_transform

def load_model(model_path, num_classes=5):
    """Load trained model"""
    model = LaneDetectionCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_frame(model, frame, transform=None):
    """Predict lane segmentation for a single frame"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_size = frame_rgb.shape[:2]
    
    # Resize to model input size (same as training)
    img_size = (96, 256)
    frame_resized = cv2.resize(frame_rgb, (img_size[1], img_size[0]))
    
    # Convert to tensor and normalize
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
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
        0: [0, 0, 0],      # Background - transparent
        1: [0, 0, 255],    # Lane 1 - red
        2: [0, 255, 0],    # Lane 2 - green  
        3: [255, 0, 0],    # Lane 3 - blue
        4: [0, 255, 255]   # Lane 4 - yellow
    }
    
    # Create colored mask
    colored_mask = np.zeros((*pred_mask_resized.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        if class_id > 0:  # Skip background
            colored_mask[pred_mask_resized == class_id] = color
    
    # Overlay on original frame (BGR format)
    overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
    
    return overlay, pred_mask_resized

def process_video_folder(model, video_folder, output_folder, max_frames=100):
    """Process all images in a video folder and create overlay video"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all jpg files and sort them
    image_files = [f for f in os.listdir(video_folder) if f.endswith('.jpg')]
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"No jpg files found in {video_folder}")
        return
    
    # Limit frames for quick demo
    image_files = image_files[:max_frames]
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(video_folder, image_files[0]))
    height, width = first_image.shape[:2]
    
    # Create video writer
    video_name = os.path.basename(video_folder)
    output_video_path = os.path.join(output_folder, f"{video_name}_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))
    
    print(f"Processing {len(image_files)} frames from {video_folder}")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(video_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            continue
            
        # Predict and overlay
        overlay_frame, pred_mask = predict_frame(model, frame)
        
        # Write frame to video
        video_writer.write(overlay_frame)
        
        # Save a few individual frames for inspection
        if i < 5:
            cv2.imwrite(os.path.join(output_folder, f"frame_{i:03d}_overlay.jpg"), overlay_frame)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} frames")
    
    video_writer.release()
    print(f"Video saved to: {output_video_path}")

def process_images_folder(model, images_folder, output_folder):
    """Process images from a folder structure and create overlays"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all video folders
    video_folders = []
    for root, dirs, files in os.walk(images_folder):
        # Check if this folder contains jpg files
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if jpg_files:
            video_folders.append(root)
    
    print(f"Found {len(video_folders)} video folders")
    
    # Process first few folders for demo
    for i, video_folder in enumerate(video_folders[:3]):
        print(f"\nProcessing folder {i+1}/{min(3, len(video_folders))}: {video_folder}")
        folder_name = os.path.basename(video_folder)
        folder_output = os.path.join(output_folder, folder_name)
        process_video_folder(model, video_folder, folder_output, max_frames=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay lane detection on video')
    parser.add_argument('--model', default='outputs/models/lane_detection_model.pth', help='Path to trained model')
    parser.add_argument('--input', default='culane_dataset/driver_23_30frame', help='Input folder with video frames')
    parser.add_argument('--output', default='outputs/videos', help='Output folder for videos')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Please train the model first by running: python lane_detection.py")
        exit(1)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Process videos
    print(f"Processing videos from: {args.input}")
    process_images_folder(model, args.input, args.output)
    
    print("Video processing complete!") 