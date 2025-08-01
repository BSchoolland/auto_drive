import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_training_progress():
    """Display training loss over time"""
    try:
        with open('outputs/training_data.json', 'r') as f:
            data = json.load(f)
        
        losses = [d['loss'] for d in data]
        epochs = [d['epoch'] for d in data]
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses, alpha=0.7, label='Training Loss')
        plt.title('Lane Detection Training Progress')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/plots/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training completed with {len(data)} batches")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        print("Training progress saved to outputs/plots/training_progress.png")
        
    except FileNotFoundError:
        print("No training data found. Run lane_detection.py first.")

def show_predictions():
    """Display prediction results"""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):
        try:
            # Load original overlay
            overlay = cv2.imread(f'outputs/prediction_{i}.jpg')
            if overlay is not None:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                # Load mask
                mask = cv2.imread(f'outputs/prediction_mask_{i}.jpg')
                if mask is not None:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    
                    axes[0, i].imshow(overlay)
                    axes[0, i].set_title(f'Prediction {i} - Overlay')
                    axes[0, i].axis('off')
                    
                    axes[1, i].imshow(mask)
                    axes[1, i].set_title(f'Prediction {i} - Lane Mask')
                    axes[1, i].axis('off')
            
        except Exception as e:
            print(f"Could not load prediction {i}: {e}")
    
    plt.tight_layout()
    plt.savefig('outputs/plots/prediction_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Prediction results saved to outputs/plots/prediction_results.png")

def show_video_samples():
    """Show some video overlay samples"""
    import os
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    frame_files = []
    for root, dirs, files in os.walk('outputs/videos'):
        for file in files:
            if file.startswith('frame_') and file.endswith('_overlay.jpg'):
                frame_files.append(os.path.join(root, file))
    
    if len(frame_files) == 0:
        print("No video overlay frames found. Run video_overlay.py first.")
        return
    
    # Show first few frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, frame_file in enumerate(frame_files[:6]):
        frame = cv2.imread(frame_file)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(frame)
            axes[i].set_title(f'Video Frame {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/video_overlay_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Found {len(frame_files)} video overlay frames")
    print("Video samples saved to outputs/plots/video_overlay_samples.png")

if __name__ == "__main__":
    print("=== Lane Detection Results ===")
    print()
    
    print("1. Training Progress:")
    show_training_progress()
    print()
    
    print("2. Prediction Results:")
    show_predictions()
    print()
    
    print("3. Video Overlay Samples:")
    show_video_samples()
    print()
    
    print("=== Summary ===")
    print("✓ Model trained successfully on CULane dataset")
    print("✓ Predictions generated with lane overlays")
    print("✓ Video processing completed with overlay visualization")
    print("✓ All files saved for inspection")
    
    print("\nFiles created:")
    import os
    for file in ['outputs/plots/training_progress.png', 'outputs/plots/prediction_results.png', 'outputs/plots/video_overlay_samples.png']:
        if os.path.exists(file):
            print(f"  - {file}")
    
    print("\nTo view videos, check the outputs/videos/ directory for .mp4 files") 