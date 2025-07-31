import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_fake_data(batch_size, input_size, num_classes, num_batches):
    """Generate fake training data"""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size)
        y = torch.randint(0, num_classes, (batch_size,))
        data.append((x, y))
    return data

def test_speed(device_name, device):
    print(f"\nTesting on {device_name}...")
    
    # Parameters
    batch_size = 128
    input_size = 1024
    hidden_size = 512
    num_classes = 10
    num_batches = 100
    
    # Create model and move to device
    model = SimpleModel(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate fake data
    print(f"Generating {num_batches} batches of fake data...")
    train_data = generate_fake_data(batch_size, input_size, num_classes, num_batches)
    
    # Warm up
    print("Warming up...")
    for i in range(5):
        x, y = train_data[i]
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Time the training
    print("Starting speed test...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nResults for {device_name}:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per batch: {total_time/num_batches:.4f} seconds")
    print(f"Samples per second: {(batch_size * num_batches) / total_time:.2f}")
    
    return total_time

def main():
    print("PyTorch GPU Speed Test")
    print("=" * 30)
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test CPU
    cpu_time = test_speed("CPU", torch.device('cpu'))
    
    # Test GPU if available
    if torch.cuda.is_available():
        gpu_time = test_speed("GPU (CUDA)", torch.device('cuda'))
        
        print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x faster on GPU")
    else:
        print("\nGPU not available for testing")

if __name__ == "__main__":
    main() 