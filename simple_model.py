import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
import time
from server import start_server_thread

# XOR
inputs = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
outputs = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Global variables for loss tracking
loss_data = []
last_save_time = 0

# Define model
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2,4), nn.Sigmoid(), nn.Linear(4,1), nn.Sigmoid())

    def forward(self, inputs):
        return self.model(inputs) 

    def training_step(self, batch, _index):
        global loss_data, last_save_time
        
        inputs, expected_outputs = batch
        inference_outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(inference_outputs, expected_outputs)
        
        # Track loss with timestamp
        current_time = time.time()
        loss_data.append({
            "epoch": self.current_epoch,
            "loss": loss.item(),
            "timestamp": current_time
        })
        # time.sleep(0.01) # simulate a slow training process
        
        # Save to JSON if at least 1 second has passed
        if current_time - last_save_time >= 1.0:
            with open("loss_data.json", "w") as f:
                json.dump(loss_data, f)
            last_save_time = current_time
        
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1)

# Start HTTP server in background thread
server_thread = start_server_thread(loss_data)

# load the data
dataset = torch.utils.data.TensorDataset(inputs, outputs)

# batch the data and ready it for training
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

# create the model object
model = MyModel()
trainer = pl.Trainer(max_epochs=10000, logger=False, enable_checkpointing=False, enable_progress_bar=False)

print("Training started. Visit http://localhost:8000 to see live loss graph")

# epic training arc
trainer.fit(model, dataloader)

# Final save
with open("loss_data.json", "w") as f:
    json.dump(loss_data, f)

# test the model after training
print('testing against the training data (do not do this at home)')

# turn off gradient calculation to be faster
with torch.no_grad():
    for i in range(len(inputs)):
        pred = model(inputs[i]).item()
        print(f'{inputs[i].tolist()} -> {pred} (expected {outputs[i].item()})')

print("Training complete. Server will continue running until you stop the program.")
print("Visit http://localhost:8000 to see the final loss graph")

# Keep the server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")

    