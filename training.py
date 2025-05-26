from ultralytics import YOLO


# Check if CUDA is available
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')
model.train(data='Driver-Behavior-4/data.yaml', epochs=50, batch=16, device=device)
