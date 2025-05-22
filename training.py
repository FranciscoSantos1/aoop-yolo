from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='Abnormal-Driver-Behaviour-10/data.yaml', epochs=50, imgsz=640)
