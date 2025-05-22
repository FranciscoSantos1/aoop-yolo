from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

def detect_risky_behavior(image_path):
    results = model(image_path)
    
    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = box.conf.item()
            
    
            if confidence > 0.5:
                if class_name == 'No_Seatbelt':
                    print(f"🚨 WARNING: Seatbelt not detected! (Confidence: {confidence:.2f})")
                elif class_name == 'Looking_Sideways':
                    print(f"🚨 WARNING: Driver looking sideways! (Confidence: {confidence:.2f})")
                elif class_name == 'Phone_Usage':
                    print(f"🚨 WARNING: Phone usage detected! (Confidence: {confidence:.2f})")


detect_risky_behavior('path/to/your/image.jpg')
