import cv2
import os
from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt')

image_folder = 'my_car_photos'

RISKY_BEHAVIORS = {
    'Cigarette': '🚨 WARNING: Smoking detected!',
    'Drinking': '🚨 WARNING: Drinking detected!',
    'Eating': '🚨 WARNING: Eating detected!',
    'Phone': '🚨 WARNING: Phone usage detected!',
    'Seatbelt': '✅ Seatbelt detected!',
}

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[ERROR] Could not open image: {filename}")
            continue

        image_resized = cv2.resize(image, (640, 640))

        results = model.predict(source=image_resized, conf=0.4)[0]

        detected_classes = set()
        for box in results.boxes:
            class_id = int(box.cls.item())
            class_name = results.names[class_id]
            confidence = box.conf.item()

            if confidence > 0.4:
                detected_classes.add(class_name)
                if class_name in RISKY_BEHAVIORS:
                    print(f"{filename} ➤ {RISKY_BEHAVIORS[class_name]} (Confidence: {confidence:.2f})")

        if 'Seatbelt' not in detected_classes:
            print(f"{filename} ➤ 🚨 WARNING: No seatbelt detected!")

        annotated_img = results.plot()
        cv2.imshow(f'Detection - {filename}', annotated_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("✅ Verification completed.")
