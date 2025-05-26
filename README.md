# AOOP YOLO Risky Behavior Detection

This project uses a YOLO model to detect risky driver behaviors (smoking, drinking, eating, phone usage, and seatbelt usage) in car images.

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV (`cv2`)

Install dependencies:
```bash
pip install ultralytics opencv-python
```

## Setup

1. **Clone this repository** and navigate to the project folder.

2. **Model Weights**  
   The YOLO weights trained from my dataset are in:
   ```
   runs/detect/train4/weights/best.pt
   ```

3. **Images**  
   The test images are in `my_car_photos` folder.

## Usage

Run the detection script:
```bash
python detect_risky_behavior.py
```

The script will:
- Process each image in `my_car_photos`
- Print warnings for detected risky behaviors
- Show annotated images with detections



---

**Francisco Santos - AOOP 24/25**