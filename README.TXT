# 🎯 YOLO Object Detection

A real-time object detection project built with **Python, OpenCV, and Ultralytics YOLO**.

The project uses a webcam to detect objects in real time. It also supports training a custom YOLO model to recognize objects that are not included in the standard pretrained model.

## 🚀 Features

- 📷 Real-time webcam object detection
- 👤 Person detection
- 🎧 Headphones detection
- 📖 Book detection
- 🖊️ Pen detection
- 🔧 Custom YOLO model training
- 🎯 Confidence-based detection
- ⚡ Real-time processing using OpenCV
- 🏷️ Custom dataset labeling

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- LabelImg
- NumPy

## 📁 Project Structure

```text
FACE_RECOGNITION/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
│
├── main.py
├── my_objects.pt
├── yolo11n.pt
├── yolo8m.pt
├── yolo8n-face.pt
├── yolo8n-parking.pt
├── yolo8n-person.pt
│
└── myenv/
