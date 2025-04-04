# PYTORCH YOLO Model

## Description
This repository contains a pytorch implementation of a YOLO-like model for object detection.
It supports training and model inference.

This project focuses on object detection using the YOLO (You Only Look Once) model, a popular Convolutional Neural Network (CNN) architecture known for its speed and accuracy. The process involved two major steps: starting with a pre-trained model on the COCO dataset (which includes 80 common object classes), and then fine-tuning it using a small custom dataset tailored to specific use cases.

🏷️ Custom Classes & Use Case
The model was trained to detect both common objects like “person” and “plant”, as well as custom categories like:

logo

selected

unselected

To keep things simple and focus on proof of concept, I intentionally limited the number of custom classes. However, the project is scalable, and I plan to expand the categories and improve accuracy in future iterations.

🗂️ Data Preparation
Creating the dataset was one of the more time-consuming parts of the project. I used Makesense.ai, a free, browser-based annotation tool, to label the images. The annotated data was exported in YOLO format, where each .txt file corresponds to an image and contains one line per object:

<class_id> <x_center> <y_center> <width> <height>
All values are normalized between 0 and 1.

⚙️ Model Setup and Preprocessing
The model setup involved integrating both the COCO-pretrained weights and my fine-tuned layers into the YOLOv5 framework. All training images were resized to 640x640 pixels to meet YOLOv5's input requirements, while maintaining their aspect ratio.

🏋️ Training Configuration
The model was trained for 160 epochs with a batch size of 8. While training, I applied multiple data augmentation techniques to enhance the dataset and reduce overfitting.


## Installation
```sh
python3.9 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

## Training
```sh
python scripts/train.py
```

## Inference
```sh
python scripts/inference.py
```
