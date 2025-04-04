import torch
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, coco_model="yolov5n.pt", custom_model="training/fine_tuned_model4/weights/best.pt", conf_threshold=0.3):
        """
        Initialize YOLO models for inference.
        :param coco_model: Path to the YOLO model trained on COCO (for "person").
        :param custom_model: Path to the fine-tuned model (for "logo", "selected", "unselected").
        :param conf_threshold: Confidence threshold for filtering detections.
        """
        self.coco_model = YOLO(coco_model)  # Load COCO model (for detecting all classes)
        self.custom_model = YOLO(custom_model)  # Load fine-tuned model (for "logo", "selected", "unselected")
        self.conf_threshold = conf_threshold

        # Merge COCO classes and custom classes with unique indices
        num_coco_classes = len(self.coco_model.names)
        self.class_names = {i: name for i, name in self.coco_model.names.items()}  # Load COCO class names
        
        # Shift custom class IDs to avoid conflicts
        self.custom_class_offset = num_coco_classes  # Offset custom classes
        for idx, class_name in self.custom_model.names.items():
            self.class_names[self.custom_class_offset + idx] = class_name  # Add new classes

        print(f"✅ Loaded {len(self.class_names)} classes (COCO + Custom)")

    def preprocess_image(self, image):
        """
        Resize image while maintaining aspect ratio and pad to 640x640.
        :param image: Input image.
        :return: Preprocessed image ready for inference.
        """
        h, w, _ = image.shape
        scale = 640 / max(h, w)  # Scale factor to fit within 640x640
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize while maintaining aspect ratio
        resized_image = cv2.resize(image, (new_w, new_h))

        # Create a 640x640 black canvas
        padded_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Align top instead of centering
        pad_x = (640 - new_w) // 2
        pad_y = 0  # Align top
        padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image

        return padded_image, (pad_x, pad_y, scale)

    def run_on_image(self, image_path, save_output=True, output_path="output_image.png"):
        """
        Runs YOLO inference to detect all COCO classes and custom classes in an image.
        :param image_path: Path to the input image.
        :param save_output: Whether to save the output image.
        :param output_path: Path to save the output image.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Image '{image_path}' not found!")
            return

        # Preprocess image
        image_resized, (pad_x, pad_y, scale) = self.preprocess_image(image)

        # Perform inference
        coco_results = self.coco_model(image_resized)  # Detect all COCO classes
        custom_results = self.custom_model(image_resized)  # Detect custom classes

        results = [coco_results, custom_results]  # Combine results from both models

        for model_results in results:
            for result in model_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0].item()  # Confidence score
                    class_id = int(box.cls[0].item())  # Predicted class ID

                    if model_results == custom_results:  # Adjust class IDs from custom model
                        class_id += self.custom_class_offset
                    
                    if confidence >= self.conf_threshold:
                        # Convert back to original image scale
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)

                        class_name = self.class_names.get(class_id, "Unknown")

                        # Print detected class and its label for debugging
                        print(f"Detected ID: {class_id}, Mapped Label: {class_name}")

                        # Label and draw bounding box
                        color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save output image
        if save_output:
            cv2.imwrite(output_path, image)
            print(f"✅ Output saved at {output_path}")

        return image

    def extract_predictions(self, image_path):
        """
        Extracts all detections ('person', 'logo', 'selected', 'unselected') from an image.
        :param image_path: Path to the input image.
        :return: List of detections (class_name, confidence, bbox).
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Image '{image_path}' not found!")
            return []

        # Preprocess image
        image_resized, (pad_x, pad_y, scale) = self.preprocess_image(image)

        # Run inference on both models
        coco_results = self.coco_model(image_resized)  # Detect all COCO classes
        custom_results = self.custom_model(image_resized)  # Detect custom classes

        detections = []

        # Extract detections from both models
        for model_results in [coco_results, custom_results]:
            for result in model_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    if model_results == custom_results:  # Adjust class IDs from custom model
                        class_id += self.custom_class_offset
                    
                    if confidence >= self.conf_threshold:
                        # Convert back to original image scale
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)

                        class_name = self.class_names.get(class_id, "Unknown")
                        detections.append((class_name, confidence, [x1, y1, x2, y2]))

        return detections


# Run inference
if __name__ == "__main__":
    yolo = YOLOInference(
        coco_model="yolov5n.pt",  # COCO model for detecting all classes
        custom_model="training/fine_tuned_model4/weights/best.pt",  # Fine-tuned model for "logo", "selected", "unselected"
        conf_threshold=0.3
    )

    # Inference on a single image
    image_path = "dataset/images/val/Screenshot 2025-03-14 at 10.20.15 PM.png"
    yolo.run_on_image(image_path, save_output=True, output_path="output_Screenshot.png")

    # Print extracted predictions
    detections = yolo.extract_predictions(image_path)

    if detections:
        for obj in detections:
            class_label, confidence, bbox = obj  # Extract values
            print(f"Label: {class_label}, Confidence: {confidence:.2f}, BBox: {bbox}")
    else:
        print("❌ No detections found.")
