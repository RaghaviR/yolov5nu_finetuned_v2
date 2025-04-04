from ultralytics import YOLO

class ModelTrainer:
    @staticmethod
    def train(dataset_path='dataset/data.yaml', epochs=160, batch_size=8, model_path='models/yolov5nu.pt'):
        print(f"🚀 Fine-tuning YOLOv5 on dataset: {dataset_path} for {epochs} epochs")
        
        # Step 1: Load the pre-trained COCO model
        model = YOLO(model_path)  # This is the pre-trained YOLOv5 model on COCO
        
        # Step 2: Fine-tune the model with the provided dataset and configurations
        model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,  # Image size remains unchanged
            batch=batch_size,
            project="training",  # Save training logs and outputs here
            name="fine_tuned_model",  # Name of the fine-tuned model
            save=True,  # Save the model during training
            lr0=0.00005,  # Lower learning rate
            weight_decay=0.001,  # Regularization
            dropout=0.3,  # Regularization (dropout)

            # 🔹 Data Augmentation 🔹
            hsv_h=0.1,  # Hue change
            hsv_s=0.1,  # Saturation change
            hsv_v=0.1,  # Brightness change
            translate=0.2,  # Translation variations
            scale=0.4,  # Scaling
            shear=0.2,  # Shear distortions
            flipud=0.1,  # Vertical flip
            fliplr=0.5,  # Horizontal flip

            # 🔹 Avoid Class Mixing 🔹
            mosaic=0.2,  # Mosaic augmentation
            mixup=0.2,  # Mixup augmentation
            copy_paste=0.1,  # Copy-paste augmentation

            # 🔹 Classification Loss Adjustment 🔹
            cls=1.0,  # Balance classification loss
            patience=50,  # Early stopping if no improvement
            val=True  # Enable validation during training
        )

        print("✅ Fine-tuned YOLOv5 model training complete!")

# Train the model
if __name__ == "__main__":
    ModelTrainer.train(dataset_path="dataset/data.yaml", epochs=150, batch_size=8)
