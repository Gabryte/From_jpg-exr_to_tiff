from ultralytics import YOLO
import sys

def main():
    try:
        # Load the last.pt as the STARTING POINT for a NEW training session.
        # This will initialize the model with weights from the 200-epoch run.
        model = YOLO('/datasets/gsantanni/mines_dataset/my_yolo_train/single_gpu_test_run/weights/last.pt')

        print("Starting YOLO training (Unfrozen, Extended Epochs, Reset LR) with the following configuration:")
        print("-" * 50)

        # Start the training process with all the specified arguments.
        model.train(
            data='data.yaml',             # Path to the dataset configuration file
            epochs=500,                   # <--- NEW: Extended to 500 epochs
            patience=25,                  # Early stopping patience (good safeguard)
            batch=8,                      # Total batch size for training
            imgsz=640,                    # Image size for training and validation
            save=True,                    # Save checkpoints and results
            save_period=-1,               # Save model every `save_period` epochs (-1 for final only)
            cache='disk',                 # Cache images for faster training ('ram' or 'disk')
            device='0',                   # 0 for single GPU
            workers=8,                    # Number of DataLoader workers
            optimizer='AdamW',            # Optimizer to use
            # pretrained=True,            # Not needed when loading a .pt directly into YOLO()
            seed=777,                     # Random seed for reproducibility
            deterministic=True,           # Enforce deterministic operations for reproducibility
            single_cls=False,             # Treat all classes as a single class set as false
            rect=False,                   # Rectangular training
            cos_lr=True,                  # Use cosine learning rate scheduler (will restart from lr0)
            close_mosaic=25,              # Disable mosaic augmentation for the last N epochs
            amp=True,                     # Use Automatic Mixed Precision (AMP) training
            profile=True,                 # Profile YOLO speed
            freeze=0,                     # <--- CRUCIAL: All layers are now unfrozen
            multi_scale=True,             # Use multi-scale training
            val=True,                     # Perform validation during training
            save_json=True,               # Save detection results to a JSON file
            lr0=0.001,                    # Initial learning rate (will apply from the start of this new run)
            lrf=0.01,                     # Final learning rate factor
            warmup_epochs=3.0,            # Number of warmup epochs for learning rate
            warmup_momentum=0.8,          # Warmup initial momentum
            warmup_bias_lr=0.1,           # Warmup initial bias learning rate
            weight_decay=0.0005,          # Optimizer weight decay
            hsv_h=0.02,                   # HSV hue augmentation
            scale=0.5,                    # Image scaling augmentation
            hsv_s=0.7,                    # HSV saturation augmentation
            hsv_v=0.7,                    # HSV value augmentation
            degrees=15.0,                 # Image rotation augmentation
            translate=0.15,               # Image translation augmentation
            shear=0.5,                    # Image shear augmentation
            mosaic=1.0,                   # Mosaic augmentation probability
            mixup=0.1,                    # MixUp augmentation probability
            perspective=0.0005,           # Perspective augmentation
            flipud=0.0,                   # Flip image up-down probability
            fliplr=0.5,                   # Flip image left-right probability
            erasing=0.1,                  # Random erasing augmentation probability
            auto_augment=False,           # Use auto-augmentation policy
            project='my_yolo_train',      # Top-level directory
            #NEW UNIQUE NAME FOR THIS RUN
            name='mono_gpu_b8_w8_unfreeze_from_200epochs_to_500_new_lr',
            # resume=True
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()