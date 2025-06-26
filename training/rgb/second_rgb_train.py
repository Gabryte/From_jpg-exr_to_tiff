from ultralytics import YOLO
import sys

def main():
    try:
        # Start again from the GOOD 200-epoch checkpoint
        model = YOLO('/datasets/gsantanni/mines_dataset/my_yolo_train/single_gpu_test_run/weights/last.pt')

        print("Starting YOLO training (Unfrozen, Extended Epochs, LOWERED LR, MORE AUGMENTATION) with the following configuration:")
        print("-" * 50)

        model.train(
            data='data.yaml',             # Path to the dataset configuration file
            epochs=500,                   # Keep 500 epochs
            patience=25,
            batch=8,
            imgsz=640,
            save=True,
            save_period=-1,
            cache='disk',
            device='0',
            workers=8,
            optimizer='AdamW',
            seed=777,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=25,
            amp=True,
            profile=True,
            freeze=0,                     # All layers unfrozen
            multi_scale=True,
            val=True,
            save_json=True,
            lr0=0.0001,                   # <--- **LOWERED LEARNING RATE**
            lrf=0.01,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            hsv_h=0.02,
            scale=0.5,
            hsv_s=0.7,
            hsv_v=0.7,
            degrees=15.0,
            translate=0.15,
            shear=0.5,
            mosaic=1.0,
            mixup=0.3,                   # <--- **INCREASED MIXUP**
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            erasing=0.3,                 # <--- **INCREASED ERASING**
            auto_augment=False,
            project='my_yolo_train',
            # Use a new, distinct name for this attempt
            name='mono_gpu_b8_w8_unfreeze_from_200epochs_to_500_lower_lr_more_aug', # <--- **NEW NAME**
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()