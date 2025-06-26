from ultralytics import YOLO
import sys

def main():
    try:
        # Load the last.pt as the STARTING POINT for a NEW training session
        model = YOLO('/datasets/gsantanni/mines_dataset/my_yolo_train/second_attempt_mono_gpu_b8_w8_unfreeze_from_200epochs/weights/last.pt')

        print("Starting YOLO training with the following configuration:")
        print("-" * 50)

        model.train(
            data='data.yaml',
            epochs=500,
            patience=75,  # Increased patience for harder convergence
            batch=8,
            imgsz=800,    # <<< INCREASED IMAGE SIZE FOR SMALL OBJECTS
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
            freeze=0,
            multi_scale=True,
            val=True,
            save_json=True,
            lr0=0.0001,
            lrf=0.01,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            hsv_h=0.02,
            hsv_s=0.5,    # <<< REDUCED SATURATION AUGMENTATION
            hsv_v=0.5,    # <<< REDUCED VALUE AUGMENTATION
            scale=0.1,
            degrees=10.0, # <<< SLIGHTLY REDUCED ROTATION
            translate=0.1, # <<< SLIGHTLY REDUCED TRANSLATION
            shear=0.2,    # <<< REDUCED SHEAR AUGMENTATION
            mosaic=1.0,
            mixup=0.5,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            erasing=0.1,  # <<< REDUCED ERASING FOR SMALL, CAMOUFLAGED OBJECTS
            auto_augment=False,
            project='my_yolo_train',
            name='third_attempt_large_imgsz_tuned_augs_for_butterfly', # <<< NEW NAME
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
