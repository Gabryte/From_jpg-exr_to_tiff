from ultralytics import YOLO
import sys

def main():
    try:
        # Load the yolo11s.pt as the STARTING POINT for a transfer learning procedure
        model = YOLO('/datasets/gsantanni/mines_dataset/yolo11s.pt')

        print("Starting YOLO training with the following configuration:")
        print("-" * 50)

        model.train(
            data='data.yaml',
            epochs=200,
            patience=25,
            batch=8,
            imgsz=640,
            save=True,
            save_period=-1,
            cache='disk',
            device='0',
            workers=8,
            optimizer='AdamW',
            pretrained = True,
            seed=777,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=20,
            amp=True,
            profile=False,
            freeze=10,
            multi_scale=True,
            val=True,
            save_json=True,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.7,
            scale=0.5,
            degrees=15.0,
            translate=0.15,
            shear=0.5,
            mosaic=1.0,
            mixup=0.1,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            erasing=0.1,  # <<< REDUCED ERASING FOR SMALL, CAMOUFLAGED OBJECTS
            auto_augment=False,
            project='my_yolo_train',
            name='single_gpu_test_run',
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()