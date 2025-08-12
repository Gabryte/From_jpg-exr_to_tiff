import sys
import os
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model
import wandb
import comet_ml


def main():
    print("--- Script new_final_train.py started ---")  # Add this line to confirm execution of THIS script
    # --- Process ID (PID) Logging ---
    pid = os.getpid()
    pid_file_path = "train_yolo.pid"
    try:
        with open(pid_file_path, "w") as f:
            f.write(str(pid))
        print(f"Process ID (PID) {pid} logged to {pid_file_path}")
    except Exception as e:
        print(f"Warning: Could not log PID to file {pid_file_path}: {e}")

    # --- Comet ml  Login ---
    try:
        comet_ml.login(project_name="hi_res_rgbd_train")
        print("Successfully logged into comet_ml.")
    except Exception as e:
        print(f"Failed to log into Comet ml: {e}")
        print("Please ensure your API key is correct.")
        sys.exit(1)

    try:
        # model_yaml_path = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/yolo11s.yaml'  # Your base YAML file
        #resume_checkpoint_path = '/home/jacobo/dataset/new_train_tiff/my_yolo_train/200_epochs_classic_dataset_rgbd_train/weights/last.pt'
        resume_checkpoint_path = '/mnt/PSSDX31/high_res_dataset/yolo11s.pt'
        # --- On-the-fly modification of model_config for 4 channels ---

        model = YOLO(resume_checkpoint_path)
        model.model.yaml['ch'] = 4
        model.model.model[0].conv.in_channels = 4
        model.model.model, model.model.save = parse_model(deepcopy(model.model.yaml), ch=4)

        if hasattr(model.model.model[0], 'conv'):
            print(f"Verified: First conv layer now expects {model.model.model[0].conv.in_channels} input channels.")

        # --- End of on-the-fly modification ---
        # Verify again before training starts
        if hasattr(model.model.model[0], 'conv'):
            print(
                f"Final verification: Model's first conv layer will train with {model.model.model[0].conv.in_channels} input channels.")
        else:
            print("Final warning: Could not verify first conv layer before training.")

        # --- Debug: Inspect Detect layer's nc ---
        # The Detect layer is typically the last layer in the model.model sequence.
        if hasattr(model.model, 'model') and len(model.model.model) > 0 and hasattr(model.model.model[-1], 'nc'):
            print(f"Debug: Detect layer (model.model.model[-1]) reports 'nc': {model.model.model[-1].nc}")
        else:
            print("Debug: Could not find or inspect 'nc' attribute on the Detect layer.")

        # --- Train the model ---
        model.train(
            data='data.yaml',
            epochs=30,
            patience=4,
            batch=1,
            imgsz=1920,
            save=True,
            save_period=-1,
            cache='disk',
            device='0',
            workers=8,
            optimizer='AdamW',
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=0,
            amp=True,
            profile=False,
            freeze=0,
            multi_scale=True,
            scale=0.1,
            val=True,
            save_json=True,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            hsv_h=0.15,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=15.0,
            translate=0.10,
            mosaic=1.0,
            mixup=0.55,
            cutmix=0.5,
            perspective=0.0004,
            flipud=0.0,
            fliplr=0.5,
            copy_paste=1.0,
            copy_paste_mode='mixup',
            erasing=0.9,
            auto_augment=False,
            project='my_yolo_train',
            name='30_epochs_high_res_rgbd_train',
            resume=False,
            pretrained=True
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()