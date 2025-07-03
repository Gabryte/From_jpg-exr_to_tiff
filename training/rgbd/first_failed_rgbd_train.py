from ultralytics import YOLO
import sys
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
import wandb
import os # Import the os module to get the process ID
from pathlib import Path # Import Path for easier path manipulation


def main():
    try:
        # --- Log Process ID (PID) ---
        pid = os.getpid()
        pid_file_path = "train_yolo.pid"
        try:
            with open(pid_file_path, "w") as f:
                f.write(str(pid))
            print(f"Process ID (PID) {pid} logged to {pid_file_path}")
        except Exception as e:
            print(f"Warning: Could not log PID to file {pid_file_path}: {e}")

        # --- Weights & Biases Login ---
        try:
            # Replace with your actual API key, or rely on `wandb login` in terminal
            wandb.login(key="REPLACE_WITH_ACTUAL_KEY")
            print("Successfully logged into Weights & Biases.")
        except Exception as e:
            print(f"Failed to log into Weights & Biases: {e}")
            print("Please ensure your API key is correct or run 'wandb login' in your terminal.")
            sys.exit(1)

        # --- Model Initialization for 4 Channels and Resumption ---
        # Specify the exact path to the checkpoint you want to resume from.
        # This will load the model, optimizer state, and continue epochs.
        # Make sure this path is correct for the run you want to resume.
        resume_checkpoint_path = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/my_yolo_train/fifth_attempt_rgbd_train_converted_pngs2/weights/last.pt'
        model_yaml_path = 'yolo11s.yaml' # Fallback for new training

        try:
            model = YOLO(resume_checkpoint_path)
            print(f"Successfully loaded model and state from {resume_checkpoint_path} for resumption.")
            if hasattr(model.model.model[0], 'conv') and model.model.model[0].conv.in_channels != 4:
                print(f"Adjusting first conv layer for 4 channels (was {model.model.model[0].conv.in_channels}). This might reset some weights.")
                model.model.yaml['ch'] = 4
                model.model.model, model.model.save = parse_model(deepcopy(model.model.yaml), ch=4)

        except Exception as e:
            print(f"Could not load specified checkpoint for resumption: {e}.")
            print("Starting new training run from scratch or from default Yolo11s weights.")
            model = YOLO(model_yaml_path)
            model.model.yaml['ch'] = 4
            model.model.model, model.model.save = parse_model(deepcopy(model.model.yaml), ch=4)

        if hasattr(model.model.model[0], 'conv'):
            print(f"Verified: First conv layer now expects {model.model.model[0].conv.in_channels} input channels.")


        # --- Weights & Biases Run Initialization ---
        # It's good practice to give a new run name when making code changes,
        # or if you truly intend to resume the exact previous run, ensure the name matches.
        # For debugging purposes, a new name helps isolate issues.
        run = wandb.init(project="my_yolo_train", name="fifth_attempt_rgbd_train_converted_pngs_automatic_log", resume="allow")

        # --- Re-add the automatic W&B callback ---
        from wandb.integration.ultralytics import add_wandb_callback # Re-import the callback
        add_wandb_callback(model, enable_model_checkpointing=True)


        print("Starting YOLO training with the following configuration:")
        print("-" * 50)

        # --- Train the model ---
        model.train(
            data='data.yaml',
            epochs=1000,
            patience=75,
            batch=16,
            imgsz=800,
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
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            hsv_h=0.02,
            hsv_s=0.5,
            hsv_v=0.5,
            scale=0.1,
            degrees=10.0,
            translate=0.1,
            shear=0.2,
            mosaic=1.0,
            mixup=0.5,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            erasing=0.1,
            auto_augment=False,
            project='my_yolo_train',
            name='fifth_attempt_rgbd_train_converted_pngs2', # This should match the name of the run you're resuming
            resume=True
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

        # --- Log results.csv and results.json as Artifacts (optional, but good for archiving) ---
        run_save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else None
        if run_save_dir and run_save_dir.exists():
            print(f"Attempting to save results.csv and results.json from: {run_save_dir}")
            results_csv_path = run_save_dir / "results.csv"
            results_json_path = run_save_dir / "results.json"

            if results_csv_path.is_file():
                results_artifact = wandb.Artifact(
                    name="yolo_training_results",
                    type="results",
                    description="Ultralytics YOLO training results CSV and JSON"
                )
                results_artifact.add_file(str(results_csv_path))
                if results_json_path.is_file():
                    results_artifact.add_file(str(results_json_path))
                run.log_artifact(results_artifact)
                print("results.csv and results.json logged as W&B Artifact.")
            else:
                print(f"Warning: results.csv not found at {results_csv_path}. Cannot log as Artifact.")
        else:
            print("Warning: Could not determine run save directory. Cannot log results files as Artifacts.")

        wandb.finish()

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
