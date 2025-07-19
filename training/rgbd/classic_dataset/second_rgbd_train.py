import sys
import os
from pathlib import Path
import torch
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model
import wandb
import yaml
import tempfile  # Import tempfile for creating temporary files
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
        comet_ml.login(project_name="mines_rgbd_train")
        print("Successfully logged into comet_ml.")
    except Exception as e:
        print(f"Failed to log into Comet ml: {e}")
        print("Please ensure your API key is correct.")
        sys.exit(1)

    try:
        # model_yaml_path = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/yolo11s.yaml'  # Your base YAML file
        resume_checkpoint_path = '/home/jacobo/dataset/new_train_tiff/my_yolo_train/mines_rgbd_train/weights/last.pt'
        # model_pt = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/yolo11s.pt'

        # --- On-the-fly modification of model_config for 4 channels ---
        # print(f"Loading base model architecture from {model_yaml_path}...")
        # model_config={
        # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
        # Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
        # Model docs: https://docs.ultralytics.com/models/yolo11
        # Task docs: https://docs.ultralytics.com/tasks/detect

        # Parameters
        #  'nc': 4, # number of classes
        #  'ch': 4,
        #  'depth_multiple': 0.50, # model depth multiple for 's' scale
        #  'width_multiple': 0.50, # layer channel multiple for 's' scale
        #  'scales': {'s':[0.50, 0.50, 1024]}, # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
        # [depth, width, max_channels]
        # n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
        # summary: 181 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
        # m: [0.50, 1.00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
        # l: [1.00, 1.00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
        # x: [1.00, 1.50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

        # YOLO11n backbone
        #   'backbone': [[-1, 1, 'Conv', [64, 3, 2]],[-1, 1, 'Conv', [128, 3, 2]],[-1, 2, 'C3k2', [256, False, 0.25]],[-1, 1, 'Conv', [256, 3, 2]],[-1, 2, 'C3k2', [512, False, 0.25]],[-1, 1, 'Conv', [512, 3, 2]],[-1, 2, 'C3k2', [512, True]],[-1, 1, 'Conv', [1024, 3, 2]],[-1, 2, 'C3k2', [1024, True]],[-1, 1, 'SPPF', [1024, 5]],[-1, 2, 'C2PSA', [1024]]],
        # YOLO11n head
        #   'head': [[-1, 1, 'nn.Upsample', [None, 2, "nearest"]],[[-1, 6], 1, 'Concat', [1]],[-1, 2, 'C3k2', [512, False]],[-1, 1, 'nn.Upsample', [None, 2, "nearest"]],[[-1, 4], 1, 'Concat', [1]],[-1, 2, 'C3k2', [256, False]],[-1, 1, 'Conv', [256, 3, 2]],[[-1, 13], 1, 'Concat', [1]],[-1, 2, 'C3k2', [512, False]],[-1, 1, 'Conv', [512, 3, 2]],[[-1, 10], 1, 'Concat', [1]],[-1, 2, 'C3k2', [1024, True]],[[16, 19, 22], 1, 'Detect', ['nc']]]
        # }
        # Load the YAML configuration
        # with open('yolo11s-4ch.yaml', 'w') as f:
        # yaml.dump(model_config, f)

        model = YOLO(resume_checkpoint_path)
        model.model.yaml['ch'] = 4
        model.model.model[0].conv.in_channels = 4
        model.model.model, model.model.save = parse_model(deepcopy(model.model.yaml), ch=4)

        if hasattr(model.model.model[0], 'conv'):
            print(f"Verified: First conv layer now expects {model.model.model[0].conv.in_channels} input channels.")

        # --- End of on-the-fly modification ---

        # IMPORTANT: Print nc and ch values AFTER modification
        # nc_after_mod = model_config.get('nc', 'Not Found')
        # ch_after_mod = model_config.get('ch', 'Not Found')
        # print(f"Debug: 'nc' value after script modification: {nc_after_mod}")
        # print(f"Debug: 'ch' value after script modification: {ch_after_mod}")

        # Create a temporary YAML file to pass to YOLO constructor
        # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_f:
        # yaml.dump(model_config, temp_f)
        # temp_yaml_file = temp_f.name  # Store the name for cleanup

        # --- Debug: Print content of temporary YAML file ---
        # print(f"Debug: Content of temporary YAML file ({temp_yaml_file}):")
        # with open(temp_yaml_file, 'r') as f:
        # print(f.read())
        # print("-" * 50)

        # Initialize the YOLO model from the temporary YAML file
        # model = YOLO(temp_yaml_file)
        # print(f"Debug: Initialized YOLO model from {temp_yaml_file} with potentially modified 'ch'.")

        # Add a print statement to see the model's reported nc immediately after initialization
        # if hasattr(model.model, 'nc'):
        # print(f"Debug: Model.model.nc after initialization: {model.model.nc}")
        # elif hasattr(model.model.model[-1], 'nc'):
        # print(f"Debug: Model.model.model[-1].nc after initialization: {model.model.model[-1].nc}")
        # else:
        # print("Debug: nc attribute not found directly on model.model or its last layer after initialization.")

        # if hasattr(model.model.model[0], 'conv'):
        # This is the most crucial verification point for input channels
        # print(f"Verified: Model's first conv layer now expects {model.model.model[0].conv.in_channels} input channels.")
        # else:
        # print("Warning: Could not verify first conv layer after parsing model from YAML.")

        # --- Attempt to load weights from the resume checkpoint ---
        # Temporarily disable checkpoint loading to diagnose 'nc' issue
        # if os.path.exists(resume_checkpoint_path):
        #    try:
        #        print(f"Attempting to load weights from resume checkpoint: {resume_checkpoint_path}")
        #        # Load the state_dict from the checkpoint
        #        checkpoint_state_dict = torch.load(resume_checkpoint_path, map_location='cpu')['model'].state_dict()
        #
        #        # Get the current model's state_dict
        #        current_model_state_dict = model.model.state_dict()
        #
        #        # Filter out the first convolutional layer's weights if there's a size mismatch
        #        filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in current_model_state_dict and current_model_state_dict[k].shape == v.shape}
        #
        #        # Check for the first convolutional layer specifically
        #        first_conv_key = 'model.0.conv.weight'
        #        if first_conv_key in checkpoint_state_dict and first_conv_key in current_model_state_dict:
        #            if current_model_state_dict[first_conv_key].shape[1] != checkpoint_state_dict[first_conv_key].shape[1]:
        #                print(f"Skipping first conv layer weights due to channel mismatch: current has {current_model_state_dict[first_conv_key].shape[1]} channels, checkpoint has {checkpoint_state_dict[first_conv_key].shape[1]}.")
        #                filtered_state_dict.pop(first_conv_key, None) # Remove it if mismatched
        #            else:
        #                print("First conv layer channels match, loading weights.")
        #
        #        # Load the filtered state_dict
        #        model.model.load_state_dict(filtered_state_dict, strict=False)
        #        print(f"Successfully loaded compatible weights from {resume_checkpoint_path}.")
        #
        #    except Exception as e:
        #        print(f"Warning: Could not load weights from {resume_checkpoint_path} or adapt them. Some weights might not be loaded. Error: {e}")
        # else:
        # print(
        # f"Skipping resume checkpoint loading. Starting training with randomly initialized or default Yolo11s 4-channel weights based on yolo11s.yaml.")

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
            epochs=650,
            patience=75,
            batch=16,
            imgsz=800,
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
            name='mines_rgbd_train',
            resume=True,
            pretrained=True
        )
        print("-" * 50)
        print("YOLO training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
