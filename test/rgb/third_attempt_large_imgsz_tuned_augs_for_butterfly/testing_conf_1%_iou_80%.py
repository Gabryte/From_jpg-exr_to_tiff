from ultralytics import YOLO
import sys

def main():
    try:
        model = YOLO('/home/jacobo/dataset/third_rgb_train_weights/best.pt')

        print("Starting YOLO model evaluation on the TEST split...")
        print("-" * 50)

        metrics = model.val(
            data='/home/jacobo/dataset/test_dataset_rgb/data.yaml',
            imgsz=800,
            batch=16,
            device='0',
            workers=8,
            save_json=True,
            plots=True,          # To save visual outputs
            split='test',        # <<< IMPORTANT: Explicitly tell val() to use the 'test' split
            project='my_yolo_evaluation',
            name='eval_on_rgb_test_data_3',
            verbose = True,
            rect=False,
            single_cls=False,
            iou=0.8,
            conf=0.01,
        )

        print("-" * 50)
        print("YOLO model evaluation completed successfully!")
        print(f"mAP50-95: {metrics.box.map}")
        print(f"mAP50: {metrics.box.map50}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
