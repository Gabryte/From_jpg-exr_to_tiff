import os
import sys
from lib.global_dataset_functions.resize import resize_rgb_and_depth_maintain_aspect_ratio
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.box_and_plotting_helpers import calculate_iou, xywhn_to_xyxy, plot_error_visualization
from lib.utility import normalize_array_to_range
from lib.exr_functions import load_single_channel_exr_map
import yaml
import supervision as sv # Import supervision

def analyze_model_errors_rgbd(
        model_path,
        test_images_dir,
        test_labels_dir,
        test_depth_dir,
        output_dir,
        iou_threshold=0.5,
        conf_threshold=0.25,
        class_conf_thresholds=None,
        class_iou_thresholds=None,
        global_min_log_depth=None,
        global_max_log_depth=None,
        TARGET_WIDTH=640,
        use_inference_slicer=False, # New parameter to enable/disable slicer
        slicer_slice_wh=(640, 640), # Default slice size
        slicer_overlap_ratio=(15, 15) # Default overlap ratio
):
    """
    Analyzes YOLOv11 RGBD model errors, categorizing detections and visualizing them.
    Includes visualization of ground truth boxes in false positive/negative frames
    and displays bounding box error (IoU). The final print statements are saved to a log file.

    Args:
        model_path (str): Path to your trained YOLOv11 model (e.g., 'runs/train/exp/weights/best.pt').
        test_images_dir (str): Directory containing test RGB images (e.g., JPEG, PNG).
        test_labels_dir (str): Directory containing YOLO-format (.txt) ground truth labels.
        test_depth_dir (str): Directory containing raw EXR depth maps for distance analysis.
        output_dir (str): Directory to save analysis results and visualizations.
        iou_threshold (float): IoU threshold to consider a detection a True Positive.
        conf_threshold (float): Confidence threshold for model predictions.
        class_conf_thresholds (dict, optional): A dictionary mapping class_id to custom confidence thresholds.
                                                If a class_id is not present, conf_threshold will be used.
        class_iou_thresholds (dict, optional): A dictionary mapping class_id to custom IoU thresholds.
                                                If a class_id is not present, iou_threshold will be used.
        global_min_log_depth (float): Minimum log depth for normalization (from your dataset).
        global_max_log_depth (float): Maximum log depth for normalization (from your dataset).
        TARGET_WIDTH (int): The target width used during image processing.
        use_inference_slicer (bool): If True, use Supervision's InferenceSlicer for inference.
        slicer_slice_wh (tuple): (width, height) of the slices for InferenceSlicer.
        slicer_overlap_ratio (float): Overlap ratio between slices for InferenceSlicer.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations", "bbox_iou_by_class"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations", "confidence_distribution_by_class"), exist_ok=True)
    # New directory for class-wise detection errors by distance
    os.makedirs(os.path.join(output_dir, "visualizations", "detection_errors_by_class_and_distance"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "false_positives"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "false_positives", "localization_errors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "false_positives", "classification_errors"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "false_positives", "spurious_detections"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "false_negatives"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations", "false_negatives_details"), exist_ok=True) # New for FN distributions

    log_file_path = os.path.join(output_dir, "analysis_log.txt")
    original_stdout = sys.stdout

    if class_conf_thresholds is None:
        class_conf_thresholds = {}
    if class_iou_thresholds is None:
        class_iou_thresholds = {}

    print("📊 Starting detailed model error analysis...")

    model = YOLO(model_path)
    print(f"✅ Model loaded from: {model_path}")

    analysis_results = {
        "total_frames": 0,
        "true_positives": 0,
        "false_positives_localization_error": 0,
        "false_positives_classification_error": 0,
        "false_positives_spurious_detection": 0,
        "false_negatives": 0,
        "detection_errors_by_distance": {},
        "detection_errors_by_class_and_distance": {},
        "classification_errors": {},
        "bbox_errors_iou_tp_by_class": {},
        "bbox_ious_fp_localization_by_class": {},
        "bbox_ious_fp_classification_by_class": {},
        "tp_conf_by_class": {},
        "fp_loc_conf_by_class": {},
        "fp_cls_conf_by_class": {},
        "fn_no_pred_conf_by_class": {}, # NEW: Confidence for FNs where no pred was made
        "fn_rejected_pred_conf_by_class": {}, # NEW: Confidence for FNs where pred was rejected
        "fn_no_pred_1minus_iou_by_class": {}, # NEW: (1-IoU) for FNs where no pred was made (will be 1.0)
        "fn_rejected_pred_1minus_iou_by_class": {}, # NEW: (1-IoU) for FNs where pred was rejected
        # NEW: Global lists for FN distributions
        "global_fn_no_pred_confs": [],
        "global_fn_rejected_pred_confs": [],
        "global_fn_no_pred_1minus_ious": [],
        "global_fn_rejected_pred_1minus_ious": [],
        # NEW: Global lists for Bbox Error (IoU) distributions
        "global_tp_1minus_ious": [],  # Global list for (1 - IoU) of True Positives
        "global_fp_loc_ious": [],     # Global list for IoU of FP Localization Errors
        "global_fp_cls_ious": [],     # Global list for IoU of FP Classification Errors
        # NEW: Global lists for Confidence distributions
        "global_tp_confs": [],        # Global list for confidence of True Positives
        "global_fp_loc_confs": [],    # Global list for confidence of FP Localization Errors
        "global_fp_cls_confs": [],    # Global list for confidence of FP Classification Errors
    }

    if hasattr(model, 'names') and model.names is not None:
        class_names = model.names
    else:
        print("⚠️ Warning: Could not find class names from the model. Using default names (e.g., 'class_0').")
        try:
            yaml_path = os.path.join(os.path.dirname(model_path), 'data.yaml')
            with open(yaml_path, 'r') as f_yaml:
                data_config = yaml.safe_load(f_yaml)
            class_names = data_config.get('names', {i: f'class_{i}' for i in range(80)})
            print(f"✅ Class names loaded from {yaml_path}")
        except FileNotFoundError:
            class_names = {i: f'class_{i}' for i in range(80)}
            print("⚠️ Warning: Could not load class names from data.yaml. Using generic class_0, class_1 etc.")

    image_files = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.tiff'))])

    distance_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.inf]
    distance_labels = [
        f'{distance_bins[i]}-{distance_bins[i + 1]}m' if i < len(distance_bins) - 1 else f'>{distance_bins[i]}m' for
        i
        in range(len(distance_bins) - 1)]
    for label in distance_labels:
        analysis_results["detection_errors_by_distance"][label] = {"TP": 0, "FP": 0, "FN": 0}

    # Initialize nested dictionary for class-wise distance errors
    for class_id in class_names.keys():
        analysis_results["detection_errors_by_class_and_distance"][class_id] = {}
        for label in distance_labels:
            analysis_results["detection_errors_by_class_and_distance"][class_id][label] = {"TP": 0, "FP": 0, "FN": 0}

    # Define the callback for the InferenceSlicer if it's used
    slicer = None
    if use_inference_slicer:
        def callback_slicer(image_slice: np.ndarray) -> sv.Detections:
            result = model(image_slice, verbose=False, conf=0.0001, iou=0.0001)[0]
            return sv.Detections.from_ultralytics(result)

        slicer = sv.InferenceSlicer(
            callback=callback_slicer,
            slice_wh=slicer_slice_wh,
            overlap_ratio_wh=None,
            overlap_wh=slicer_overlap_ratio,
        )
        print(f"🔄 Using InferenceSlicer with slice_wh={slicer_slice_wh} and overlap_ratio={slicer_overlap_ratio}")


    with tqdm(total=len(image_files), desc="Analyzing Frames", file=original_stdout) as pbar:
        for img_filename in image_files:
            base_filename = os.path.splitext(img_filename)[0]
            rgb_path = os.path.join(test_images_dir, img_filename)
            label_path = os.path.join(test_labels_dir, f"{base_filename}.txt")
            depth_exr_path = os.path.join(test_depth_dir, f"{base_filename}.exr")

            analysis_results["total_frames"] += 1

            img_rgb_4ch_raw = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            if img_rgb_4ch_raw is None:
                print(f"Skipping {rgb_path}: Could not load 4-channel TIFF image.")
                pbar.update(1)
                continue

            img_rgb_bgr_3ch = img_rgb_4ch_raw[:, :, :3]
            img_rgb_display = cv2.cvtColor(img_rgb_bgr_3ch, cv2.COLOR_BGR2RGB)

            initial_h, initial_w, _ = img_rgb_4ch_raw.shape

            depth_map_raw = load_single_channel_exr_map(depth_exr_path)
            if depth_map_raw is None:
                print(f"Skipping {depth_exr_path}: Could not load depth map.")
                pbar.update(1)
                continue

            current_h, current_w = initial_h, initial_w
            depth_map_for_analysis = depth_map_raw

            # Prepare the 4-channel input for the model, considering TARGET_WIDTH
            if TARGET_WIDTH:
                img_rgb_resized, depth_map_resized_to_target_width = resize_rgb_and_depth_maintain_aspect_ratio(
                    TARGET_WIDTH=TARGET_WIDTH,
                    rgb_frame=img_rgb_display,
                    depth_map=depth_map_raw
                )
                current_h, current_w, _ = img_rgb_resized.shape
                depth_map_for_analysis = depth_map_resized_to_target_width # This is the resized depth map

                log_depth_for_model = np.log1p(depth_map_for_analysis)
                normalized_depth_float_for_model = normalize_array_to_range(
                    log_depth_for_model,
                    min_val=global_min_log_depth,
                    max_val=global_max_log_depth,
                    target_range=(0, 1)
                )
                depth_channel_for_model_input = np.clip((normalized_depth_float_for_model * 255.0), 0, 255).astype(
                    np.uint8)
                depth_channel_for_model_input_hwc = np.expand_dims(depth_channel_for_model_input, axis=2)

                img_rgb_resized_bgr = cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR)
                model_input_image_4ch = np.concatenate((img_rgb_resized_bgr, depth_channel_for_model_input_hwc),
                                                       axis=2)
                img_for_display = img_rgb_resized # Use resized RGB for display
            else:
                depth_map_for_analysis = cv2.resize(depth_map_raw, (initial_w, initial_h),
                                                    interpolation=cv2.INTER_LINEAR)
                img_for_display = img_rgb_display
                model_input_image_4ch = img_rgb_4ch_raw # Use original 4-channel image if no target width

            gt_boxes = []
            try:
                with open(label_path, 'r') as f_label:
                    for line in f_label:
                        parts = list(map(float, line.strip().split()))
                        class_id = int(parts[0])
                        # GT boxes should be based on the dimensions the model sees (current_w, current_h)
                        bbox_xyxy = xywhn_to_xyxy(parts[1:], current_w, current_h)

                        center_x, center_y = int((bbox_xyxy[0] + bbox_xyxy[2]) / 2), int(
                            (bbox_xyxy[1] + bbox_xyxy[3]) / 2)
                        center_x = np.clip(center_x, 0, current_w - 1)
                        center_y = np.clip(center_y, 0, current_h - 1)

                        gt_depth = depth_map_for_analysis[center_y, center_x]
                        gt_boxes.append(
                            {'bbox': bbox_xyxy, 'class_id': class_id, 'detected': False, 'distance': gt_depth})
            except FileNotFoundError:
                pass

            pred_boxes_all_conf = []

            if use_inference_slicer and slicer is not None:
                # Pass the 4-channel image to the slicer
                detections_sv = slicer(model_input_image_4ch)
                # Convert Supervision Detections back to a list of dictionaries for consistent processing
                for i in range(len(detections_sv.xyxy)):
                    bbox_xyxy = detections_sv.xyxy[i].astype(int).tolist()
                    class_id = int(detections_sv.class_id[i])
                    conf = float(detections_sv.confidence[i])

                    center_x, center_y = int((bbox_xyxy[0] + bbox_xyxy[2]) / 2), int(
                        (bbox_xyxy[1] + bbox_xyxy[3]) / 2)
                    center_x = np.clip(center_x, 0, current_w - 1)
                    center_y = np.clip(center_y, 0, current_h - 1)
                    pred_depth = depth_map_for_analysis[center_y, center_x]

                    current_conf_threshold_for_pred = class_conf_thresholds.get(class_id, conf_threshold)
                    current_iou_threshold_for_pred = class_iou_thresholds.get(class_id, iou_threshold)

                    pred_boxes_all_conf.append({
                        'bbox': bbox_xyxy,
                        'class_id': class_id,
                        'conf': conf,
                        'distance': pred_depth,
                        'applied_conf_threshold': current_conf_threshold_for_pred,
                        'applied_iou_threshold': current_iou_threshold_for_pred
                    })

            else:
                # Original inference logic
                results = model(model_input_image_4ch, verbose=False, conf=0.0001, iou=0.0001)

                if results and results[0].boxes:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        bbox_xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        center_x, center_y = int((bbox_xyxy[0] + bbox_xyxy[2]) / 2), int(
                            (bbox_xyxy[1] + bbox_xyxy[3]) / 2)
                        center_x = np.clip(center_x, 0, current_w - 1)
                        center_y = np.clip(center_y, 0, current_h - 1)
                        pred_depth = depth_map_for_analysis[center_y, center_x]

                        current_conf_threshold_for_pred = class_conf_thresholds.get(class_id, conf_threshold)
                        current_iou_threshold_for_pred = class_iou_thresholds.get(class_id, iou_threshold)

                        pred_boxes_all_conf.append({
                            'bbox': bbox_xyxy,
                            'class_id': class_id,
                            'conf': conf,
                            'distance': pred_depth,
                            'applied_conf_threshold': current_conf_threshold_for_pred,
                            'applied_iou_threshold': current_iou_threshold_for_pred
                        })

            # Filter predictions based on confidence threshold for TP/FP analysis
            pred_boxes_filtered = [
                p for p in pred_boxes_all_conf if p['conf'] >= p['applied_conf_threshold']
            ]

            matched_gt_indices_this_frame = set()

            for i, pred in enumerate(pred_boxes_filtered):
                pred_class_id = pred['class_id']
                current_iou_threshold_for_pred_class = class_iou_thresholds.get(pred_class_id, iou_threshold)

                best_iou_overall = 0.0
                best_gt_idx_overall = -1

                best_iou_same_class = 0.0
                best_gt_idx_same_class = -1

                for j, gt in enumerate(gt_boxes):
                    iou = calculate_iou(pred['bbox'], gt['bbox'])

                    if iou > best_iou_overall:
                        best_iou_overall = iou
                        best_gt_idx_overall = j

                    if gt['class_id'] == pred_class_id and iou > best_iou_same_class:
                        best_iou_same_class = iou
                        best_gt_idx_same_class = j

                is_true_positive = False

                if best_gt_idx_same_class != -1:
                    if best_iou_same_class >= current_iou_threshold_for_pred_class and not \
                            gt_boxes[best_gt_idx_same_class]['detected']:
                        analysis_results["true_positives"] += 1
                        gt_boxes[best_gt_idx_same_class]['detected'] = True
                        matched_gt_indices_this_frame.add(best_gt_idx_same_class)

                        if pred_class_id not in analysis_results["bbox_errors_iou_tp_by_class"]:
                            analysis_results["bbox_errors_iou_tp_by_class"][pred_class_id] = []
                        analysis_results["bbox_errors_iou_tp_by_class"][pred_class_id].append(
                            1.0 - best_iou_same_class)
                        # NEW: Add to global TP IoU list
                        analysis_results["global_tp_1minus_ious"].append(1.0 - best_iou_same_class)

                        is_true_positive = True

                        if pred_class_id not in analysis_results["classification_errors"]:
                            analysis_results["classification_errors"][pred_class_id] = {"TP": 0, "FP": 0, "FN": 0}
                        analysis_results["classification_errors"][pred_class_id]["TP"] += 1

                        # Update detection errors by distance (overall)
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_distance"][distance_labels[k]]["TP"] += 1
                                break

                        # NEW: Update detection errors by class and distance
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_class_and_distance"][pred_class_id][
                                    distance_labels[k]]["TP"] += 1
                                break

                        # New: Record confidence for True Positives
                        if pred_class_id not in analysis_results["tp_conf_by_class"]:
                            analysis_results["tp_conf_by_class"][pred_class_id] = []
                        analysis_results["tp_conf_by_class"][pred_class_id].append(pred['conf'])
                        analysis_results["global_tp_confs"].append(pred['conf']) # Add to global TP confs

                if not is_true_positive:
                    if best_gt_idx_same_class != -1 and best_iou_same_class > 0:
                        analysis_results["false_positives_localization_error"] += 1
                        if pred_class_id not in analysis_results["bbox_ious_fp_localization_by_class"]:
                            analysis_results["bbox_ious_fp_localization_by_class"][pred_class_id] = []
                        analysis_results["bbox_ious_fp_localization_by_class"][pred_class_id].append(
                            best_iou_same_class)
                        # NEW: Add to global FP Loc IoU list
                        analysis_results["global_fp_loc_ious"].append(best_iou_same_class)

                        # Update detection errors by distance (overall)
                        
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_distance"][distance_labels[k]]["FP"] += 1
                                break

                        # NEW: Update detection errors by class and distance
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_class_and_distance"][pred_class_id][
                                    distance_labels[k]]["FP"] += 1
                                break

                        gt_for_vis = gt_boxes[best_gt_idx_same_class]
                        plot_error_visualization(
                            img=img_for_display,
                            output_path=os.path.join(output_dir, "false_positives", "localization_errors",
                                                     f"{base_filename}_fp_loc_{i}.png"),
                            title=f"Localization Error: {class_names.get(pred_class_id, 'Unknown Class')}",
                            pred_bbox=pred['bbox'],
                            pred_class_name=class_names.get(pred_class_id, 'Unknown Class'),
                            pred_conf=pred['conf'],
                            pred_distance=pred['distance'],
                            gt_bbox=gt_for_vis['bbox'],
                            gt_class_name=class_names.get(gt_for_vis['class_id'], 'Unknown Class'),
                            gt_distance=gt_for_vis['distance'],
                            iou_value=best_iou_same_class,
                            error_type="FP_localization_error",
                            custom_conf_used=pred['applied_conf_threshold']
                        )
                        # New: Record confidence for FP localization errors
                        if pred_class_id not in analysis_results["fp_loc_conf_by_class"]:
                            analysis_results["fp_loc_conf_by_class"][pred_class_id] = []
                        analysis_results["fp_loc_conf_by_class"][pred_class_id].append(pred['conf'])
                        analysis_results["global_fp_loc_confs"].append(pred['conf']) # Add to global FP-Loc confs

                    elif best_gt_idx_overall != -1 and best_iou_overall > 0 and pred_class_id != \
                            gt_boxes[best_gt_idx_overall]['class_id']:
                        analysis_results["false_positives_classification_error"] += 1
                        if pred_class_id not in analysis_results["bbox_ious_fp_classification_by_class"]:
                            analysis_results["bbox_ious_fp_classification_by_class"][pred_class_id] = []
                        analysis_results["bbox_ious_fp_classification_by_class"][pred_class_id].append(
                            best_iou_overall)
                        # NEW: Add to global FP Cls IoU list
                        analysis_results["global_fp_cls_ious"].append(best_iou_overall)

                        # Update detection errors by distance (overall)
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_distance"][distance_labels[k]]["FP"] += 1
                                break

                        # NEW: Update detection errors by class and distance
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_class_and_distance"][pred_class_id][
                                    distance_labels[k]]["FP"] += 1
                                break

                        if pred_class_id not in analysis_results["classification_errors"]:
                            analysis_results["classification_errors"][pred_class_id] = {"TP": 0, "FP": 0, "FN": 0}
                        analysis_results["classification_errors"][pred_class_id]["FP"] += 1

                        gt_for_vis = gt_boxes[best_gt_idx_overall]
                        plot_error_visualization(
                            img=img_for_display,
                            output_path=os.path.join(output_dir, "false_positives", "classification_errors",
                                                     f"{base_filename}_fp_cls_{i}.png"),
                            title=f"Misclassification: Pred {class_names.get(pred_class_id, 'Unknown')}, True {class_names.get(gt_for_vis['class_id'], 'Unknown')}",
                            pred_bbox=pred['bbox'],
                            pred_class_name=class_names.get(pred_class_id, 'Unknown Class'),
                            pred_conf=pred['conf'],
                            pred_distance=pred['distance'],
                            gt_bbox=gt_for_vis['bbox'],
                            gt_class_name=class_names.get(gt_for_vis['class_id'], 'Unknown Class'),
                            gt_distance=gt_for_vis['distance'],
                            iou_value=best_iou_overall,
                            error_type="FP_classification_error",
                            custom_conf_used=pred['applied_conf_threshold']
                        )
                        # New: Record confidence for FP classification errors
                        if pred_class_id not in analysis_results["fp_cls_conf_by_class"]:
                            analysis_results["fp_cls_conf_by_class"][pred_class_id] = []
                        analysis_results["fp_cls_conf_by_class"][pred_class_id].append(pred['conf'])
                        analysis_results["global_fp_cls_confs"].append(pred['conf']) # Add to global FP-Cls confs
                    else:
                        analysis_results["false_positives_spurious_detection"] += 1

                        # Update detection errors by distance (overall)
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_distance"][distance_labels[k]]["FP"] += 1
                                break

                        # NEW: Update detection errors by class and distance
                        for k, bin_upper_bound in enumerate(distance_bins):
                            if pred['distance'] <= bin_upper_bound:
                                analysis_results["detection_errors_by_class_and_distance"][pred_class_id][
                                    distance_labels[k]]["FP"] += 1
                                break

                        if pred_class_id not in analysis_results["classification_errors"]:
                            analysis_results["classification_errors"][pred_class_id] = {"TP": 0, "FP": 0, "FN": 0}
                        analysis_results["classification_errors"][pred_class_id]["FP"] += 1

                        plot_error_visualization(
                            img=img_for_display,
                            output_path=os.path.join(output_dir, "false_positives", "spurious_detections",
                                                     f"{base_filename}_fp_spurious_{i}.png"),
                            title=f"Spurious Detection: {class_names.get(pred_class_id, 'Unknown Class')}",
                            pred_bbox=pred['bbox'],
                            pred_class_name=class_names.get(pred_class_id, 'Unknown Class'),
                            pred_conf=pred['conf'],
                            pred_distance=pred['distance'],
                            error_type="FP_no_gt_match",
                            custom_conf_used=pred['applied_conf_threshold']
                        )

            # Analyze False Negatives (FN)
            for j, gt in enumerate(gt_boxes):
                if not gt['detected']:
                    analysis_results["false_negatives"] += 1
                    gt_class_id = gt['class_id']
                    gt_distance = gt['distance']

                    if gt_class_id not in analysis_results["classification_errors"]:
                        analysis_results["classification_errors"][gt_class_id] = {"TP": 0, "FP": 0, "FN": 0}
                    analysis_results["classification_errors"][gt_class_id]["FN"] += 1

                    # Update detection errors by distance (overall)
                    for k, bin_upper_bound in enumerate(distance_bins):
                        if gt_distance <= bin_upper_bound:
                            analysis_results["detection_errors_by_distance"][distance_labels[k]]["FN"] += 1
                            break

                    # NEW: Update detection errors by class and distance
                    for k, bin_upper_bound in enumerate(distance_bins):
                        if gt_distance <= bin_upper_bound:
                            analysis_results["detection_errors_by_class_and_distance"][gt_class_id][
                                distance_labels[k]]["FN"] += 1
                            break

                    # Find the closest rejected prediction for this FN
                    closest_rejected_pred_for_fn = None
                    max_iou_for_fn = 0.0

                    for p_all in pred_boxes_all_conf:
                        current_iou_with_gt = calculate_iou(p_all['bbox'], gt['bbox'])
                        # A prediction is "rejected" if its confidence is below its class's threshold OR
                        # if its IoU with GT is below its class's IoU threshold (even if confidence is high)
                        # And ensure it's the correct class or closely related if we're looking for localization/classification issues
                        is_rejected = (p_all['conf'] < p_all['applied_conf_threshold']) or \
                                      (current_iou_with_gt < p_all['applied_iou_threshold'])

                        if is_rejected and current_iou_with_gt > max_iou_for_fn and p_all['class_id'] == gt_class_id: # Only consider same class for IoU-based rejection
                            max_iou_for_fn = current_iou_with_gt
                            closest_rejected_pred_for_fn = p_all

                    if closest_rejected_pred_for_fn:
                        # This is an FN due to rejected prediction
                        plot_error_visualization(
                            img=img_for_display,
                            output_path=os.path.join(output_dir, "false_negatives", f"{base_filename}_fn_{j}.png"),
                            title=f"False Negative: Object Missed ({class_names.get(gt['class_id'], 'Unknown Class')})",
                            gt_bbox=gt['bbox'],
                            gt_class_name=class_names.get(gt['class_id'], 'Unknown Class'),
                            gt_distance=gt['distance'],
                            error_type="FN_details",  # Use the new FN_details type
                            fn_closest_pred_bbox=closest_rejected_pred_for_fn['bbox'],
                            fn_closest_pred_class_name=class_names.get(closest_rejected_pred_for_fn['class_id'],
                                                                       'Unknown Class'),
                            fn_closest_pred_conf=closest_rejected_pred_for_fn['conf'],
                            fn_closest_pred_iou=max_iou_for_fn,
                            fn_closest_pred_min_conf_req=closest_rejected_pred_for_fn['applied_conf_threshold'],
                            fn_closest_pred_min_iou_req=closest_rejected_pred_for_fn['applied_iou_threshold']
                        )
                        # Record for rejected FN
                        if gt_class_id not in analysis_results["fn_rejected_pred_conf_by_class"]:
                            analysis_results["fn_rejected_pred_conf_by_class"][gt_class_id] = []
                        analysis_results["fn_rejected_pred_conf_by_class"][gt_class_id].append(closest_rejected_pred_for_fn['conf'])
                        if gt_class_id not in analysis_results["fn_rejected_pred_1minus_iou_by_class"]:
                            analysis_results["fn_rejected_pred_1minus_iou_by_class"][gt_class_id] = []
                        analysis_results["fn_rejected_pred_1minus_iou_by_class"][gt_class_id].append(1.0 - max_iou_for_fn)
                        # NEW: Add to global lists
                        analysis_results["global_fn_rejected_pred_confs"].append(closest_rejected_pred_for_fn['conf'])
                        analysis_results["global_fn_rejected_pred_1minus_ious"].append(1.0 - max_iou_for_fn)

                    else:
                        # This is an FN where no suitable prediction (even rejected) was found
                        plot_error_visualization(
                            img=img_for_display,
                            output_path=os.path.join(output_dir, "false_negatives", f"{base_filename}_fn_{j}.png"),
                            title=f"False Negative: Object Missed (No Suitable Prediction)",
                            gt_bbox=gt['bbox'],
                            gt_class_name=class_names.get(gt['class_id'], 'Unknown Class'),
                            gt_distance=gt['distance'],
                            error_type="FN"  # Use the old FN type for "no prediction"
                        )
                        # Record for FNs with no prediction
                        if gt_class_id not in analysis_results["fn_no_pred_conf_by_class"]:
                            analysis_results["fn_no_pred_conf_by_class"][gt_class_id] = []
                        # Assign a nominal low confidence (e.g., 0) as no prediction was made
                        analysis_results["fn_no_pred_conf_by_class"][gt_class_id].append(0.0)
                        if gt_class_id not in analysis_results["fn_no_pred_1minus_iou_by_class"]:
                            analysis_results["fn_no_pred_1minus_iou_by_class"][gt_class_id] = []
                        # Assign 1.0 for 1-IoU as there was no overlap
                        analysis_results["fn_no_pred_1minus_iou_by_class"][gt_class_id].append(1.0)
                        # NEW: Add to global lists
                        analysis_results["global_fn_no_pred_confs"].append(0.0)
                        analysis_results["global_fn_no_pred_1minus_ious"].append(1.0)
            pbar.update(1)

    print("\n📈 Generating analysis plots...")
    with open(log_file_path, 'w') as f:
        sys.stdout = f
        if analysis_results["classification_errors"]:
            class_ids = sorted(analysis_results["classification_errors"].keys())
            fp_counts = [analysis_results["classification_errors"][cid]["FP"] for cid in class_ids]
            fn_counts = [analysis_results["classification_errors"][cid]["FN"] for cid in
                         class_ids]
            tp_counts = [analysis_results["classification_errors"][cid]["TP"] for cid in
                         class_ids]

            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(class_ids))

            bar1 = ax.bar(index - bar_width, tp_counts, bar_width, label='True Positives (TP)',
                          color='forestgreen')
            bar2 = ax.bar(index, fp_counts, bar_width, label='False Positives (FP)',
                          color='salmon')
            bar3 = ax.bar(index + bar_width, fn_counts, bar_width, label='False Negatives (FN)',
                          color='skyblue')

            ax.set_xlabel('Class ID')
            ax.set_ylabel('Count')
            ax.set_title('Detection Performance by Class (TP, FP, FN)')
            ax.set_xticks(index)
            ax.set_xticklabels([class_names.get(cid, f'class_{cid}') for cid in class_ids], rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", "classification_errors_by_class.png"))
            plt.close()
            print("Generated: classification_errors_by_class.png")
        else:
            print("No classification errors to plot.")

        all_tracked_classes = set(analysis_results["bbox_errors_iou_tp_by_class"].keys()) | \
                              set(analysis_results["bbox_ious_fp_localization_by_class"].keys()) | \
                              set(analysis_results["bbox_ious_fp_classification_by_class"].keys()) | \
                              set(analysis_results["tp_conf_by_class"].keys()) | \
                              set(analysis_results["fp_loc_conf_by_class"].keys()) | \
                              set(analysis_results["fp_cls_conf_by_class"].keys()) | \
                              set(analysis_results["fn_no_pred_conf_by_class"].keys()) | \
                              set(analysis_results["fn_rejected_pred_conf_by_class"].keys())

        if all_tracked_classes:
            for class_id in sorted(list(all_tracked_classes)):
                tp_ious = analysis_results["bbox_errors_iou_tp_by_class"].get(class_id, [])
                fp_loc_ious = analysis_results["bbox_ious_fp_localization_by_class"].get(class_id, [])
                fp_cls_ious = analysis_results["bbox_ious_fp_classification_by_class"].get(class_id, [])

                if tp_ious or fp_loc_ious or fp_cls_ious:
                    plt.figure(figsize=(10, 6))
                    class_name = class_names.get(class_id, f'class_{class_id}')

                    if tp_ious:
                        plt.hist(tp_ious, bins=20, edgecolor='black', alpha=0.7,
                                 label='True Positives (1 - IoU)', color='forestgreen')

                    if fp_loc_ious:
                        plt.hist(fp_loc_ious, bins=20, edgecolor='black', alpha=0.7,
                                 label='FP: Localization Error (IoU with GT)', color='orange', histtype='step',
                                 linestyle=':')

                    if fp_cls_ious:
                        plt.hist(fp_cls_ious, bins=20, edgecolor='black', alpha=0.7,
                                 label='FP: Classification Error (IoU with GT)', color='purple', histtype='step',
                                 linestyle='-')

                    plt.title(f'Distribution of Bounding Box Errors and IoUs for Class: {class_name}')
                    plt.xlabel('IoU Value (1-IoU for TP, IoU for FPs)')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)
                    plt.legend()
                    output_path = os.path.join(output_dir, "visualizations", "bbox_iou_by_class",
                                               f"bbox_error_distribution_class_{class_name}.png")
                    plt.savefig(output_path)
                    plt.close()
                    print(f"Generated: bbox_iou_by_class/bbox_error_distribution_class_{class_name}.png")

            # NEW: Global Bounding Box Error/IoU Distribution
            print("\n--- Generating Global Bounding Box Error/IoU Distribution ---")
            if analysis_results["global_tp_1minus_ious"] or analysis_results["global_fp_loc_ious"] or analysis_results["global_fp_cls_ious"]:
                plt.figure(figsize=(10, 6))

                if analysis_results["global_tp_1minus_ious"]:
                    plt.hist(analysis_results["global_tp_1minus_ious"], bins=20, edgecolor='black', alpha=0.7,
                             label='True Positives (1 - IoU)', color='forestgreen')
                if analysis_results["global_fp_loc_ious"]:
                    plt.hist(analysis_results["global_fp_loc_ious"], bins=20, edgecolor='black', alpha=0.7,
                             label='FP: Localization Error (IoU with GT)', color='orange', histtype='step', linestyle=':')
                if analysis_results["global_fp_cls_ious"]:
                    plt.hist(analysis_results["global_fp_cls_ious"], bins=20, edgecolor='black', alpha=0.7,
                             label='FP: Classification Error (IoU with GT)', color='purple', histtype='step', linestyle='-')

                plt.title('Global Distribution of Bounding Box Errors and IoUs (All Classes)')
                plt.xlabel('IoU Value (1-IoU for TP, IoU for FPs)')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.legend()
                global_bbox_iou_output_path = os.path.join(output_dir, "visualizations", "global_bbox_error_distribution.png")
                plt.savefig(global_bbox_iou_output_path)
                plt.close()
                print(f"Generated: {os.path.basename(global_bbox_iou_output_path)}")
            else:
                print("No global bounding box error distributions to plot.")


            # New plotting logic for confidence distribution (Per-Class)
            for class_id in sorted(list(all_tracked_classes)):
                tp_confs = analysis_results["tp_conf_by_class"].get(class_id, [])
                fp_loc_confs = analysis_results["fp_loc_conf_by_class"].get(class_id, [])
                fp_cls_confs = analysis_results["fp_cls_conf_by_class"].get(class_id, [])

                if tp_confs or fp_loc_confs or fp_cls_confs:
                    plt.figure(figsize=(10, 6))
                    class_name = class_names.get(class_id, f'class_{class_id}')

                    if tp_confs:
                        plt.hist(tp_confs, bins=20, edgecolor='black', alpha=0.7,
                                 label='True Positives (Confidence)', color='forestgreen')
                    if fp_loc_confs:
                        plt.hist(fp_loc_confs, bins=20, edgecolor='black', alpha=0.7,
                                 label='FP: Localization Error (Confidence)', color='orange', histtype='step',
                                 linestyle=':')
                    if fp_cls_confs:
                        plt.hist(fp_cls_confs, bins=20, edgecolor='black', alpha=0.7,
                                 label='FP: Classification Error (Confidence)', color='purple', histtype='step',
                                 linestyle='-')

                    plt.title(f'Distribution of Predicted Confidence for Class: {class_name}')
                    plt.xlabel('Predicted Confidence')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)
                    plt.legend()
                    output_path = os.path.join(output_dir, "visualizations", "confidence_distribution_by_class",
                                               f"confidence_distribution_class_{class_name}.png")
                    plt.savefig(output_path)
                    plt.close()
                    print(f"Generated: confidence_distribution_by_class/confidence_distribution_class_{class_name}.png")

            # NEW: Global Confidence Distribution (All Classes)
            print("\n--- Generating Global Confidence Distribution (All Classes) ---")
            if analysis_results["global_tp_confs"] or analysis_results["global_fp_loc_confs"] or analysis_results["global_fp_cls_confs"]:
                plt.figure(figsize=(10, 6))

                if analysis_results["global_tp_confs"]:
                    plt.hist(analysis_results["global_tp_confs"], bins=20, edgecolor='black', alpha=0.7,
                             label='True Positives (Confidence)', color='forestgreen')
                if analysis_results["global_fp_loc_confs"]:
                    plt.hist(analysis_results["global_fp_loc_confs"], bins=20, edgecolor='black', alpha=0.7,
                             label='FP: Localization Error (Confidence)', color='orange', histtype='step', linestyle=':')
                if analysis_results["global_fp_cls_confs"]:
                    plt.hist(analysis_results["global_fp_cls_confs"], bins=20, edgecolor='black', alpha=0.7,
                             label='FP: Classification Error (Confidence)', color='purple', histtype='step', linestyle='-')

                plt.title('Global Distribution of Predicted Confidence (All Classes)')
                plt.xlabel('Predicted Confidence')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.legend()
                global_conf_output_path = os.path.join(output_dir, "visualizations", "global_confidence_distribution.png")
                plt.savefig(global_conf_output_path)
                plt.close()
                print(f"Generated: {os.path.basename(global_conf_output_path)}")
            else:
                print("No global confidence distributions to plot.")


            # NEW: Plotting overlapped FN distributions by confidence and 1-IoU (Per-Class)
            print("\n--- Generating False Negative Distributions (Per-Class) ---")
            fn_details_output_dir = os.path.join(output_dir, "visualizations", "false_negatives_details")
            os.makedirs(fn_details_output_dir, exist_ok=True)

            for class_id in sorted(list(all_tracked_classes)):
                class_name = class_names.get(class_id, f'class_{class_id}')
                fn_no_pred_confs = analysis_results["fn_no_pred_conf_by_class"].get(class_id, [])
                fn_rejected_pred_confs = analysis_results["fn_rejected_pred_conf_by_class"].get(class_id, [])
                fn_no_pred_1minus_ious = analysis_results["fn_no_pred_1minus_iou_by_class"].get(class_id, [])
                fn_rejected_pred_1minus_ious = analysis_results["fn_rejected_pred_1minus_iou_by_class"].get(class_id, [])

                # Plotting Confidence Distributions for FNs
                if fn_no_pred_confs or fn_rejected_pred_confs:
                    plt.figure(figsize=(10, 6))
                    if fn_no_pred_confs:
                        plt.hist(fn_no_pred_confs, bins=20, edgecolor='black', alpha=0.7,
                                 label='FN: No Prediction (Confidence)', color='red')
                    if fn_rejected_pred_confs:
                        plt.hist(fn_rejected_pred_confs, bins=20, edgecolor='black', alpha=0.7,
                                 label='FN: Rejected Prediction (Confidence)', color='darkred', histtype='step', linestyle='--')

                    plt.title(f'False Negative Confidence Distribution for Class: {class_name}')
                    plt.xlabel('Predicted Confidence')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)
                    plt.legend()
                    output_path_conf = os.path.join(fn_details_output_dir,
                                               f"fn_confidence_distribution_class_{class_name.replace(' ', '_')}.png")
                    plt.savefig(output_path_conf)
                    plt.close()
                    print(f"Generated: {os.path.basename(output_path_conf)}")

                # Plotting (1-IoU) Distributions for FNs
                if fn_no_pred_1minus_ious or fn_rejected_pred_1minus_ious:
                    plt.figure(figsize=(10, 6))
                    if fn_no_pred_1minus_ious:
                        plt.hist(fn_no_pred_1minus_ious, bins=20, edgecolor='black', alpha=0.7,
                                 label='FN: No Prediction (1 - IoU)', color='blue')
                    if fn_rejected_pred_1minus_ious:
                        plt.hist(fn_rejected_pred_1minus_ious, bins=20, edgecolor='black', alpha=0.7,
                                 label='FN: Rejected Prediction (1 - IoU)', color='darkblue', histtype='step', linestyle='--')

                    plt.title(f'False Negative (1 - IoU) Distribution for Class: {class_name}')
                    plt.xlabel('1 - IoU Value')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', alpha=0.75)
                    plt.legend()
                    output_path_iou = os.path.join(fn_details_output_dir,
                                               f"fn_1minus_iou_distribution_class_{class_name.replace(' ', '_')}.png")
                    plt.savefig(output_path_iou)
                    plt.close()
                    print(f"Generated: {os.path.basename(output_path_iou)}")
            else:
                print("No confidence distributions to plot per class.")


            # NEW: Global False Negative Distributions
            print("\n--- Generating Global False Negative Distributions ---")
            # Global Confidence Distribution for FNs
            if analysis_results["global_fn_no_pred_confs"] or analysis_results["global_fn_rejected_pred_confs"]:
                plt.figure(figsize=(10, 6))
                if analysis_results["global_fn_no_pred_confs"]:
                    plt.hist(analysis_results["global_fn_no_pred_confs"], bins=20, edgecolor='black', alpha=0.7,
                             label='FN: No Prediction (Confidence)', color='red')
                if analysis_results["global_fn_rejected_pred_confs"]:
                    plt.hist(analysis_results["global_fn_rejected_pred_confs"], bins=20, edgecolor='black', alpha=0.7,
                             label='FN: Rejected Prediction (Confidence)', color='darkred', histtype='step', linestyle='--')
                plt.title('Global False Negative Confidence Distribution')
                plt.xlabel('Predicted Confidence')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.legend()
                global_output_path_conf = os.path.join(fn_details_output_dir, "global_fn_confidence_distribution.png")
                plt.savefig(global_output_path_conf)
                plt.close()
                print(f"Generated: {os.path.basename(global_output_path_conf)}")
            else:
                print("No global FN confidence distributions to plot.")

            # Global (1-IoU) Distribution for FNs
            if analysis_results["global_fn_no_pred_1minus_ious"] or analysis_results["global_fn_rejected_pred_1minus_ious"]:
                plt.figure(figsize=(10, 6))
                if analysis_results["global_fn_no_pred_1minus_ious"]:
                    plt.hist(analysis_results["global_fn_no_pred_1minus_ious"], bins=20, edgecolor='black', alpha=0.7,
                             label='FN: No Prediction (1 - IoU)', color='blue')
                if analysis_results["global_fn_rejected_pred_1minus_ious"]:
                    plt.hist(analysis_results["global_fn_rejected_pred_1minus_ious"], bins=20, edgecolor='black', alpha=0.7,
                             label='FN: Rejected Prediction (1 - IoU)', color='darkblue', histtype='step', linestyle='--')
                plt.title('Global False Negative (1 - IoU) Distribution')
                plt.xlabel('1 - IoU Value')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.legend()
                global_output_path_iou = os.path.join(fn_details_output_dir, "global_fn_1minus_iou_distribution.png")
                plt.savefig(global_output_path_iou)
                plt.close()
                print(f"Generated: {os.path.basename(global_output_path_iou)}")
            else:
                print("No global FN 1-IoU distributions to plot.")

        else:
            print("No bounding box errors to plot overall.")
            print("No confidence distributions to plot per class.")

        if analysis_results["detection_errors_by_distance"]:
            distances = list(analysis_results["detection_errors_by_distance"].keys())
            tp_dist = [analysis_results["detection_errors_by_distance"][d]["TP"] for d in distances]
            fp_dist = [analysis_results["detection_errors_by_distance"][d]["FP"] for d in distances]
            fn_dist = [analysis_results["detection_errors_by_distance"][d]["FN"] for d in distances]

            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(distances))

            bar1 = ax.bar(index - bar_width, tp_dist, bar_width, label='True Positives', color='forestgreen')
            bar2 = ax.bar(index, fp_dist, bar_width, label='False Positives (Total)', color='salmon')
            bar3 = ax.bar(index + bar_width, fn_dist, bar_width, label='False Negatives', color='skyblue')

            ax.set_xlabel('Distance Bin (meters)')
            ax.set_ylabel('Count')
            ax.set_title('Detection Performance by Object Distance')
            ax.set_xticks(index)
            ax.set_xticklabels(distances, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "visualizations", "detection_errors_by_distance.png"))
            plt.close()
            print("Generated: detection_errors_by_distance.png")
        else:
            print("No detection errors by distance to plot.")

        # NEW: Plotting detection errors by class and distance
        print("\n--- Generating Detection Performance by Class and Object Distance ---")
        class_distance_output_dir = os.path.join(output_dir, "visualizations", "detection_errors_by_class_and_distance")
        os.makedirs(class_distance_output_dir, exist_ok=True)

        for class_id in sorted(analysis_results["detection_errors_by_class_and_distance"].keys()):
            class_name = class_names.get(class_id, f'class_{class_id}')
            class_data = analysis_results["detection_errors_by_class_and_distance"][class_id]

            # Check if there's any data for this class before plotting
            if any(class_data[label]["TP"] > 0 or class_data[label]["FP"] > 0 or class_data[label]["FN"] > 0 for label
                   in distance_labels):

                tp_dist_class = [class_data[d]["TP"] for d in distance_labels]
                fp_dist_class = [class_data[d]["FP"] for d in distance_labels]
                fn_dist_class = [class_data[d]["FN"] for d in distance_labels]

                fig, ax = plt.subplots(figsize=(12, 6))
                bar_width = 0.25
                index = np.arange(len(distance_labels))

                ax.bar(index - bar_width, tp_dist_class, bar_width, label='True Positives', color='forestgreen')
                ax.bar(index, fp_dist_class, bar_width, label='False Positives (Total)', color='salmon')
                ax.bar(index + bar_width, fn_dist_class, bar_width, label='False Negatives', color='skyblue')

                ax.set_xlabel('Distance Bin (meters)')
                ax.set_ylabel('Count')
                ax.set_title(f'Detection Performance for Class: {class_name} by Object Distance')
                ax.set_xticks(index)
                ax.set_xticklabels(distance_labels, rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(class_distance_output_dir,
                                         f"detection_errors_class_{class_name.replace(' ', '_')}_by_distance.png"))
                plt.close()
                print(
                    f"Generated: detection_errors_by_class_and_distance/detection_errors_class_{class_name.replace(' ', '_')}_by_distance.png")
            else:
                print(f"No detection errors to plot by distance for class: {class_name}.")

        total_predictions = analysis_results['true_positives'] + \
                            analysis_results['false_positives_localization_error'] + \
                            analysis_results['false_positives_classification_error'] + \
                            analysis_results['false_positives_spurious_detection']

        total_ground_truths = analysis_results['true_positives'] + analysis_results['false_negatives']

        if total_predictions > 0:
            precision = analysis_results['true_positives'] / total_predictions
            print(f"\nPrecision: {precision:.4f}")
        else:
            print("\nPrecision: N/A (no detections)")

        if total_ground_truths > 0:
            recall = analysis_results['true_positives'] / total_ground_truths
            print(f"Recall: {recall:.4f}")
        else:
            print("Recall: N/A (no ground truth objects)")

        print("\n--- Classification Error Details ---")
        for class_id, errors in analysis_results["classification_errors"].items():
            class_name = class_names.get(class_id, f'class_{class_id}')
            print(
                f"Class {class_name} (ID: {class_id}): TP = {errors['TP']}, FP = {errors['FP']}, FN = {errors['FN']}")

        print("\n--- Detection Errors by Distance Details (Overall) ---")
        for distance_bin, counts in analysis_results["detection_errors_by_distance"].items():
            print(f"Distance Bin {distance_bin}: TP = {counts['TP']}, FP = {counts['FP']}, FN = {counts['FN']}")

        print("\n--- Detection Errors by Class and Distance Details ---")
        for class_id in sorted(analysis_results["detection_errors_by_class_and_distance"].keys()):
            class_name = class_names.get(class_id, f'class_{class_id}')
            print(f"\nClass: {class_name} (ID: {class_id})")
            class_data = analysis_results["detection_errors_by_class_and_distance"][class_id]
            for distance_bin, counts in class_data.items():
                print(f"  Distance Bin {distance_bin}: TP = {counts['TP']}, FP = {counts['FP']}, FN = {counts['FN']}")

    sys.stdout = original_stdout
    print(f"Analysis logs saved to: {log_file_path}")
    print("Analysis complete! Check the 'analysis_results' directory for visualizations and logs. 🧐")