
import numpy as np

import os
import matplotlib.patches as patches

from matplotlib import pyplot as plt
def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x_min, y_min, x_max, y_max] format.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def xywhn_to_xyxy(normalized_coords, img_w, img_h):
    """Converts normalized YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)."""
    x_center, y_center, box_w, box_h = normalized_coords

    x_min = int((x_center - box_w / 2) * img_w)
    y_min = int((y_center - box_h / 2) * img_h)
    x_max = int((x_center + box_w / 2) * img_w)
    y_max = int((y_center + box_h / 2) * img_h)
    return [x_min, y_min, x_max, y_max]

def plot_error_visualization(
        img,
        output_path,
        title,
        pred_bbox=None,
        pred_class_name=None,
        pred_conf=None,
        pred_distance=None,
        gt_bbox=None,
        gt_class_name=None,
        gt_distance=None,
        iou_value=None,
        error_type="FP",
        custom_conf_used=None,
        fn_closest_pred_bbox=None,
        fn_closest_pred_class_name=None,
        fn_closest_pred_conf=None,
        fn_closest_pred_iou=None,
        fn_closest_pred_min_conf_req=None,
        fn_closest_pred_min_iou_req=None
):
    """
    Visualizes different types of model errors, including detailed information for False Negatives.

    Args:
        img (np.array): The image to visualize.
        output_path (str): Path to save the visualization.
        title (str): Title for the plot.
        pred_bbox (list): Bounding box of the prediction [x_min, y_min, x_max, y_max].
        pred_class_name (str): Class name of the prediction.
        pred_conf (float): Confidence score of the prediction.
        pred_distance (float): Distance of the predicted object.
        gt_bbox (list): Bounding box of the ground truth [x_min, y_min, x_max, y_max].
        gt_class_name (str): Class name of the ground truth.
        gt_distance (float): Distance of the ground truth object.
        iou_value (float): IoU value between prediction and ground truth.
        error_type (str): Type of error ('FP_localization_error', 'FP_classification_error',
                          'FP_no_gt_match', 'FN', 'FN_details').
        custom_conf_used (float): The confidence threshold used for the prediction (if custom).
        fn_closest_pred_bbox (list): Bounding box of the closest rejected prediction for FN.
        fn_closest_pred_class_name (str): Class name of the closest rejected prediction for FN.
        fn_closest_pred_conf (float): Confidence of the closest rejected prediction for FN.
        fn_closest_pred_iou (float): IoU of the closest rejected prediction with GT for FN.
        fn_closest_pred_min_conf_req (float): Minimum confidence required for the closest rejected pred's class.
        fn_closest_pred_min_iou_req (float): Minimum IoU required for the closest rejected pred's class.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()

    def add_box(bbox, class_name, color, linestyle, label_prefix, conf=None, distance=None, iou=None,
                conf_req=None, iou_req=None):
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle
        )
        ax.add_patch(rect)

        label_text = f'{label_prefix}: {class_name}'
        if conf is not None:
            label_text += f' (Conf: {conf:.2f}'
            if conf_req is not None:
                label_text += f' / Req:{conf_req:.2f})'
            else:
                label_text += ')'
        if distance is not None:
            label_text += f', Dist: {distance:.2f}m'
        if iou is not None:
            label_text += f', IoU: {iou:.2f}'
            if iou_req is not None:
                label_text += f' / Req:{iou_req:.2f}'

        plt.text(
            bbox[0], bbox[1] - 10,
            label_text,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'),
            fontsize=9, color='white'
        )

    if error_type == "FP_localization_error":
        if pred_bbox:
            add_box(pred_bbox, pred_class_name, 'red', '--', "Predicted", conf=pred_conf, distance=pred_distance)
        if gt_bbox:
            add_box(gt_bbox, gt_class_name, 'green', '-', "Ground Truth", distance=gt_distance)
        if iou_value is not None:
            plt.title(f"Localization Error: {pred_class_name}\nIoU with GT: {iou_value:.2f}")

    elif error_type == "FP_classification_error":
        if pred_bbox:
            add_box(pred_bbox, pred_class_name, 'red', '--', "Predicted", conf=pred_conf, distance=pred_distance)
        if gt_bbox:
            add_box(gt_bbox, gt_class_name, 'green', '-', "Ground Truth", distance=gt_distance)
        if iou_value is not None:
            plt.title(f"Misclassification: Predicted {pred_class_name}, True {gt_class_name}\nIoU: {iou_value:.2f}")

    elif error_type == "FP_no_gt_match":
        if pred_bbox:
            add_box(pred_bbox, pred_class_name, 'red', '--', "Predicted", conf=pred_conf, distance=pred_distance)
        plt.title(f"Spurious Detection: {pred_class_name} (No GT Match)")

    elif error_type == "FN":
        # This FN is when NO prediction was found, not even a rejected one close by
        if gt_bbox:
            add_box(gt_bbox, gt_class_name, 'blue', '--', "Ground Truth", distance=gt_distance)
            plt.text(
                gt_bbox[0], gt_bbox[3] + 5,
                'MISSED BY MODEL (NO PREDICTION)',
                bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='none'),
                fontsize=6, color='black', fontweight='bold'
            )
        plt.title(title)

    elif error_type == "FN_details": # This is for FN where a prediction *was* made but rejected
        if gt_bbox:
            add_box(gt_bbox, gt_class_name, 'blue', '-', "Ground Truth", distance=gt_distance)
        if fn_closest_pred_bbox:
            add_box(fn_closest_pred_bbox, fn_closest_pred_class_name, 'red', ':',
                    "Closest (Rejected) Pred", conf=fn_closest_pred_conf, iou=fn_closest_pred_iou,
                    conf_req=fn_closest_pred_min_conf_req, iou_req=fn_closest_pred_min_iou_req)
        plt.title(f"False Negative (FN) - Ground Truth Missed\nClosest Rejected Prediction IoU: {fn_closest_pred_iou:.2f}")

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_binned_bar_chart(binned_data, title, output_path, x_label, y_label):
    """
    Generates and saves a bar chart for binned data.
    Args:
        binned_data (dict): A dictionary where keys are bin labels and values are counts.
        title (str): The title of the plot.
        output_path (str): The full path to save the plot.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
    """
    if not binned_data:
        print(f"No data to plot for: {title}")
        return

    bins = list(binned_data.keys())
    counts = list(binned_data.values())

    # Sort bins numerically if they are numbers (e.g., '0-5m', '5-10m')
    try:
        # Extract the numerical start of the bin for sorting
        sorted_indices = np.argsort([float(b.split('-')[0]) if isinstance(b, str) and '-' in b else float(b) for b in bins])
        bins = [bins[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
    except ValueError:
        pass # If bins are not numeric or in unexpected format, keep original order

    plt.figure(figsize=(10, 6))
    plt.bar(bins, counts, color='skyblue', edgecolor='black')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
