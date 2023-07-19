"""
This file calculates the mAP for a model and displays the results.
In order to run it, you first have to run the "detection_export_yolo" or the "detection_export_resnet" depending on which
model you want to evaluate.
"""

import converter

from enumerators import BBType, BBFormat, CoordinatesType, MethodAveragePrecision
from evaluator import get_metrics
from src.utils.path import select_path

if __name__ == "__main__":
    METHODE = MethodAveragePrecision.EVERY_POINT_INTERPOLATION
    image_size = (448, 448)  # needed for Faster-RCNN
    # image_size = (1080, 1080) # needed fo YOLO

    print("Select path to ground truth annotations and make sure the detections are in './detections/'")
    gts_dir = select_path()
    dets_dir = './detections/'

    # Convert annotations to bounding boxes
    gts = converter.text2bb(gts_dir, image_size, BBType.GROUND_TRUTH, type_coordinates=CoordinatesType.RELATIVE)
    dets = converter.text2bb(dets_dir, image_size, BBType.DETECTED, bb_format=BBFormat.XYX2Y2,
                             type_coordinates=CoordinatesType.ABSOLUTE)
    assert (len(gts) > 0)
    assert (len(dets) > 0)

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9]
    results_dict = {}
    for threshold in thresholds:
        results_dict[threshold] = get_metrics(gts, dets, iou_threshold=threshold, method=METHODE, generate_table=True)
        mAP = results_dict[threshold]['mAP']
        print(f'The mAP for the threshold {threshold} is {round(mAP, 4)}')
