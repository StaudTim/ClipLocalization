import numpy as np
import pandas as pd
import sys

from bounding_box import BoundingBox
from collections import Counter
from enumerators import MethodAveragePrecision


def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    [mrec.append(e) for e in rec]

    mpre = []
    [mpre.append(e) for e in prec]

    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []

    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


def get_metrics(gt_boxes,
                 det_boxes,
                 iou_threshold=0.5,
                 method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                 generate_table=False):
    """Get the metrics.
    Args:
        .._boxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP
        """
    ret = {}
    # Get classes of all bounding boxes separating them by classes
    gt_classes_only = []
    classes_bbs = {}
    for bb in gt_boxes:
        c = bb.get_class_id()
        gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        c = bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)

    # Precision x Recall is obtained individually by each class
    for c, v in classes_bbs.items():
        # Report results only in the classes that are in the GT
        if c not in gt_classes_only:
            continue
        npos = len(v['gt'])
        # sort detections by decreasing confidence
        dects = [a for a in sorted(v['det'], key=lambda bb: bb.get_confidence(), reverse=True)]
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of expected detections for each image
        detected_gt_per_image = Counter([bb.get_image_name() for bb in gt_boxes])
        for key, val in detected_gt_per_image.items():
            detected_gt_per_image[key] = np.zeros(val)
        # print(f'Evaluating class: {c}')
        dict_table = {
            'image': [],
            'confidence': [],
            'TP': [],
            'FP': [],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
        # Loop through detections
        for idx_det, det in enumerate(dects):
            img_det = det.get_image_name()

            if generate_table:
                dict_table['image'].append(img_det)
                dict_table['confidence'].append(f'{100 * det.get_confidence():.2f}%')

            # Find ground truth image
            gt = [gt for gt in classes_bbs[c]['gt'] if gt.get_image_name() == img_det]
            # Get the maximum iou among all detectins in the image
            iouMax = sys.float_info.min
            # Given the detection det, find ground-truth with the highest iou
            for j, g in enumerate(gt):
                # print('Ground truth gt => %s' %
                #       str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                iou = BoundingBox.iou(det, g)
                if iou > iouMax:
                    iouMax = iou
                    id_match_gt = j
            # Assign detection as TP or FP
            if iouMax >= iou_threshold:
                # gt was not matched with any detection
                if detected_gt_per_image[img_det][id_match_gt] == 0:
                    TP[idx_det] = 1  # detection is set as true positive
                    detected_gt_per_image[img_det][
                        id_match_gt] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                    if generate_table:
                        dict_table['TP'].append(1)
                        dict_table['FP'].append(0)
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    if generate_table:
                        dict_table['FP'].append(1)
                        dict_table['TP'].append(0)
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
            else:
                FP[idx_det] = 1  # detection is set as false positive
                if generate_table:
                    dict_table['FP'].append(1)
                    dict_table['TP'].append(0)
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if generate_table:
            dict_table['acc TP'] = list(acc_TP)
            dict_table['acc FP'] = list(acc_FP)
            dict_table['precision'] = list(prec)
            dict_table['recall'] = list(rec)
            table = pd.DataFrame(dict_table)
        else:
            table = None
        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
            [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
        elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
            [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
        else:
            Exception('method not defined')
        # add class result in the dictionary to be returned
        ret[c] = {
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'method': method,
            'iou': iou_threshold,
            'table': table
        }
    # For mAP, only the classes in the gt set should be considered
    mAP = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)
    return {'per_class': ret, 'mAP': mAP}
