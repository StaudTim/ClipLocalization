import cv2
import os
import validations

from bounding_box import BoundingBox
from enumerators import BBFormat, BBType, CoordinatesType
from src.utils.evaluation_utils import get_annotation_files


def _show_box(annotation_path, x, y, x2, y2, img_size):
    img_path = annotation_path.replace('.txt', '.png')
    img = cv2.imread(img_path)
    if img is None or img.size == 0:
        return
    resized_img = cv2.resize(img, img_size)

    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(resized_img, (int(x), int(y)), (int(x2), int(y2)), color, thickness)

    cv2.imshow("image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def text2bb(annotations_path,
            img_size,
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE,
            show_img=False):
    """
    Convert txt file to bounding boxes.
    :param annotations_path: Path to annotation files
    :param img_size: Define image size
    :param bb_type: Define which bounding box type you got. Are they detected bounding boxes from your model or are they ground truth annotations
    :param bb_format: Define in which format the bounding boxes are e.g. x, y, width, height
    :param type_coordinates: Define if your coordinate are relative to the image size or if they are absolute values
    :param show_img: If true the image will be shown with the detected bounding boxes. Usefull if you want to check if everything works find while debugging
    :return: Return converted bounding boxes
    """
    ret = []

    # Get annotation files in the path
    annotation_files = get_annotation_files(annotations_path)
    for file_path in annotation_files:
        if type_coordinates == CoordinatesType.ABSOLUTE:
            if bb_type == BBType.GROUND_TRUTH and not validations.is_absolute_text_format(
                    file_path, num_blocks=[5], blocks_abs_values=[4]):
                continue
            if bb_type == BBType.DETECTED and not validations.is_absolute_text_format(
                    file_path, num_blocks=[6], blocks_abs_values=[4]):
                continue
        elif type_coordinates == CoordinatesType.RELATIVE:
            if bb_type == BBType.GROUND_TRUTH and not validations.is_relative_text_format(
                    file_path, num_blocks=[5], blocks_rel_values=[4]):
                continue
            if bb_type == BBType.DETECTED and not validations.is_relative_text_format(
                    file_path, num_blocks=[6], blocks_rel_values=[4]):
                continue
        # Loop through lines
        with open(file_path, "r") as f:

            img_filename = os.path.basename(file_path)
            img_filename, file_suffix = os.path.splitext(img_filename)

            for line in f:
                if line.replace(' ', '') == '\n':
                    continue
                splitted_line = line.split(' ')
                class_id = splitted_line[0]
                if bb_type == BBType.GROUND_TRUTH:
                    confidence = None
                    x1 = float(splitted_line[1])
                    y1 = float(splitted_line[2])
                    w = float(splitted_line[3])
                    h = float(splitted_line[4])
                elif bb_type == BBType.DETECTED:
                    confidence = float(splitted_line[1])
                    x1 = float(splitted_line[2])
                    y1 = float(splitted_line[3])
                    w = float(splitted_line[4])
                    h = float(splitted_line[5])
                bb = BoundingBox(image_name=img_filename,
                                 class_id=class_id,
                                 coordinates=(x1, y1, w, h),
                                 img_size=img_size,
                                 confidence=confidence,
                                 type_coordinates=type_coordinates,
                                 bb_type=bb_type,
                                 format=bb_format)
                # If the format is correct, x,y,w,h,x2,y2 must be positive
                x, y, w, h = bb.get_absolute_bounding_box(format=BBFormat.XYWH)
                _, _, x2, y2 = bb.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
                if x < 0 or y < 0 or w < 0 or h < 0 or x2 < 0 or y2 < 0:
                    continue
                ret.append(bb)

                if show_img:
                    _show_box(file_path, x, y, x2, y2, img_size)
    return ret
