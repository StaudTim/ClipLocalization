import cv2
import numpy as np
import os
import random
import warnings

from tqdm import tqdm
from src.utils.general_utils import save, draw_boxes

warnings.filterwarnings('ignore')


def brightness(path, delta, show_images=False, only_with_annotations=True):
    """
    Change the brightness level of the images. The value for changing the brightness level will be chosen randomly between a min and max value.
    :param path: Path to source folder
    :param delta: Used for brightness level. E.g if 0.2 the brightness level will be chosen randomly for each image between 0.8 and 1.2
    :param show_images: If true, the images will be shown in order to make sure the changes are valid
    :param only_with_annotations: If true, augmentation will only be applied if the image contains an object which should be detected
    """
    for filename in tqdm(os.listdir(path)):
        tmp_delta = random.uniform(delta * -1, delta)
        brightness_scale = 1.0 + tmp_delta
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            annotation_path = os.path.join(path, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            annotations = np.loadtxt(annotation_path, delimiter=' ')

            if annotations.shape == (5,):
                annotations = annotations[np.newaxis, :]
            elif annotations.shape == (0,) and only_with_annotations:
                continue

            bright_img = cv2.convertScaleAbs(img, alpha=brightness_scale, beta=0)
            bright_img_path = os.path.join(path, 'bright_' + filename)
            bright_annotation_path = os.path.join(path,
                                                  'bright_' + filename.replace('.jpg', '.txt').replace('.png',
                                                                                                       '.txt'))
            save(bright_img, bright_img_path, annotations, bright_annotation_path)

            if show_images:
                draw_boxes(img, annotations)
                draw_boxes(bright_img, annotations)


def _calculate_rotation(img, annotations, angle):
    # Compute the rotation matrix
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image and the bounding box coordinates
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    rotated_annotations = np.zeros_like(annotations)

    for i in range(annotations.shape[0]):
        bbox = annotations[i]
        _, x_rel, y_rel, w_rel, h_rel = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

        # Convert rotated bounding box coordinates back to pixel values
        x = x_rel * img.shape[1]
        y = y_rel * img.shape[0]
        w = w_rel * img.shape[1]
        h = h_rel * img.shape[0]

        # Apply rotation
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        corners = np.hstack((corners, np.ones((4, 1))))
        transformed_corners = rotation_matrix.dot(corners.T).T
        x_min, y_min = np.min(transformed_corners[:, :2], axis=0)
        x_max, y_max = np.max(transformed_corners[:, :2], axis=0)
        width = x_max - x_min
        height = y_max - y_min

        # Convert rotated bounding box coordinates back to relative values
        x_rel = x_max / img.shape[1]
        y_rel = y_max / img.shape[0]
        w_rel = width / img.shape[1]
        h_rel = height / img.shape[0]
        rotated_annotations[i] = [0, x_rel, y_rel, w_rel, h_rel]

    return rotated_img, rotated_annotations


def rotate(path, angle=180, show_images=False, only_with_annotations=True):
    """
    Rotate image around the center and update the annotation files.
    :param path: Path to source folder
    :param angle: Angle for rotation
    :param show_images: If true, the image will be shown in order to make sure the rotation is valid
    :param only_with_annotations: If true, augmentation will only be applied if the image contains an object which should be detected
    """
    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            annotation_path = os.path.join(path, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            annotations = np.loadtxt(annotation_path, delimiter=' ')

            if annotations.shape == (5,):
                annotations = annotations[np.newaxis, :]
            elif annotations.shape == (0,) and only_with_annotations:
                continue

            rotated_img, rotated_annotations = _calculate_rotation(img, annotations, angle)
            rotated_img_path = os.path.join(path, 'rotated_' + filename)
            rotated_annotation_path = os.path.join(path,
                                                   'rotated_' + filename.replace('.jpg', '.txt').replace('.png',
                                                                                                         '.txt'))
            save(rotated_img, rotated_img_path, rotated_annotations, rotated_annotation_path)

            if show_images:
                draw_boxes(img, annotations)
                draw_boxes(rotated_img, rotated_annotations)


def _get_annotations_within_crop(annotations, crop_x1, crop_y1, crop_size, img_shape, threshold=0.5):
    remove_idx = []
    for i in range(annotations.shape[0]):
        bbox = annotations[i]
        _, x_rel, y_rel, w_rel, h_rel = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

        # Convert bounding box coordinates back to pixel values
        center_x = x_rel * img_shape[1]
        center_y = y_rel * img_shape[0]
        bbox_w = w_rel * img_shape[1]
        bbox_h = h_rel * img_shape[0]

        # Calculate overlap between box and crop area
        bbox_x1 = center_x - bbox_w / 2
        bbox_y1 = center_y - bbox_h / 2
        crop_x2 = crop_x1 + crop_size[1]
        crop_y2 = crop_y1 + crop_size[0]
        x1 = max(bbox_x1, crop_x1)
        y1 = max(bbox_y1, crop_y1)
        x2 = min(bbox_x1 + bbox_w, crop_x2)
        y2 = min(bbox_y1 + bbox_h, crop_y2)
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)
        box_area = bbox_w * bbox_h
        overlap_fraction = overlap_area / box_area

        if overlap_fraction < threshold:
            remove_idx.append(i)
        else:
            bbox_x1 = bbox_x1 - crop_x1
            bbox_y1 = bbox_y1 - crop_y1
            bbox_x2 = bbox_x1 + bbox_w
            bbox_y2 = bbox_y1 + bbox_h

            # Check if box is fully within the image boundaries
            if bbox_x1 < 0:
                bbox_w = bbox_w + bbox_x1
                bbox_x1 = 0

            if bbox_y1 < 0:
                bbox_h = bbox_h + bbox_y1
                bbox_y1 = 0

            if bbox_x2 > crop_size[1]:
                outside = bbox_x2 - crop_size[1]
                bbox_w = bbox_w - outside
                bbox_x2 = crop_size[1]

            if bbox_y2 > crop_size[0]:
                outside = bbox_y2 - crop_size[0]
                bbox_h = bbox_h - outside
                bbox_y2 = crop_size[0]

            # Update relative coordinates to match the cropped image
            x_rel = (bbox_x1 + bbox_w / 2) / crop_size[1]
            y_rel = (bbox_y1 + bbox_h / 2) / crop_size[0]
            w_rel = bbox_w / crop_size[1]
            h_rel = bbox_h / crop_size[0]

            annotations[i][1] = x_rel
            annotations[i][2] = y_rel
            annotations[i][3] = w_rel
            annotations[i][4] = h_rel

    filtered_annotations = np.delete(annotations, remove_idx, axis=0)
    return filtered_annotations


def crop_image(img, annotations, crop_size=(300, 500)):
    """
    Searches for a ROI (region of interest) and applies cropping. Also changes the annotations files and makes sure the coordinates
    of the bounding boxes are updated.
    :param crop_size: height, width of the region of interest
    :return: Returns cropped image and annotations
    """
    idx = np.random.randint(0, annotations.shape[0])
    bbox = annotations[idx]

    # Convert relative coordinates to absolute coordinates
    bbox_h, bbox_w, _ = img.shape
    bbox_x1 = int((bbox[1] - bbox[3] / 2) * bbox_w)
    bbox_y1 = int((bbox[2] - bbox[4] / 2) * bbox_h)
    bbox_x2 = int((bbox[1] + bbox[3] / 2) * bbox_w)
    bbox_y2 = int((bbox[2] + bbox[4] / 2) * bbox_h)

    # new x and y coordinates
    min_x = bbox_x2 - crop_size[1]
    if min_x < 0:
        min_x = 0

    max_x = bbox_x1
    if max_x + crop_size[1] > img.shape[1]:
        max_x = max_x - (max_x + crop_size[1] - img.shape[1])

    min_y = bbox_y2 - crop_size[0]
    if min_y < 0:
        min_y = 0

    max_y = bbox_y1
    if max_y + crop_size[0] > img.shape[0]:
        max_y = max_y - (max_y + crop_size[0] - img.shape[0])

    if min_x > max_x or min_y > max_y:
        return img, annotations, True

    if min_x == max_x:
        x_pos = min_x
    else:
        x_pos = np.random.randint(min_x, max_x)

    if min_y == max_y:
        y_pos = min_y
    else:
        y_pos = np.random.randint(min_y, max_y)
    cropped_img = img[y_pos: y_pos + crop_size[0], x_pos:x_pos + crop_size[1]]
    cropped_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]))
    cropped_annotations = _get_annotations_within_crop(annotations, x_pos, y_pos, crop_size, img.shape)
    return cropped_img, cropped_annotations, False


def flip_along_y_axis(img, annotations, show_images=False):
    """
    Flip image along the y-axis
    :param img: Path to image
    :param annotations: Path to annotation
    :param show_images: If true, the image will be shown in order to make sure the flipping is valid
    :return: Returns flipped image and annotations
    """
    flipped_image = cv2.flip(img, 1)
    flipped_annotations = annotations.copy()

    # Update x-coordinates of bounding boxes
    for i in range(len(flipped_annotations)):
        x = flipped_annotations[i][1]
        flipped_x = 1.0 - x
        flipped_annotations[i][1] = flipped_x

    if show_images:
        draw_boxes(img, annotations)
        draw_boxes(flipped_image, flipped_annotations)

    return flipped_image, flipped_annotations


def apply_randomly(path_img, path_label=None, delta=0.2, show_images=False, only_with_annotations=False):
    """
    Apply randomly data augmentation techniques like rotation, flipping along the y-axis and changing the brightness.
    Makes sure to get more variety. For example the brightness level won't be the same in consecutive images.
    :param path_img: Path to image folder
    :param path_label: Path to folder with annotations
    :param delta: Used for brightness level. E.g if 0.2 the brightness level will be chosen randomly for each image between 0.8 and 1.2
    :param show_images: If true, the images will be shown in order to make sure the changes are valid
    :param only_with_annotations: If true, augmentation will only be applied if the image contains an object which should be detected
    """

    if path_label is None:
        path_label = path_img

    for filename in tqdm(os.listdir(path_img)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(path_img, filename)
            img = cv2.imread(img_path)
            annotation_path = os.path.join(path_label, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            annotations = np.loadtxt(annotation_path, delimiter=' ')

            if annotations.shape == (5,):
                annotations = annotations[np.newaxis, :]
            elif annotations.shape == (0,) and only_with_annotations:
                continue

            apply_brightness = bool(random.getrandbits(1))
            apply_rotation = bool(random.getrandbits(1))
            apply_y_flip = bool(random.getrandbits(1))

            if not apply_brightness and not apply_rotation and not apply_y_flip:
                continue

            if apply_brightness and delta > 0.0:
                tmp_delta = random.uniform(delta * -1, delta)
                brightness_scale = 1.0 + tmp_delta
                img = cv2.convertScaleAbs(img, alpha=brightness_scale, beta=0)

            if apply_y_flip:
                img, annotations = flip_along_y_axis(img, annotations)

            if apply_rotation:
                img, annotations = _calculate_rotation(img, annotations, 180)

            save(img, img_path, annotations, annotation_path)

            if show_images:
                draw_boxes(img, annotations)


if __name__ == "__main__":
    folder_path = '../../dataset/clip'
    apply_randomly(folder_path)
