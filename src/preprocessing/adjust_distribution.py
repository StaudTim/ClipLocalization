import cv2
import numpy as np
import os
import random
import shutil

from src.preprocessing.data_augmentation import crop_image, brightness, flip_along_y_axis, rotate
from src.utils.general_utils import save
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


def _count_annotations(path):
    counter_non_empty = 0
    counter_files = 0
    for filename in (os.listdir(path)):
        if filename.endswith(".txt"):
            counter_files += 1
            file_path = os.path.join(path, filename)
            with open(file_path, "r") as f:
                contents = f.read()
                if contents.strip():
                    counter_non_empty += 1

    percentage = round(counter_non_empty / counter_files, 3)
    print("Number of non-empty annotation files:", counter_non_empty, '/', counter_files)
    print(f' -> {percentage}% of the images have annotations')
    return counter_non_empty, counter_files, percentage


def adjust_distribution(path, target_distribution, only_count=False):
    """
    You can provide a distribution value and this function will make sure the distribution of pictures with
    objects in it and without will reach this level.
    :param path: Path to source folder
    :param target_distribution: Distribution value e.g. 0,8 means 80 percent of the images will have objects in it
    :param only_count: If true, the distribution will not be changed. It just prints the current distribution value
    """
    non_empty, total, distribution = _count_annotations(path)
    if only_count:
        return
    elif distribution > target_distribution:
        print('The amount of images with annotations is already higher than the target-distribution you provided')
        return

    target_files = int(target_distribution * total)
    missing_files = target_files - non_empty
    files = os.listdir(path)
    random.shuffle(files)

    added_files = 0
    removed_empty_files = 0
    for file in tqdm(files):
        if file.endswith('.jpg') or file.endswith('.png'):
            if 'cropped_' in file:
                continue

            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            annotation_path = os.path.join(path, file.replace('.jpg', '.txt').replace('.png', '.txt'))
            annotations = np.loadtxt(annotation_path, delimiter=' ')

            if annotations.shape == (5,):
                annotations = annotations[np.newaxis, :]
            elif annotations.shape == (0,) and removed_empty_files < missing_files:
                os.remove(img_path)
                os.remove(annotation_path)
                removed_empty_files += 1
                continue
            elif annotations.shape == (0,) and removed_empty_files >= missing_files:
                continue

            if added_files >= missing_files:
                continue
            cropped_img, cropped_annotations, error = crop_image(img, annotations)
            if error:
                continue

            cropped_img_path = os.path.join(path, 'cropped_' + file)
            cropped_annotation_path = os.path.join(path,
                                                   'cropped_' + file.replace('.jpg', '.txt').replace('.png', '.txt'))
            save(cropped_img, cropped_img_path, cropped_annotations, cropped_annotation_path)
            added_files += 1

    _count_annotations(path)


def copy_files_with_certain_name(path, prefix, destination='../../dataset/real_data'):
    """
    Copy files with a specific prefix to a new folder. Can be used to create a combined data set of phantom and real data.
    :param path: Path to source folder
    :param prefix: Prefix which every file should have which you want to copy
    :param destination: Path to destination folder
    """
    os.makedirs(destination, exist_ok=True)
    files = [
        file for file in os.listdir(path)
        if file.startswith(prefix) and (file.endswith('.png') or file.endswith('.txt'))
    ]

    for file in files:
        source_path = os.path.join(path, file)
        destination_path = os.path.join(destination, file)
        shutil.copyfile(source_path, destination_path)


def apply_augmentation_to_real_data(path, only_with_annotations=True):
    """
    Apply data augmentation techniques like rotation, flipping along the y-axis and changing the brightness. Can be used in combination
    with the function "copy_files_with_certain_name" to increase the amount of images.
    :param path: Path to source folder
    :param only_with_annotations: If true, augmentations techniques will only be applied to images which contains objects to detect
    """
    rotate(path, only_with_annotations=only_with_annotations)

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

            img, annotations = flip_along_y_axis(img, annotations)

            flipped_img_path = os.path.join(path, 'flipped_' + filename)
            flipped_annotation_path = os.path.join(path,
                                                   'flipped_' + filename.replace('.jpg', '.txt').replace('.png',
                                                                                                         '.txt'))
            save(img, flipped_img_path, annotations, flipped_annotation_path)

    brightness(path, delta=0.2, only_with_annotations=only_with_annotations)


if __name__ == "__main__":
    # path = "../../dataset/clip"
    # adjust_distribution(path, 0.85)

    path = "../../dataset/val_images"
    path_augmentation = '../../dataset/real_data'
    prefix = "80_vid_80_"
    copy_files_with_certain_name(path, prefix, path_augmentation)

    path_augmentation = '../../dataset/real_data'
    apply_augmentation_to_real_data(path_augmentation)
