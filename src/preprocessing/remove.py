import numpy as np
import os
import warnings

from tqdm import tqdm
from src.utils.path import select_path

warnings.filterwarnings('ignore')


def remove_unlabeled_data(path):
    """
    Removes all images and annotation files inside a folder which don't have an object to detect.
    Faster-RCNN (Resnet) had an issue if you provided images with no bounding boxes.
    :param path: Path to source folder
    """
    files_to_remove = []
    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.txt'):
            annotation_path = os.path.join(path, filename)
            annotations = np.loadtxt(annotation_path, delimiter=' ')

            if annotations.shape == (0,):
                img_path = os.path.join(path, filename.replace('.txt', '.png'))
                files_to_remove.append(img_path)
                files_to_remove.append(annotation_path)

    for path_file in files_to_remove:
        os.remove(path_file)


if __name__ == "__main__":
    path = select_path()
    remove_unlabeled_data(path)
