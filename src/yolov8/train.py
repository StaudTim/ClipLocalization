import cv2
import numpy as np
import os
import shutil
import random
import torch
import warnings

from pathlib import Path
from src.preprocessing.data_augmentation import apply_randomly
from sklearn.model_selection import KFold
from ultralytics import YOLO
from tqdm import tqdm

warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


def train(name, epochs, batch, model_typ='m'):
    """
    Train a yolo model
    :param name: Name under which the model will be saved
    :param epochs: Defines epochs for training
    :param batch: Defines batch size for training
    :param model_typ: Defines the model which will be used as a starting point. E.g. 's' for small
    """
    rel_path = '../../dataset/data.yaml'
    full_path = str(Path(rel_path).resolve())

    if model_typ != 'b' and model_typ != 'data_augment':
        model = YOLO(f'../models/yolov8{model_typ}.pt')
    elif model_typ == 'b':
        model = YOLO(f'../../models/best.pt')
    else:
        model = YOLO(f'runs/detect/{name}/weights/last.pt')

    model.train(
        data=full_path,
        imgsz=1088,
        epochs=epochs,
        batch=batch,
        device=device.index,
        name=name,
        dropout=0.15
    )


def save_files_to_dir(image_files, index, path_dir, size=(1088, 1088), path='../../dataset/clip'):
    """
    Saves files to a directory. Can be used for cross validation to save the train and validation images in a separate folder.
    Also allows to check the distribution of the images.
    :param image_files: List of image files
    :param index: List of indexes for the images which should be saved to a new directory
    :param path_dir: Path for destination folder
    :param size: Images will be resized to this size
    :param path: Path source folder
    """
    for i in tqdm(index):
        image_file = image_files[i]
        annotation_file = image_file.replace('.png', '.txt')
        image_path = os.path.join(path, image_file)
        annotation_path = os.path.join(path, image_file.replace('.png', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.resize(image, size)
        annotations = np.loadtxt(annotation_path, delimiter=' ')
        if annotations.shape == (5,):
            annotations = annotations[np.newaxis, :]

        cv2.imwrite(os.path.join(path_dir, 'images', image_file), image)
        fmt = '%.6f'
        if not annotations.shape == (0,):
            fmt = '%d', '%.6f', '%.6f', '%.6f', '%.6f'
        np.savetxt(os.path.join(path_dir, 'labels', annotation_file), annotations, delimiter=' ', fmt=fmt)


def _create_dir(train_dir, val_dir):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)


if __name__ == "__main__":
    kf_active = True
    data_augment = True

    if torch.cuda.is_available():
        path = '../../dataset/clip'
        n_folds = 5
        model_name = input('Save model under the name: ')
        epochs = int(input('Define epochs: '))
        batch = int(input('Define batch size: '))
        model_typ = input(
            'Choose between the different model typs ("n" = nano, "s" = small, "m" = middle, "b" = models/best.pt): ')

        if kf_active:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            image_files = [f for f in os.listdir(path) if f.endswith('.png')]
            for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
                train_dir = os.path.join(path, 'train')
                val_dir = os.path.join(path, 'validation')
                _create_dir(train_dir, val_dir)

                save_files_to_dir(image_files, train_idx, train_dir)
                save_files_to_dir(image_files, val_idx, val_dir)

                tmp_model_name = model_name + f'_fold{fold}'
                if data_augment:
                    train(tmp_model_name, 1, batch, model_typ)
                    for epoch in range(1, epochs):
                        augment_path_img = os.path.join(train_dir, 'images')
                        augment_path_labels = os.path.join(train_dir, 'labels')
                        apply_randomly(augment_path_img, augment_path_labels, delta=0.0)
                        train(tmp_model_name, 1, batch, 'data_augment')
                        tmp_model_name = model_name + f'_fold{fold}{epoch * "2"}'
                else:
                    train(tmp_model_name, epochs, batch, model_typ)
        else:
            train_files = [f for f in os.listdir(path) if f.endswith('.png')]
            val_path = '../../dataset/val_images'
            val_files = [f for f in os.listdir(val_path) if f.endswith('.png')]
            num_files = len(val_files)
            num_samples = int(num_files * 0.3)  # 20% of the total files
            val_samples = random.sample(val_files, num_samples)

            train_dir = os.path.join(path, 'train')
            val_dir = os.path.join(path, 'validation')
            _create_dir(train_dir, val_dir)

            save_files_to_dir(train_files, list(range(len(train_files))), train_dir)
            save_files_to_dir(val_samples, list(range(len(val_samples))), val_dir, path=val_path)

            train(model_name, epochs, batch, model_typ)

    else:
        print("cuda is not available..")
