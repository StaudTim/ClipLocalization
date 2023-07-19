import cv2
import numpy as np
import os
import random
import torch

from tqdm import tqdm


def split_dataset(root_path, validation_split=0.2, extensions_img='png', transform=None, size=(448, 448)):
    """
    Split your data inside a folder to a train and validation set. Also run some transformations if you want to.
    :param root_path: Path to source folder
    :param validation_split: Defines the ratio between train and validation data
    :param extensions_img: Defines the typ of your images e.g. png, jpg
    :param transform: Provide some transformation if you want to perform them on your data before splitting.
    :param size: Defines the size which your images are resized to
    :return: Returns a test and validation set
    """
    dataset = []
    for filename in tqdm(os.listdir(root_path)):
        if filename.endswith(extensions_img):
            image_path = os.path.join(root_path, filename)
            image = cv2.imread(image_path)
            if size is not None:
                image = cv2.resize(image, size)
            dataset.append(image)

    random.shuffle(dataset)
    total_samples = len(dataset)
    split_index = int(total_samples * (1 - validation_split))

    train_dataset = torch.tensor(np.array(dataset[:split_index]), dtype=torch.float32).permute(0, 3, 1, 2)
    val_dataset = torch.tensor(np.array(dataset[split_index:]), dtype=torch.float32).permute(0, 3, 1, 2)

    if transform is not None:
        train_dataset = transform(train_dataset)
        val_dataset = transform(val_dataset)
    return train_dataset, val_dataset


def get_data_loader(dataset, batch, shuffle=False, num_workers=1):
    """
    Returns a data loader which you can use for e.g. training
    :param dataset: Provide your dataset here
    :param batch: Defines your batch size.
    :param shuffle: If true, the data will be shuffeled
    :param num_workers:
    :return:
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return data_loader
