import cv2
from collections import defaultdict, deque
import datetime
import numpy as np
import os
import glob as glob
import random
import time
import torch
import torch.distributed as dist

from src.utils.logging import log
from src.utils.transforms import get_train_transform, get_valid_transform, get_train_aug
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as et


"""
Some helper functions and classes you need in order to be able to train and evaluate your Faster-RCNN model.
"""

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """

    def __init__(
            self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map

    def __call__(
            self,
            model,
            current_valid_map,
            model_name
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            torch.save(model.state_dict(), f"{model_name}.pth")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    log(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class CustomDataset(Dataset):
    def __init__(self, images_path, labels_path, width, height, classes, transforms=None, use_train_aug=False,
                 train=False, mosaic=False):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.mosaic = mosaic
        self.image_file_types = ['*.png']
        self.all_image_paths = []

        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        self.read_and_clean()

    def read_and_clean(self):
        # Discard any images and labels when the XML
        # file does not contain any object.
        keep_annot_paths = []
        keep_image_paths = []

        for annot_path, image_path in zip(self.all_annot_paths, self.all_image_paths):
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present:
                keep_annot_paths.append(annot_path)
                keep_image_paths.append(image_path)
            else:
                image_name = os.path.splitext(os.path.basename(annot_path))[0]
                print(f"Removing {annot_path} and corresponding {image_path}")

        self.all_annot_paths = keep_annot_paths
        self.all_image_paths = keep_image_paths

        # Discard any image file when no annotation file
        # is not found for the image.
        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, image_name.split('.png')[0] + '.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")
                # items = [item for item in items if item != element]
                self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]
                # self.all_images.remove(image_name)

        # for image_path in self.all_image_paths:
        #     image_name = image_path.split(os.path.sep)[-1].split('.jpg')[0]
        #     possible_xml_name = f"{self.labels_path}/{image_name.split('.jpg')[0]}.xml"
        #     if possible_xml_name not in self.all_annot_paths:
        #         print(f"{possible_xml_name} not found...")
        #         print(f"Removing {image_name} image")
        #         self.all_image_paths.remove(image_path)

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Capture the corresponding XML file for getting the annotations.
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin, ymin, xmax, ymax = self.check_image_and_annotation(
                xmin, ymin, xmax, ymax, image_width, image_height
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
               boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmin, ymin, xmax, ymax, width, height):
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0

        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width

        return xmin, ymin, xmax, ymax

    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
        """
        image, _, _, _, _, _, _, _ = self.load_image_and_labels(index=index)
        orig_image = image.copy()
        # Resize the image according to the `confg.py` resize.
        image = cv2.resize(image, resize_factor)
        h, w, c = image.shape
        s = h // 2

        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        # Create empty image with the above resized image.
        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indexes):
            image, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )
            # Resize the current image according to the above resize,
            # else `result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]`
            # will give error when image sizes are different.
            image = cv2.resize(image, resize_factor)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            for class_name in labels:
                result_classes.append(class_name)

        final_classes = []
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        for idx in range(len(result_boxes)):
            if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                final_classes.append(result_classes[idx])
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
        ]
        return orig_image, result_image / 255., torch.tensor(result_boxes), \
               torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        if not self.mosaic:
            image, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and self.mosaic:
            while True:
                image, image_resized, boxes, labels, \
                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                    idx, resize_factor=(self.height, self.width)
                )
                if len(boxes) > 0:
                    break

        # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if self.use_train_aug:  # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                               bboxes=target['boxes'],
                               labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# Prepare the final datasets and data loaders.
def create_train_dataset(
        train_dir_images, train_dir_labels,
        resize_width, resize_height, classes,
        use_train_aug=False,
        mosaic=True
):
    train_dataset = CustomDataset(
        train_dir_images, train_dir_labels,
        resize_width, resize_height, classes,
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, mosaic=mosaic
    )
    return train_dataset


def create_valid_dataset(
        valid_dir_images, valid_dir_labels,
        resize_width, resize_height, classes
):
    valid_dataset = CustomDataset(
        valid_dir_images, valid_dir_labels,
        resize_width, resize_height, classes,
        get_valid_transform(),
        train=False
    )
    return valid_dataset


def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
