import cv2
import math
import os
import shutil
import sys
import time
import torch
import torchvision

from sklearn.model_selection import KFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.utils.coco import CocoEvaluator, get_coco_api_from_dataset
from src.utils.fasterrcnn_utils import create_train_dataset, create_valid_dataset, create_train_loader, \
    create_valid_loader, Averager, SaveBestModel, reduce_dict, MetricLogger, SmoothedValue
from src.utils.logging import log_in_csv


# current_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(current_dir)


def create_dir(train_dir, val_dir):
    """
    Create directories if they don't exist. Otherwise, remove files inside the directory.
    :param train_dir: Path where training directory will be created
    :param val_dir: Path where validation directory will be created
    """
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)


def save_files_to_dir(image_files, index, path_root, path_destination):
    """
    Saves files to a directory. Can be used for cross validation to save the train and validation images in a separate folder.
    Also allows to check the distribution of the images.
    :param image_files: List of image files
    :param index: List of indexes for the images which should be saved to a new directory
    :param path_root: Path to source folder
    :param path_destination: Path to destination folder
    """
    for i in tqdm(index):
        image_file = image_files[i]
        image_path = os.path.join(path_root, image_file)
        annotation_file = image_file.replace('.png', '.xml')
        annotation_path = os.path.join(path_root, annotation_file)

        shutil.copy(image_path, path_destination)
        shutil.copy(annotation_path, path_destination)


def _get_width_height(path, image_name):
    image_path = os.path.join(image_name, path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return height, width


def create_model(num_classes):
    """
    Creates a Faster-RCNN model with resnet50 as backbone.
    :param num_classes:
    :return: Returns the created model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, train_loss_hist, print_freq, scaler=None,
                    scheduler=None
                    ):
    """
    Train your model for one epoch.
    :param model: Provide your model here
    :param optimizer: Provide your optimizer
    :param data_loader: Data loader which returns our batches
    :param device: Specifies the device used for training (CPU/GPU)
    :param epoch: Current epoch number
    :param train_loss_hist: Provide an instance of the "Averager" class
    :param print_freq: Defines the frequenz how often information is displayed
    :param scaler: Provide a scaler if used
    :param scheduler: Provide a scheduler if used
    :return: Returns metrics like the different loss which are calculated during training
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # List to store batch losses.
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    step_counter = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step_counter += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (step_counter / len(data_loader)))

    return metric_logger, batch_loss_list, batch_loss_cls_list, batch_loss_box_reg_list, batch_loss_objectness_list, batch_loss_rpn_list


@torch.inference_mode()
def evaluate(model, data_loader, device):
    """
    Evaluation loop.
    :param model: Provide your model
    :param data_loader: Data loader which returns batches
    :param device: Specifies the device used for evaluation (CPU/GPU)
    :return: Returns metrics like the different loss which are calculated during evaluation
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)

    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    coco_evaluator = CocoEvaluator(coco, iou_types)

    counter = 0
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, stats


if __name__ == "__main__":
    path_data = '../../dataset/clip/resnet'
    path_train = '../../dataset/clip/resnet/train'
    path_val = '../../dataset/clip/resnet/val'
    path_models = 'models'
    path_pretrained = 'models/combined.pt'
    resume_training = False

    os.makedirs(path_models, exist_ok=True)
    model_name = input('Save model under the name: ')
    epochs = int(input('Define epochs: '))
    batch = int(input('Define batch size: '))

    classes = [
        '__background__',
        'clips'
    ]

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    image_files = [f for f in os.listdir(path_data) if f.endswith('.png')]
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        create_dir(path_train, path_val)

        save_files_to_dir(image_files, train_idx, path_data, path_train)
        save_files_to_dir(image_files, val_idx, path_data, path_val)

        height, width = _get_width_height(image_files[0], path_data)
        train_dataset = create_train_dataset(
            path_train, path_train,
            width, height, classes,
            use_train_aug=False,
            mosaic=False
        )
        valid_dataset = create_valid_dataset(
            path_val, path_val,
            width, height, classes
        )
        train_loader = create_train_loader(train_dataset, batch, 1)
        valid_loader = create_valid_loader(valid_dataset, batch, 1)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}\n")

        model = create_model(len(classes))
        if resume_training:
            model = torch.load(path_pretrained)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
        save_best_model = SaveBestModel()

        train_loss_hist = Averager()
        train_loss_list = []
        loss_cls_list = []
        loss_box_reg_list = []
        loss_objectness_list = []
        loss_rpn_list = []
        train_loss_list_epoch = []
        val_map_05 = []
        val_map = []

        for epoch in range(epochs):
            print(f'[TRAIN]: Epoch {epoch + 1} from {epochs} (current kfold={fold})')
            train_loss_hist.reset()

            _, batch_loss_list, batch_loss_cls_list, batch_loss_box_reg_list, batch_loss_objectness_list, batch_loss_rpn_list = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                epoch,
                train_loss_hist,
                print_freq=100,
                scheduler=None
            )
            _, stats = evaluate(
                model,
                valid_loader,
                device=device
            )

            # Append the current epoch's batch-wise losses to the `train_loss_list`.
            train_loss_list.extend(batch_loss_list)
            loss_cls_list.extend(batch_loss_cls_list)
            loss_box_reg_list.extend(batch_loss_box_reg_list)
            loss_objectness_list.extend(batch_loss_objectness_list)
            loss_rpn_list.extend(batch_loss_rpn_list)
            train_loss_list_epoch.append(train_loss_hist.value)
            val_map_05.append(stats[1])
            val_map.append(stats[0])

            tmp_model_name = f'{path_models}/{model_name}_fold{fold}'
            save_best_model(
                model,
                val_map[-1],
                tmp_model_name
            )
            log_in_csv(tmp_model_name, epoch + 1, stats)
