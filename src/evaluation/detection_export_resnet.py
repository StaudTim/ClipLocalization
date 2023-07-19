import cv2
import numpy as np
import os
import time
import torch

from numba import prange
from src.utils.general_utils import collect_all_images
from src.utils.transforms import infer_transforms
from src.resnet.detect_fasterrcnn import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


def _save_txt_file(boxes, score, img_name):
    output_file = f'detections/{img_name}.txt'
    open(output_file, 'w').close()

    if len(boxes) != 0:
        with open(output_file, "a") as f:
            for i in range(len(boxes)):
                conf = np.array2string(round(score[i], 2))
                if conf.strip() == '1' or conf.strip() == '1.':
                    conf = '0.99'

                f.write(f"0 .{conf.split('.')[1]} {int(boxes[i][0])} {int(boxes[i][1])} "
                        f"{int(boxes[i][2])} {int(boxes[i][3])}\n")


def process_image(path, model, device, size=(448, 448)):
    """
    Run inference of a Faster-RCNN model and save predicted bounding boxes and confidence levels to a txt file
    :param path: List with image paths
    :param model: Model which should be used for inference
    :param device: Specifies the device for the inference (CPU/GPU)
    :param size: Defines the size to which the image will be resized
    """
    start_time = time.time()
    image_name = path.split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(path)
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = infer_transforms(image)
    image = torch.unsqueeze(image, 0)
    end_time = time.time()
    preprocessing_time = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(device))
    end_time = time.time()
    inference_time = end_time - start_time

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    _save_txt_file(boxes, scores, image_name)
    print(f'Preprocessing: {preprocessing_time}, Inference: {inference_time}')


if __name__ == '__main__':
    path_model = '../resnet/models/epochs25_fold0.pth'
    path_test_images = '../test_images'
    os.makedirs('detections', exist_ok=True)

    model = load_model(path_model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device).eval()

    test_images = collect_all_images(path_test_images)
    num_images = len(test_images)

    # Process images in parallel using prange
    for i in prange(num_images):
        path = test_images[i]
        process_image(path, model, device)
