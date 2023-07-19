import cv2
import os
import numpy as np

from tqdm import tqdm
from numba import prange
from ultralytics import YOLO

model_path = '../../runs/detect/resume_14_05_f0_fold0/weights/best.pt'
model = YOLO(model_path)
current_image_idx = 0


def detect_objects(image, image_name):
    """
    Run inference of a YOLO model and save predicted bounding boxes and confidence levels to a txt file
    :param image: Image which should be used for detection
    :param image_name: Name of the image
    """
    boxes = model(image)
    coordinates = boxes[0].boxes.xyxy.tolist()
    output_file = f"detections/{os.path.splitext(image_name)[0]}.txt"

    if not os.path.exists('detections'):
        os.makedirs('detections')

    # Empty existing file / create file for all images
    open(output_file, 'w').close()

    if len(coordinates) > 0:
        for i in range(len(coordinates)):
            conf = np.array2string(np.round(boxes[0].boxes.conf[i].cpu().numpy(), 2))
            if conf.strip() == '1':
                conf = '0.99'

            with open(output_file, "a") as f:
                f.write(f"0 .{conf.split('.')[1]} {int(coordinates[i][0])} {int(coordinates[i][1])} "
                        f"{int(coordinates[i][2])} {int(coordinates[i][3])}\n")


if __name__ == '__main__':
    image_paths = os.listdir('../test_images')
    tmp_images = [cv2.imread(os.path.join('../test_images', path)) for path in tqdm(image_paths)]
    for idx in prange(len(tmp_images)):
        image_file = tmp_images[idx]
        if image_file is None or image_file.size == 0:
            continue
        detect_objects(cv2.resize(image_file, (1080, 1080)), image_paths[idx])
