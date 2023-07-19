import base64
import cv2
import glob
import numpy as np
import sys
import os
import time
import torch

from flask import Flask, render_template, request
from utils.image_handler import ImageHandler
from ultralytics import YOLO

from src.resnet.train_fasterrcnn import create_model
from src.utils.transforms import infer_transforms
from src.utils.general_utils import inference_annotations

app = Flask(__name__, static_folder='static', template_folder='templates')


def _is_docker():
    return 'HOSTNAME' in os.environ


def _get_image_str(image):
    retval, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()


def _pick_starting_model(folder_path):
    files = os.listdir(folder_path)
    if files:
        model_path = os.path.join(folder_path, files[0])
    else:
        print("No models found in the folder:", folder_path)
        sys.exit()
    return model_path


def detect_objects_yolo(model, image, detection_threshold=0.25):
    """
    Run inference of YOLO model.
    :param model: Model for inference
    :param image: Image for detection
    :param detection_threshold: Detections with a confidence under this threshold will be ignored.
    """
    colors = np.random.uniform(0, 255, size=(1, 3))

    boxes = model(image)
    coordinates = boxes[0].boxes.xyxy.tolist()
    if len(coordinates) > 0:
        image = inference_annotations(boxes, detection_threshold, ['clips'], colors, image, model_type='yolo')

    image = cv2.resize(image, (446, 446))
    info_text = 'Inference Time: {:.1f}ms per image'.format(boxes[0].speed['inference'])
    return image, info_text


def detect_objects_fasterrcnn(model, image, detection_threshold=0.5):
    """
    Run inference of Faster-RCNN model.
    :param model: Model for inference
    :param image: Image for detection
    :param detection_threshold: Detections with a confidence under this threshold will be ignored.
    """
    colors = np.random.uniform(0, 255, size=(len(fasterrcnn_classes), 3))
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = infer_transforms(image)
    image = torch.unsqueeze(image, 0)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(fasterrcnn_device))
    end_time = time.time()
    inference_time = end_time - start_time

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        orig_image = inference_annotations(outputs, detection_threshold, fasterrcnn_classes, colors, orig_image)

    info_text = 'Inference Time: {:.1f}ms per image'.format(inference_time)
    return orig_image, info_text


@app.route('/', methods=['GET', 'POST'])
def index():
    global current_image_idx, fasterrcnn_model, yolo_model, yolo_model_path, fasterrcnn_model_path, yolo_model_folder, \
        fasterrcnn_model_folder
    images = image_handler.get_images()

    # Get the list of available models in the folders
    yolo_models = [model_file.split('/')[-1] for model_file in glob.glob(yolo_model_folder + '*.pt')]
    fasterrcnn_models = [model_file.split('/')[-1] for model_file in glob.glob(fasterrcnn_model_folder + '*.pth')]

    if request.method == 'POST':
        if 'backward' in request.form:
            current_image_idx = (current_image_idx - 1) % len(images)
        elif 'forward' in request.form:
            current_image_idx = (current_image_idx + 1) % len(images)

        selected_model_yolo = request.form.get('model_selection_yolo')
        selected_model_fasterrcnn = request.form.get('model_selection_fasterrcnn')

        # Reload the YOLO model if the selection is changed
        if selected_model_yolo and selected_model_yolo != yolo_model_path:
            yolo_model_path = os.path.join(yolo_model_folder, selected_model_yolo)
            yolo_model = YOLO(yolo_model_path)

        # Reload the Faster R-CNN model if the selection is changed
        if selected_model_fasterrcnn and selected_model_fasterrcnn != fasterrcnn_model_path:
            fasterrcnn_model_path = os.path.join(fasterrcnn_model_folder, selected_model_fasterrcnn)
            fasterrcnn_model = create_model(len(fasterrcnn_classes))
            fasterrcnn_model.load_state_dict(torch.load(fasterrcnn_model_path, map_location=fasterrcnn_device))
            fasterrcnn_model.to(fasterrcnn_device).eval()

    yolo_image, fasterrcnn_image = images[current_image_idx]
    yolo_image, info_text_yolo = detect_objects_yolo(yolo_model, yolo_image.copy())
    fasterrcnn_image, info_text_fasterrcnn = detect_objects_fasterrcnn(fasterrcnn_model, fasterrcnn_image.copy())

    yolo_image_str = _get_image_str(yolo_image)
    fasterrcnn_image_str = _get_image_str(fasterrcnn_image)

    return render_template('index.html',
                           image_str_yolo=yolo_image_str,
                           image_str_fasterrcnn=fasterrcnn_image_str,
                           info_text_yolo=info_text_yolo,
                           info_text_fasterrcnn=info_text_fasterrcnn,
                           yolo_models=yolo_models,
                           fasterrcnn_models=fasterrcnn_models,
                           selected_model_yolo=yolo_model_path.split('/')[-1],
                           selected_model_fasterrcnn=fasterrcnn_model_path.split('/')[-1])


if __name__ == '__main__':
    # Define paths to directories with different models
    yolo_model_folder = 'models/yolo/'
    fasterrcnn_model_folder = 'models/faster_rccn/'

    # Define path to test images
    test_images = 'test_images'

    if _is_docker():
        prefix = '/src/'
        yolo_model_folder = os.path.join(prefix, yolo_model_folder)
        fasterrcnn_model_folder = os.path.join(prefix, fasterrcnn_model_folder)
        test_images = os.path.join(prefix, test_images)

    yolo_model_path = _pick_starting_model(yolo_model_folder)
    fasterrcnn_model_path = _pick_starting_model(fasterrcnn_model_folder)

    # YOLO configuration
    yolo_model = YOLO(yolo_model_path)
    yolo_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Faster R-CNN configuration
    fasterrcnn_classes = ['__background__', 'clips']

    fasterrcnn_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fasterrcnn_model = create_model(len(fasterrcnn_classes))
    fasterrcnn_model.load_state_dict(torch.load(fasterrcnn_model_path, map_location=fasterrcnn_device))
    fasterrcnn_model.to(fasterrcnn_device).eval()

    current_image_idx = 0

    image_handler = ImageHandler(image_path=test_images)
    app.run(host='0.0.0.0', port=8000, debug=True)
