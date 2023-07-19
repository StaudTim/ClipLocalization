import base64
import cv2
import numpy as np
import os

from flask import Flask, render_template, request
from src.utils.image_handler import ImageHandler
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

model_path = '../models/yolo/best.pt'
model = YOLO(model_path)
app = Flask(__name__, static_folder='../static', template_folder='../templates')
current_image_idx = 0


def _get_image_str(image):
    retval, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()


def detect_objects(image):
    """
    Run inference of a yolo model.
    :param image: Image for detection
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)

    boxes = model(image)
    coordinates = boxes[0].boxes.xyxy.tolist()
    if len(coordinates) > 0:
        for i in range(len(coordinates)):
            cv2.rectangle(image, (int(coordinates[i][0]), int(coordinates[i][1])),
                          (int(coordinates[i][2]), int(coordinates[i][3])), color, 2)

            conf = np.array2string(np.round(boxes[0].boxes.conf[i].cpu().numpy(), 2))
            text_x = int(coordinates[i][0]) - 55
            text_y = int(coordinates[i][1]) - 5
            cv2.putText(image, conf, (text_x, text_y), font, font_scale, color, thickness=1)

    info_text = 'Speed: {preprocess:.1f}ms preprocess, {inference:.1f}ms inference, {postprocess:.1f}ms postprocess per image'.format(
        **boxes[0].speed)
    return image, info_text


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Display results on a flask app
    """
    global current_image_idx
    images = image_handler.get_images()
    if request.method == 'POST':
        if 'backward' in request.form:
            current_image_idx = (current_image_idx - 1) % len(images)
        elif 'forward' in request.form:
            current_image_idx = (current_image_idx + 1) % len(images)

    image, _ = images[current_image_idx]
    image, info_text = detect_objects(image.copy())
    image_str = _get_image_str(image)
    return render_template('index_yolo.html', image_str=image_str, info_text=info_text)


if __name__ == '__main__':
    image_handler = ImageHandler()
    app.run(host='0.0.0.0', port=8000, debug=False)
