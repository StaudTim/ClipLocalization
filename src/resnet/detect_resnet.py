import cv2
import numpy as np
import os
import time
import torch
import torchvision.models as models

from src.utils.transforms import infer_transforms
from src.utils.general_utils import inference_annotations_resnet, collect_all_images


def load_model(path_model, classes=2):
    """
        Load your Resnet18 model architecture and initialize your weights with the weight from the file you provided
        :param path_model: Path to your weights file
        :param classes: Defines the number of different objects your model was trained to detect. If you only have one object than you need to provide two
        classes (__background__, YourClass)
        :return: Returns pretrained model
        """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, classes)
    model.load_state_dict(torch.load(path_model, map_location=device))
    return model


if __name__ == '__main__':
    path_model = 'models/first_try.pth'
    path_test_images = '../test_images'
    show_images = False
    save_predictions = True
    classes = [
        '__background__',
        'clips'
    ]

    model = load_model(path_model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device).eval()

    detection_threshold = 0.5
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    test_images = collect_all_images(path_test_images)

    frame_count = 0
    total_fps = 0
    for i in range(len(test_images)):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(device))
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        if len(outputs[0]['boxes']) != 0:
            rig_image = inference_annotations_resnet(
                outputs, detection_threshold, classes,
                colors, orig_image
            )
            if show_images:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(0)

        if show_images:
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(0)

        if save_predictions:
            os.makedirs('test_predictions', exist_ok=True)
            cv2.imwrite(f"test_predictions/{image_name}.jpg", orig_image)
            print(f"Image {i + 1} done...")
            print('-' * 50)

    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
