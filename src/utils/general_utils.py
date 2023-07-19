import cv2
import glob
import numpy as np
import os


def draw_boxes(image, annotations):
    for i in range(annotations.shape[0]):
        bbox = annotations[i]
        _, x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

        # Convert the YOLO format to pixel coordinates
        img_h, img_w, _ = image.shape
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)

        # Draw the bounding box on the image
        color = (0, 255, 0)  # green
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Show the image
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save(img, img_path, annotations, annotation_path):
    """
    Save image and annotation file in yolo format.
    :param img: Image you want to save
    :param img_path: Path where to save the image
    :param annotations: Annotations of the image
    :param annotation_path: Path where the yolo annotation file should be saved
    """
    cv2.imwrite(img_path, img)
    fmt = '%.6f'
    if not annotations.shape == (0,):
        fmt = '%d', '%.6f', '%.6f', '%.6f', '%.6f'

    np.savetxt(annotation_path, annotations, delimiter=' ', fmt=fmt)


def collect_all_images(dir_test):
    """
    Return all images in a folder
    :param dir_test: Path to source folder
    :return: Return all png files inside the source folder
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.png']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images


def inference_annotations(outputs, detection_threshold, classes, colors, orig_image, model_type='fasterrcnn'):
    """
    Draws bounding boxes and the confidence level on the image you run an inference with.
    :param outputs: Results returned by your model (coordinate bounding boxes, confidence value)
    :param detection_threshold: Detections with a confidence under this threshold will be ignored
    :param classes: Specify which classes do you have
    :param colors: List with colors for different classes
    :param orig_image: Image you run your inference on
    :param model_type: Specify if you used a Faster-RCNN or a YOLO model. Needed because the format of the output is different
    :return: Returns image with bounding boxes and confidence level
    """
    if model_type == 'fasterrcnn':
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
    elif model_type == 'yolo':
        boxes = np.array(outputs[0].boxes.xyxy.tolist())
        scores = np.array(outputs[0].boxes.conf.cpu().numpy())
        pred_classes = ['clips' for _ in range(boxes.shape[0])]

    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    boxes_score = np.round(scores[scores >= detection_threshold], decimals=2)
    draw_boxes = boxes.copy()

    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1)  # Font thickness.

    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        class_name = pred_classes[j]
        color = colors[classes.index(class_name)]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        # For filled rectangle.
        w, h = cv2.getTextSize(
            str(boxes_score[j]),
            0,
            fontScale=lw / 3,
            thickness=tf
        )[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            orig_image,
            p1,
            p2,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            orig_image,
            str(boxes_score[j]),
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.8,
            color=(255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return orig_image
    
def inference_annotations_resnet(outputs, detection_threshold, classes, colors, orig_image):
    boxes = outputs[0].data.numpy()
    scores = outputs[1].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)

    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1)  # Font thickness.

    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(boxes):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        class_name = classes[j]
        color = colors[j]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        # For filled rectangle.
        w, h = cv2.getTextSize(
            class_name,
            0,
            fontScale=lw / 3,
            thickness=tf
        )[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            orig_image,
            p1,
            p2,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            orig_image,
            class_name,
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3.8,
            color=(255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return orig_image

