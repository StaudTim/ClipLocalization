import os
import cv2

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ImageHandler(FileSystemEventHandler):
    def __init__(self, yolo_img_size=(1080, 1080), fasterrcnn_img_size=(448, 448), image_path='../test_images'):
        self._yolo_size = yolo_img_size
        self._fasterrcnn_size = fasterrcnn_img_size
        self._image_path = image_path
        self._images = []
        self._load_images()

        self._observer = Observer()
        self._observer.schedule(self, image_path, recursive=False)
        self._observer.start()

    def _load_images(self):
        image_paths = os.listdir(self._image_path)
        tmp_images = [cv2.imread(os.path.join(self._image_path, path)) for path in image_paths]
        self._images = []
        for img in tmp_images:
            if img is None or img.size == 0:
                continue
            yolo_resized_img = cv2.resize(img, self._yolo_size)
            fasterrcnn_resized_img = cv2.resize(img, self._fasterrcnn_size)
            self._images.append((yolo_resized_img, fasterrcnn_resized_img))

    def on_any_event(self, event):
        """
        Reload images if you change your image folder. For example if you remove or add some images.
        :param event:
        """
        if event.is_directory:
            return
        if event.event_type == 'modified' or event.event_type == 'created' or event.event_type == 'deleted':
            self._load_images()

    def get_images(self):
        return self._images
