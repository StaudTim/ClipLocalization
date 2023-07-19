import cv2
import numpy as np
import os
import warnings

from lxml.etree import Element, SubElement, tostring
from os.path import join
from tqdm import tqdm
from xml.dom.minidom import parseString

warnings.filterwarnings('ignore')


def _unconvert(class_id, width, height, x, y, w, h):
    xmax = round((x * width) + (w * width) / 2.0, 2)
    xmin = round((x * width) - (w * width) / 2.0, 2)
    ymax = round((y * height) + (h * height) / 2.0, 2)
    ymin = round((y * height) - (h * height) / 2.0, 2)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


def xml_transform(path_annotations, path_images):
    """
    Convert the yolo annotation files to xml files (pascal voc format).
    :param path_annotations: Path to folder with annotations
    :param path_images: Path to folder with images
    """
    l = os.listdir(path_images)
    ids = [x.split('.')[0] for x in l if x.endswith('.png')]

    annopath = join(path_annotations, '%s.txt')
    imgpath = join(path_images, '%s.png')
    outpath = join(path_images, '%s.xml')

    for i in tqdm(range(len(ids))):
        xml = None
        img_id = ids[i]
        img = cv2.imread(imgpath % img_id)
        height, width, channels = img.shape

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'clip'
        img_name = img_id + '.png'

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name

        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'Unknown'
        node_annotation = SubElement(node_source, 'annotation')
        node_annotation.text = 'Unknown'
        node_image = SubElement(node_source, 'image')
        node_image.text = 'Unknown'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm = np.loadtxt(target).reshape(-1, 5)

            if len(label_norm) == 0:
                xml = tostring(node_root, pretty_print=True)
                parseString(xml)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = _unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3],
                                      labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = 'clips'

                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'

                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_occluded = SubElement(node_object, 'occluded')
                node_occluded.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(int(new_label[1]))
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(int(new_label[3]))
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(int(new_label[2]))
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(int(new_label[4]))

                node_attributes = SubElement(node_object, 'attributes')
                node_attribute = SubElement(node_attributes, 'attribute')
                node_attribute_name = SubElement(node_attribute, 'name')
                node_attribute_name.text = 'rotation'
                node_value = SubElement(node_attribute, 'value')
                node_value.text = '0.0'

                xml = tostring(node_root, pretty_print=True)
                parseString(xml)

        f = open(outpath % img_id, "wb")
        f.write(xml)
        f.close()


if __name__ == "__main__":
    root = '../../dataset/clip'
    path_resized = '../../dataset/clip/resnet'
    os.makedirs(path_resized, exist_ok=True)

    # Resize images since it is not possible anymore after converting to pascal -> absolute coordinates
    size = (224, 224)
    for filename in tqdm(os.listdir(root)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(path_resized, filename), img)

    xml_transform(root, path_resized)
