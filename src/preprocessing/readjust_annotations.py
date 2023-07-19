'''
Readjust annotations. Provide data in the pascal voc format
'''

import glob
import math
import numpy as np
import xml.etree.ElementTree as ET

from src.utils.path import select_path


def transform(xmin, ymin, xmax, ymax, rotation):
    """
    Can be used if you have annotations files where the bounding boxes are rotated and you want to remove the rotation.
    Some models have difficulties or can't be trained with rotated bounding boxes, therefore this function was created.
    The annotation files have to be in the pascal voc format (xml files).
    :param xmin: Bounding Box xmin value
    :param ymin: Bounding Box ymin value
    :param xmax: Bounding Box xmax value
    :param ymax: Bounding Box ymax value
    :param rotation: Bounding Box rotation value
    :return: Returns new coordinates without rotation
    """
    # Convert coordinates to homogeneous coordinates
    bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    ones = np.ones((4, 1))
    bbox_h = np.concatenate((bbox, ones), axis=1)

    # Compute rotation matrix and its inverse
    theta = math.radians(rotation)
    cos_angle = math.cos(theta)
    sin_angle = math.sin(theta)
    R = np.array([[cos_angle, -sin_angle, 0],
                  [sin_angle, cos_angle, 0],
                  [0, 0, 1]])
    R_inv = np.linalg.inv(R)

    # Rotate the bbox
    bbox_rotated_h = np.dot(bbox_h, R.T)

    # Compute the minimum and maximum values of the rotated bbox
    bbox_rotated = bbox_rotated_h[:, :2]
    xmin_rotated = bbox_rotated[:, 0].min()
    ymin_rotated = bbox_rotated[:, 1].min()
    xmax_rotated = bbox_rotated[:, 0].max()
    ymax_rotated = bbox_rotated[:, 1].max()

    # Compute the dimensions of the minimum bounding box
    width = xmax_rotated - xmin_rotated
    height = ymax_rotated - ymin_rotated

    # Compute the center of the rotated bbox
    center_x_rotated = (xmin_rotated + xmax_rotated) / 2
    center_y_rotated = (ymin_rotated + ymax_rotated) / 2

    # Rotate the center of the bbox back to the original frame
    center_h = np.array([[center_x_rotated], [center_y_rotated], [1]])
    center_h_rotated = np.dot(R_inv, center_h)
    center_x = center_h_rotated[0][0]
    center_y = center_h_rotated[1][0]

    # Compute the coordinates of the minimum bounding box
    new_xmin = center_x - width / 2
    new_ymin = center_y - height / 2
    new_xmax = center_x + width / 2
    new_ymax = center_y + height / 2

    return new_xmin, new_ymin, new_xmax, new_ymax


def main():
    print("Select path to annotations folder with xml files")
    path_xml = select_path()

    for xml_file in glob.glob(path_xml + '*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Loop through each <object> element
        for obj in root.findall('object'):
            # Get the <bndbox> element and its child elements if it exists
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # Get the <attribute> element and its child elements
                attribute = obj.find('attributes/attribute')
                rotation = float(attribute.find('value').text)

                if rotation == 0.0:
                    continue

                new_xmin, new_ymin, new_xmax, new_ymax = transform(xmin, ymin, xmax, ymax, rotation)

                # Update xml-file
                bbox.find('xmin').text = str(new_xmin)
                bbox.find('ymin').text = str(new_ymin)
                bbox.find('xmax').text = str(new_xmax)
                bbox.find('ymax').text = str(new_ymax)
                attribute.find('value').text = '0.0'

        tree.write(xml_file)


if __name__ == "__main__":
    main()
