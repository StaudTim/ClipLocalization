from enum import Enum


class MethodAveragePrecision(Enum):
    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2


class CoordinatesType(Enum):
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4
