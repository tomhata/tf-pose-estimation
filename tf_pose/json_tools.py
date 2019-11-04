"""Functions to export keypoint data as JSON"""
import enum
import json


class CocoPart(enum.Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


COCO_PAIRS = [(1, 2),
              (1, 5),
              (2, 3),
              (3, 4),
              (5, 6),
              (6, 7),
              (1, 8),
              (8, 9),
              (9, 10),
              (1, 11),
              (11, 12),
              (12, 13),
              (1, 0),
              (0, 14),
              (14, 16),
              (0, 15),
              (15, 17)]


COCO_COLORS = [[255, 0, 0],
               [255, 85, 0],
               [255, 170, 0],
               [255, 255, 0],
               [170, 255, 0],
               [85, 255, 0],
               [0, 255, 0],
               [0, 255, 85],
               [0, 255, 170],
               [0, 255, 255],
               [0, 170, 255],
               [0, 85, 255],
               [0, 0, 255],
               [85, 0, 255],
               [170, 0, 255],
               [255, 0, 255],
               [255, 0, 170],
               [255, 0, 85]]


def humans_to_keypoints_dict(humans):
    """Extract keypoint information as dictionary from bodypart class."""
    human_dict = {}
    for human_idx, human in enumerate(humans):
        body_parts_dict = {}
        for part_idx, body_part in human.body_parts.items():
            part_dict = {}
            for slot in body_part.__slots__:
                part_dict[slot] = getattr(body_part, slot)
            part_dict["part_name"] = CocoPart(part_idx).name
            body_parts_dict[part_idx] = part_dict
        human_dict[str(human_idx)] = {"body_parts": body_parts_dict}
    return human_dict
