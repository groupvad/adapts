from enum import Enum

WANDB_CONF = {
    "PROJECT" : "",
    "ENTITY" : "",
    "KEY" : ""
}

class TaskType(str, Enum):
    """Task type used when generating predictions on the dataset."""
    
    DETECTION = "detection"
    SEGMENTATION = "segmentation"

class Split(str, Enum):
    """Dataset split"""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

class LabelName(int, Enum):
    """Labels encoding"""

    NORMAL = 0
    ABNORMAL = 1

    