from .basemodel import BaseModel

from .resnet18_image2imu import ResNet18Image2IMU
from .resnet18_image2imu_regress import ResNet18Image2IMURegress
from .lstm_img_to_first_imus import LstmImg2FirstImus
from .lstm_img_to_last_imus import LstmImg2LastImus
from .lstm_imu_to_next_imus import LstmImu2NextImus
from .lstm_action_planning import LstmImg2ActionPlanning
from .fully_convolutional_network import FullyConvolutional
from .resnet_scene_categorization import ResNetSceneCategorization
from .resnet_one_tower_prediction import ResNet18Image2IMUOneTowerPrediction
from .resnet_one_tower_baseline import ResNet18Image2IMUOneTower
from .resnet_one_tower_planning import ResNet18Image2IMUOneTowerPlanning

__all__ = [
    'ResNet18Image2IMU',
    'ResNet18Image2IMURegress',
    'LstmImg2FirstImus',
    'LstmImu2NextImus',
    'LstmImg2ActionPlanning',
    'LstmImg2LastImus',
    'FullyConvolutional',
    'ResNetSceneCategorization',
    'ResNet18Image2IMUOneTower',
    'ResNet18Image2IMUOneTowerPrediction',
    'ResNet18Image2IMUOneTowerPlanning',
]

# All models should inherit from BaeModel
variables = locals()
for model in __all__:
    assert issubclass(variables[model], BaseModel),\
             "All model classes should inherit from %s.%s. Model %s does not."\
                % (BaseModel.__module__, BaseModel.__name__, model)
