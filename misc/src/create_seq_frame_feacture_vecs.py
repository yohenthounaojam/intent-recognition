#
# This file creates the feature vectors of video clip's frames.
#
# Author: Ramashish Gaurav
#
# Note: On Nvidia 1660 - 6GB: ResNet18 is able to obtain features of 3 frames per
# second of clips less than or equal to 13 secs. GoogleNet requires more GPU RAM.
# It is able to works for only 2 frames per second (haven't checked the maximum
# duration of clips though).
#

import torchvision.models as models
from utils.data_prep_util import prepare_hdd_dataset_sequence_frames

model = models.resnet18(pretrained=True)
prepare_hdd_dataset_sequence_frames(model, fps_rate="3/1")
