#
# Author: Ramashish Gaurav
#
# Creates visual features of all frames in each video using a pretrained
# model.
#

import datetime
import numpy as np
import os
import torchvision.models as models

from utils import log
from utils.consts import HDD_TRAIN_SIDS, HDD_TEST_SIDS, LOGS_PATH
from utils.data_prep_util import (get_all_frames_fvecs_all_vids,
    get_all_frames_labels_all_vids)

# log.configure_log_handler(
#     "%s_%s.log" %  (os.path.join(os.path.dirname(LOGS_PATH), __file__), datetime.datetime.now()))

log.configure_log_handler(
    "%s_%s.log" %  (os.path.join(os.path.dirname(LOGS_PATH)), datetime.datetime.now().strftime('%Y-%m-%d')))
# Extract the feature vecs of all frames of all videos.
#model = models.resnet18(pretrained=True)
fps = "3/1"
#get_all_frames_fvecs_all_vids(HDD_TRAIN_SIDS + HDD_TEST_SIDS, model, fps)

# Set the labels all frames of all videos for each layer. # Layers = 7.
for layer in range(7):
  get_all_frames_labels_all_vids(HDD_TRAIN_SIDS + HDD_TEST_SIDS, layer, fps)
