import datetime
import numpy as np
import os

from utils import log
# from utils.consts import HDD_TRAIN_SIDS, HDD_TEST_SIDS, LOGS_PATH, HDD_CBUS_CSV_FILES
from utils.consts import *
from utils.data_prep_util import (get_all_frames_fvecs_all_vids,
    get_all_frames_labels_all_vids)

from utils.exp_util import *


object = get_can_bus_corresponding_to_session_id_st_en_dict(201710031645, 361413, 363913, HDD_CBUS_CSV_FILES)