#
# This file contains the experiment config constants.
#
# Author: Ramashish Gaurav
#
import os
################################################################################
##################  DIRECTORY PATHS #####################
################################################################################

# Honda Research Driving Dataset - HDD
# HDD_DATA_BASE_DIR = "/home/rgaurav/Documents/UData/auto-intention-snns/HDD_Data/"

# HDD_SAVED_INDEX_PKL_PATH = HDD_DATA_BASE_DIR + "/EAF_parsing/saved_index.pkl"
# # In HDD_VIDEO_PATH,{0} corresponds to the session ID e.g. 201702271017
# HDD_VIDEO_PATH = HDD_DATA_BASE_DIR + "/release_2019_07_08/{0}/camera/center/"
# In HDD_CBUS_CSV_PATH, {0} corresponds to the session ID and {1} corresponds
# to the CAN Bus data file csv, e.g.: "accel_pedal.csv"
# HDD_CBUS_CSV_PATH = HDD_DATA_BASE_DIR + "/release_2019_07_08/{0}/general/csv/{1}"
# HDD_CBUS_CSV_FILES = ["accel_pedal.csv", "brake_pedal.csv", "rtk_pos.csv",
#                       "rtk_track.csv", "steer.csv", "turn_signal.csv", "vel.csv",
#                       "yaw.csv"]
# HDD_FVECS_PATH = HDD_DATA_BASE_DIR + "/frame_feature_vecs/"
# LOGS_PATH = "/home/rgaurav/Documents/UProjects/auto-intention-snns/logs/"
# HDD_ALL_FRAMES_ALL_VIDS_PATH = HDD_DATA_BASE_DIR + "/frame_feature_vecs/all_frames_all_vids_fvecs_GoogLeNet/"
# HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH = HDD_DATA_BASE_DIR + "/frame_feature_vecs/all_frames_all_vids_fvecs/labels/"
# HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18 = HDD_DATA_BASE_DIR+"/experiment_outputs/vids_2d_cnn_lstm_outputs/pytorch_resnet18_512fvec_outputs/"
# HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT = HDD_DATA_BASE_DIR+"/experiment_outputs/vids_2d_cnn_lstm_outputs/pytorch_googlenet_1024fvec_outputs/"

HDD_DATA_BASE_DIR = os.path.abspath('T:\HONDAdata\HDD_Data')
HDD_SAVED_INDEX_PKL_PATH = os.path.join(HDD_DATA_BASE_DIR, "/EAF_parsing/saved_index.pkl")
HDD_SAVED_INDEX_PKL_PATH = os.path.join(HDD_DATA_BASE_DIR, "EAF_parsing", "saved_index.pkl")
# In HDD_VIDEO_PATH,{0} corresponds to the session ID e.g. 201702271017
HDD_VIDEO_PATH = os.path.join(HDD_DATA_BASE_DIR, "release_2019_07_08","{0}","camera","center")
# In HDD_CBUS_CSV_PATH, {0} corresponds to the session ID and {1} corresponds
# to the CAN Bus data file csv, e.g.: "accel_pedal.csv"
HDD_CBUS_CSV_PATH = os.path.join(HDD_DATA_BASE_DIR,"release_2019_07_08","{0}","general","csv","{1}")
HDD_CBUS_CSV_FILES = ["accel_pedal.csv", "brake_pedal.csv", "rtk_pos.csv",
                      "rtk_track.csv", "steer.csv", "turn_signal.csv", "vel.csv",
                      "yaw.csv"]
HDD_FVECS_PATH = os.path.join(HDD_DATA_BASE_DIR, "frame_feature_vecs")
LOGS_PATH = os.path.join(HDD_DATA_BASE_DIR,"logs")
HDD_ALL_FRAMES_ALL_VIDS_PATH = os.path.join(HDD_DATA_BASE_DIR, "frame_feature_vecs", "all_frames_all_vids_fvecs_GoogLeNet")
HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH = os.path.join(HDD_DATA_BASE_DIR, "frame_feature_vecs","all_frames_all_vids_fvecs","labels")
HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18 = os.path.join(HDD_DATA_BASE_DIR, "experiment_outputs", "vids_2d_cnn_lstm_outputs", "pytorch_resnet18_512fvec_outputs")
HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT = os.path.join(HDD_DATA_BASE_DIR, "experiment_outputs","vids_2d_cnn_lstm_outputs","pytorch_googlenet_1024fvec_outputs")

# Convert to abs path for cross platform ease
HDD_SAVED_INDEX_PKL_PATH = os.path.abspath(HDD_SAVED_INDEX_PKL_PATH)
HDD_VIDEO_PATH = os.path.abspath(HDD_VIDEO_PATH)
HDD_CBUS_CSV_PATH = os.path.abspath(HDD_CBUS_CSV_PATH)
HDD_FVECS_PATH = os.path.abspath(HDD_FVECS_PATH)
HDD_ALL_FRAMES_ALL_VIDS_PATH = os.path.abspath(HDD_ALL_FRAMES_ALL_VIDS_PATH)
HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH = os.path.abspath(HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH)
HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18 = os.path.abspath(HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18)
HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT = os.path.abspath(HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT)

################################################################################
############################ EXPERIMENT PARAMETERS #############################
################################################################################

############################## EVENT TYPE LABELS ###############################
# Note that the event_type lables are not increased by 1, rather kept same as in
# base paper. These are also considered as qualified labels in this work.

HDD_EVENT_TYPES = [
  # (layer, [event_type]])
  # Layer 0: Goal Oriented.
  (0, [1, 6]), # intersection passing, intersection passing
  (0, [7]), # left turn
  (0, [0]), # right turn
  (0, [8]), # crosswalk passing
  (0, [3]), # left lane change
  (0, [5]), # right lane change
  (0, [11]), # left lane branch
  (0, [2]), # merge
  (0, [4]), # right lane branch
  (0, [10]), # railroad passing
  (0, [12]), # U-turn

  # Layer 1: Cause. TODO: FIX it: event_type_idx not fall in [16, 31].
  (1, [39, 83]), # sign
  (1, [16, 41]), # congestion
  (1, [87]), # stop 4 light
  (1, [88]), # Avoid parked car
  (1, [89]), # Stop 4 pedestrian

  # TODO: Move the first column to Cause layer.
  # Layer 3: Attention (layer 5 is also attention-2).
  (3, [19, 38, 73]), # crossing vehicle, crossing vehicle, crossing vehicle
  (3, [18, 35, 78]), # red light, red light, red light
  (3, [22, 44, 75]), # crossing pedestrn, crossing pedestrn, crossing pedestrn
  (3, [28, 37, 79]), # vehicle cut-in, vechicle cut-in, vehicle cut-in
  (3, [23, 40, 76]), # merging vehicle, merging vechile, merging vehicle
  (3, [17, 39, 83]), # sign, sign, sign
  (3, [24, 46, 74]), # on-road bicyclist, on-road bicyclist, on-road bicyclist
  (3, [21, 43, 80]), # yellow light, yellow light, yellow light
  (3, [20, 42, 77]), # parked vehicle, parked vehicle, parked vehicle
  (3, [29, 45, 82]), # road work, road work, road work
  (3, [27, 36, 81]), # on-road motorcclst, on-road motorcclst, on-road motorcclst
  (3, [25, 47, 84]), # pdstrn near ego ln, pdstrn near ego ln, pdstrn near ego ln

  # Layer 6: Stimulus Driven.
  (6, [85, 86, 87, 89, 90]), # stop 4 congestion, stop 4 sign, stop 4 light,
                             # stop 4 pedestrian, stop for others
  (6, [88, 91, 92, 93]) # Avoid parked car, Avoid pedestrian near ego lane, Avoid
                        # on-road bicyclist, Avoid TP
  ]

######################## MAP OF LAYER 0 LABELS TO CLASS ########################
# Note that key values are event_type + 1.
MAP_LBL_TO_CLS_LYR_0 = {
  0: 0, # Background.
  1: 1, # 0-> Right Turn.
  2: 2, # 1-> Intersection Passing.
  3: 3, # 2-> Merge.
  4: 4, # 3-> Left Lane Change.
  5: 5, # 4-> Right Lane Branch.
  6: 6, # 5-> Right Lane Change.
  7: 2, # 6-> Intersection Passing.
  8: 7, # 7-> Left Turn.
  9: 8, # 8-> Crosswalk Passing.
  11: 9, # 10-> Railroad Passing.
  12: 10, # 11-> Left Lane Branch.
  13: 11 # 12-> U-Turn.
}

MAP_LBL_TO_CLS_LYR_1 = {

}
###################### CLASS TYPE LABELS TO EVENT TYPE NAMES ###################
HDD_LAYER1_MAP_CLS_TO_LBL = {
  0: "Background",
  1: "Right Turn",
  2: "Intersection Passing",
  3: "Merge",
  4: "Left Lane Change",
  5: "Right Lane Branch",
  6: "Right Lane Change",
  7: "Left Turn",
  8: "Crosswalk Passing",
  9: "Railroad Passing",
  10: "Left Lane Branch",
  11: "U-Turn"
}

######################## TRAINING AND TEST SESSION IDS #########################
HDD_TRAIN_SIDS = [
            '201702271017', '201702271123', '201702271136', '201702271438',
            '201702271632', '201702281017', '201702281511', '201702281709',
            '201703011016', '201703061033', '201703061107', '201703061323',
            '201703061353', '201703061418', '201703061429', '201703061456',
            '201703061519', '201703061541', '201703061606', '201703061635',
            '201703061700', '201703061725', '201703080946', '201703081008',
            '201703081055', '201703081152', '201703081407', '201703081437',
            '201703081509', '201703081549', '201703081617', '201703081653',
            '201703081723', '201703081749', '201704101354', '201704101504',
            '201704101624', '201704101658', '201704110943', '201704111011',
            '201704111041', '201704111138', '201704111202', '201704111315',
            '201704111335', '201704111402', '201704111412', '201704111540',
            '201706061021', '201706070945', '201706071021', '201706071319',
            '201706071458', '201706071518', '201706071532', '201706071602',
            '201706071620', '201706071630', '201706071658', '201706071735',
            '201706071752', '201706080945', '201706081335', '201706081445',
            '201706081626', '201706081707', '201706130952', '201706131127',
            '201706131318', '201706141033', '201706141147', '201706141538',
            '201706141720', '201706141819', '201709200946', '201709201027',
            '201709201221', '201709201319', '201709201530', '201709201605',
            '201709201700', '201709210940', '201709211047', '201709211317',
            '201709211444', '201709211547', '201709220932', '201709221037',
            '201709221238', '201709221313', '201709221435', '201709221527',
            '201710031224', '201710031247', '201710031436', '201710040938',
            '201710060950', '201710061114', '201710061311', '201710061345',
        ]

HDD_TEST_SIDS = [
            '201704101118', '201704130952', '201704131020', '201704131047',
            '201704131123', '201704131537', '201704131634', '201704131655',
            '201704140944', '201704141033', '201704141055', '201704141117',
            '201704141145', '201704141243', '201704141420', '201704141608',
            '201704141639', '201704141725', '201704150933', '201704151035',
            '201704151103', '201704151140', '201704151315', '201704151347',
            '201704151502', '201706061140', '201706061309', '201706061536',
            '201706061647', '201706140912', '201710031458', '201710031645',
            '201710041102', '201710041209', '201710041351', '201710041448',
        ]
