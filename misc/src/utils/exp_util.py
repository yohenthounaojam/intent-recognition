#
# This file contains the general utility functions for the experiment.
#
# Author: Ramashish Gaurav
#


from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight

import dateutil.parser as dp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sn
import subprocess

from . import log
from .consts import (HDD_SAVED_INDEX_PKL_PATH, HDD_VIDEO_PATH, HDD_EVENT_TYPES,
                     HDD_CBUS_CSV_PATH, HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH,
                     HDD_TRAIN_SIDS, HDD_ALL_FRAMES_ALL_VIDS_PATH, HDD_TEST_SIDS,
                     MAP_LBL_TO_CLS_LYR_0)

def get_saved_index_pkl():
  return pickle.load(open(HDD_SAVED_INDEX_PKL_PATH, "rb"))

def get_layer_and_event_type_df(layer, event_type=None):
  """
  Returns the data frame of session_ids (each video as a unique session_id), the
  corresponding start time and end time of the `layer` and `event_type`. The
  start time and end time are mentioned in milliseconds.

  Args:
    layer (int): The layer id of the driver behaviour. Ex: 0 => Goal Oriented.
    event_type (int): The corresponding event type in `layer`. For 0 layer i.e.
        Goal Oriented action, the even type could be 7 i.e. left turn.

  Note that the `event_type` if given should fall in `layer`, if not, the data
  frame returned would be empty.

  Returns:
    pandas.DataFrame
  """
  sidx = get_saved_index_pkl()
  df = sidx["events_pd"]
  if event_type is not None:
    return df[(df["layer"] == layer) & (df["event_type"] == event_type)]
  else:
    return df[df["layer"] == layer]

def get_epochs_from_session_id_st_en(session_id, start, end):
  """
  Returns epochs for the start and end of an event for a particular session.

  Note that `session_id_fname` stores the actual time of the start of the
  session. At approximately the same time the CAN bus data starts to be stored.

  Args:
    session_id (str): The session id of video.
    start (int): The start time in milliseconds of the clip.
    end (int): The end time in milliseconds of the clip.

  Returns:
    int, int : The epoch for start, The epoch for end.
  """
  session_id_fname = os.listdir(HDD_VIDEO_PATH.format(session_id))[0]
  year, month, day, hour, mins, secs = session_id_fname.split("_")[0].split("-")
  sid_dt = datetime(
      int(year), int(month), int(day), int(hour), int(mins), int(secs))
  start_epoch = int((sid_dt + timedelta(seconds=start/1000)).strftime("%s"))
  end_epoch = int((sid_dt + timedelta(seconds=end/1000)).strftime("%s"))
  return start_epoch, end_epoch

def get_epochs_from_iso_timestamps_lst(iso_ts_lst):
  """
  Returns a list of epochs for ISO timestamps. Note that this function get rids
  of the TimeZone. The ISO time format it accepts is:
    "YYYY-MM-DDThh:mm:ss.sTZD" e.g.: 1997-07-16T19:20:30.45+01:00
  Here TZD is the Time Zone Designator (Z or +hh:mm or -hh:mm) and this function
  gets rid of this TZD or +hh:mm or -hh:mm.

  Args:
    iso_ts_lst ([str]): A list of ISO timestamps.

  Returns:
    numpy.ndarray: An array of corresponding epoch.
  """
  length = len(iso_ts_lst)
  iso_epoch_arr =  np.zeros(length)
  for i in range(length):
    iso_epoch_arr[i] = int(dp.parse(iso_ts_lst[i][:-6]).strftime("%s"))

  return iso_epoch_arr

def basic_plot(y, y_label="", title=""):
  """
  Plots a basic plot.

  Args:
    y ([float]): A list of numbers to be plotted.
    y_label (str): Label of the numbers to be plotted.
  """
  fig, ax = plt.subplots(figsize=(10, 6))
  fig.suptitle(title)
  ax.plot(y)
  ax.set_ylabel(y_label)
  fig.show()

def get_event_types_for_layer_lst(layer):
  """
  Returns a list of event types for the corresponding layer.

  Args:
    layer (int): The layer, e.g. 0 => Operation_Goal-Oriented.

  Returns:
    [[]]: A list of lists.
  """
  event_types_lst = []
  for tpl in HDD_EVENT_TYPES:
    if tpl[0] == layer:
      event_types_lst.append(tpl[1])

  return event_types_lst

def get_can_bus_corresponding_to_session_id_st_en_dict(session_id, start, end,
    can_bus_data_lst):
  """
  Returns a dict of keys as CAN bus parameter and values as the corresponding
  CAN bus parameter's values.

  Args:
    session_id (str): The session ID of the video.
    start (int): The start of the video clip in milliseconds.
    end (int): The end of the video clip in milliseconds.
    can_bus_data_lst ([str]): A list of CAN bus data file names. e.g. ["file.csv"]

  Returns:
    dict{CAN bus param: Values}
  """
  can_bus_values_dict = {}

  clip_st_epoch, clip_en_epoch = get_epochs_from_session_id_st_en(
      session_id, start, end)

  for file_name in can_bus_data_lst:
    csv_path = HDD_CBUS_CSV_PATH.format(session_id, file_name)
    csv_df = pd.read_csv(csv_path)
    csv_df["iso_timestamp"] = get_epochs_from_iso_timestamps_lst(
        csv_df["iso_timestamp"].tolist())

    can_param_dict = {}
    for col in csv_df.columns[2:]:
      can_param_dict[col] = csv_df[(csv_df["iso_timestamp"] >= clip_st_epoch) &
          (csv_df["iso_timestamp"] <= clip_en_epoch)][col].tolist()
    can_bus_values_dict[file_name.split(".")[0]] = can_param_dict
  return can_bus_values_dict

def get_can_bus_df_from_clipped_dikt_df(can_vals_dikt):
  """
  Returns a data frame for the clipped CAN bus data passed in `dikt`.
  Note that the number of rows in Data Frame is equal to the length of the
  shortest CAN bus parameter's values.

  For parameters which have larger length of values, they are sampled at
  equidistant points.

  Args:
    can_vals_dikt ({}): A dict of CAN bus parameter values.

  Returns:
    pandas.DataFrame
  """
  df_params_vals_lst = []
  df_params_cols = []
  param_vals_lsts_len = []

  for key, params_dikt in can_vals_dikt.items():
    for param_key, param_vals_lst in params_dikt.items():
      param_vals_lsts_len.append(len(param_vals_lst))

  smallest_length = min(param_vals_lsts_len)

  for key, params_dikt in can_vals_dikt.items():
    for param_key, param_vals_lst in params_dikt.items():
      indices = np.round(
          np.linspace(0, len(param_vals_lst)-1, smallest_length)).astype(int)
      df_params_cols.append(param_key)
      df_params_vals_lst.append(np.array(param_vals_lst)[indices])

  return pd.DataFrame(np.array(df_params_vals_lst).T, columns=df_params_cols)

def get_layer_train_test_indices_cls_map_dict(layer, div_frc=0.7, seed=7879):
  """
  Creates the training and test indices for a layer. For each event type, this
  function randomly selects `div_frc` number of indices for training and rest
  for testing.

  Args:
    layer (int): The layer e.g. 0 => Operation_Goal-Oriented.
    div_frc (float): The training data fraction (1-`div_frc` is test fraction).
    seed (int): The seed value to reproduce results.

  Return:
    {}, {}, {}: Where key is integer and value is a list. For the first dict
        the list is a list of event type e.g. {6: [11]} where 6 is the class
        of event_type 11. For the last two dicts the list is a list of indices,
        e.g. {6: [12530, ...]} where 6 is the class and 12530 is the index.
  """
  train_idcs_dict, test_idcs_dict, class_to_et_map = {}, {}, {}

  layer_et_lst = get_event_types_for_layer_lst(layer)
  for i, et_lst in enumerate(layer_et_lst):
    train_idcs_lst, test_idcs_lst = [], []
    for et in et_lst:
      et_idcs = get_layer_and_event_type_df(layer, et).index
      if len(et_idcs) == 0:
        print("WARNING: Empty indices list for layer: %s, event type: %s"
              % (layer, et))
        continue
      random.seed(seed)
      train_idcs = random.sample(et_idcs.tolist(), int(div_frc*len(et_idcs)))
      test_idcs = list(set(et_idcs.tolist()) - set(train_idcs))
      train_idcs_lst.extend(train_idcs)
      test_idcs_lst.extend(test_idcs)

    class_to_et_map[i] = et_lst
    train_idcs_dict[i] = train_idcs_lst
    test_idcs_dict[i] = test_idcs_lst

  return train_idcs_dict, test_idcs_dict, class_to_et_map

def plot_conf_mat_heat_map(y_true, y_pred, metric="precision", y_annot=True):
  """
  Plots a heat map of the confusion matrix. Note: `normalize` over 'index' is
  Precision and over 'columns' is Recall.

  Args:
    y_true (np.array(int)): True labels of the test samples.
    y_pred (np.array(int)): Predicted labels of the test samples.
    metric (str): "precision"|"recall" -> Metric to be reported.
    y_annot (bool): Plot the `metric` on heatmap if True else not.
  """
  if metric=="precision":
    normalize = "index"
  elif metric=="recall":
    normalize = "columns"
  else:
    log.ERROR("Invalid metric: %s for calculating heat map." % metric)
    sys.exit()

  conf_mat = pd.crosstab(y_pred, y_true, rownames=["Predicted"],
                         colnames=["True"], normalize=normalize).round(4)*100
  #print(conf_mat)
  plt.figure(figsize=(8,6))
  sn.heatmap(conf_mat, annot=y_annot)
  plt.show()

def get_exp_output_metrics(y_true, y_pred, pred_scores, num_clss, avg=None):
  """
  Returns the metrics: Precision, Recall, Average Precision (AP), Mean AP.

  Args:
    TODO

  Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
  """
  classes = list(range(num_clss))
  # Check if y_true has classes same as in a particular layer.
  assert np.unique(y_true).tolist() == classes
  y_true_binary = label_binarize(y_true, classes=classes)

  prcsn_score = precision_score(y_true, y_pred, average=avg)
  recll_score = recall_score(y_true, y_pred, average=avg)
  avg_prcsn_score = average_precision_score(
      y_true_binary, pred_scores, average=avg)

  return prcsn_score, recll_score, avg_prcsn_score

def get_duration_of_video_in_secs(filename):
  """
  Returns the duration of video `filename` in seconds.

  Args:
    filename (str): Path/to/video.mp4

  Returns:
    float : Seconds.
  """
  result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", filename],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return float(result.stdout)

def get_map_of_unique_classes_for_layer(session_ids, layer):
  """
  Returns the total number of unique classes in a layer and the dict of mapping
  of HDD layer labels to experiment classes (in range 0 to total labels -1 ).
  Also makes sure that the labels (keys) in map are in the qualified range of
  labels for the layer (mentioned in consts.py).

  Args:
    session_ids [str]: Session IDs of videos.
    layer (int): The layer.

  Returns:
    int, dict, dict: Number of unique labels (which is equal to number of
        classes), map of label to class, map of class to label.
  """
  all_unique_labels = set()
  map_cls_to_lbl = defaultdict(list)
  if layer == 0:
    map_lbl_to_cls = MAP_LBL_TO_CLS_LYR_0
  elif layer == 1:
    map_lbl_to_cls = MAP_LBL_TO_CLS_LYR_1

  all_qualified_labels = get_event_types_for_layer_lst(layer)
  for session_id in session_ids:
    labels = np.load(
        HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" %
        (session_id, layer), allow_pickle=True)
    for label in labels:
      for ql in all_qualified_labels:
        if label in ql:
          all_unique_labels.add(label+1) # The saved labels are event_type + 1.
          break

  # Map the cls 0 to label 0 where label 0 is always the background by default.
  map_cls_to_lbl[0] = [0]
  for lbl in all_unique_labels:
    cls = map_lbl_to_cls[lbl]
    map_cls_to_lbl[cls].append(lbl)

  return len(map_cls_to_lbl.keys()), map_lbl_to_cls, map_cls_to_lbl

def get_count_per_class_in_layer(layer, map_lbl_to_cls, model):
  """
  Returns a list of each class instances' counts in the training data for the
  specified `layer`.

  Args:
    layer (int): The layer.
    map_lbl_to_cls (dict): The map of qualified labels to class.
    model (str): The 2D CNN's model name which was used to extract features.

  Returns:
    dict, int: The map of class to number of instances, total number of instances.

  Note: Qualified labels are the ones that are present in the consts.py for the
        particular `layer`.
  """
  cls_inst_count_dict = defaultdict(int)
  num_inst = 0
  for session_id in HDD_TRAIN_SIDS:
    _, qualified_all_fcls = get_qualified_fvecs_and_labels(
        session_id, layer, map_lbl_to_cls, model=model)
    for fcls in qualified_all_fcls:
      cls_inst_count_dict[fcls] += 1
      num_inst += 1

  return cls_inst_count_dict, num_inst

def get_qualified_fvecs_and_labels(session_id, layer, map_lbl_to_cls, model):
  """
  Returns the qualified fvecs and labels for a session. The qualified labels
  and fvecs correspond to the ones that are mentioned for the `layer` in
  consts.py. Note that the returned qualified labels are mapped to a class.

  Args:
    session_id (str): The session ID of the video.
    layer (int): The layer of action recognition.
    map_lbl_to_cls (dict): The map of qualified labels to class.
    model (str): The 2D CNN's model name which was used to extract features.

  Returns:
    numpy.ndarray, numpy.ndarray: The tuple of qualified fvecs and labels.
  """
  # Get the frames vecs and labels for a particular session ID.
  all_frames_fvecs = np.load(
      HDD_ALL_FRAMES_ALL_VIDS_PATH + "/session_%s_model_%s.npy" %
      (session_id, model), allow_pickle=True)
  all_frames_labels = np.load(
      HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" %
      (session_id, layer), allow_pickle=True)

  # Make sure that `all_frames_fvecs` and `all_frames_labels` are of same length.
  min_len = min(all_frames_fvecs.shape[0], all_frames_labels.shape[0])
  all_frames_fvecs = all_frames_fvecs[:min_len]
  all_frames_labels = all_frames_labels[:min_len]

  # Map the labels to their classes. `map_lbl_to_cls` has the mapping, as well as
  # the keys (or labels) are qualified as mentioned in consts.py for a layer.
  qualified_all_fvec, qualified_all_fcls = [], []
  for fvec, flabel in zip(all_frames_fvecs, all_frames_labels):
    if flabel in map_lbl_to_cls.keys():
      qualified_all_fvec.append(fvec)
      qualified_all_fcls.append(map_lbl_to_cls[flabel])

  qualified_all_fvec = np.array(qualified_all_fvec)
  qualified_all_fcls = np.array(qualified_all_fcls)

  return qualified_all_fvec, qualified_all_fcls

def get_accuracy_of_predicted_classes(pred_clss, test_clss, ignore_clss=[]):
  """
  Returns the accuracy of predicted classes. If `ignore_clss` is given, then
  while calculating the accuracy, those test classes are ignored.

  Args:
    pred_clss [int]: Predicted classes.
    test_clss [int]: Actual classes
    ignore_clss [int]: The classes to be ignored.

  Returns:
    float: The accuracy.
  """
  total_num_clss = 0
  acc = 0
  for pred_cls, test_cls in zip(pred_clss, test_clss):
    if test_cls in ignore_clss:
      continue
    total_num_clss += 1
    if test_cls == pred_cls:
      acc += 1

  if total_num_clss == 0:
    return 0.0

  return acc/total_num_clss

def get_sklbal_weights_per_class_in_layer(layer, map_lbl_to_cls, model):
  """
  sk-learn class_weight function is used to obtain the balanced class weights.
  Returns a list of class weights for classes in order of increasing notation.
  That is first weight corresponds to class 0, second weight corresponds to
  class 1 and so on..

  Args:
    layer (int): The layer.
    map_lbl_to_cls (dict): The map of qualified labels to class.
    model (str): The 2D CNN's model name which was used to extract features.

  Return:
    [float]
  """
  all_instance_classes = []
  for session_id in HDD_TRAIN_SIDS:
    _, qualified_all_fcls = get_qualified_fvecs_and_labels(
        session_id, layer, map_lbl_to_cls, model=model)
    for fcls in qualified_all_fcls:
      all_instance_classes.append(fcls)

  return class_weight.compute_class_weight(
      "balanced", np.unique(all_instance_classes), all_instance_classes)

def get_invbal_weights_per_class_in_layer(layer, map_lbl_to_cls, num_clss,
                                          model="ResNet"):
  """
  Simple inverse of class occurrence frequency is used to obtain class weights.
  Returns a list of class weights for classes in order of increasing notation.
  That is first weight corresponds to class 0, second weight corresponds to
  class 1 and so on..

  Args:
    layer (int): The layer.
    map_lbl_to_cls (dict): The map of qualified labels to class.
    num_clss (int): Number of classes in `layer`.
    model (str): The 2D CNN's model name which was used to extract features.

  Return:
    [float]
  """
  cls_weights = []
  cls_inst_count_dict, num_inst = get_count_per_class_in_layer(
      layer, map_lbl_to_cls, model=model)
  for cls in range(num_clss):
    if cls in cls_inst_count_dict.keys():
      cls_weights.append(1/cls_inst_count_dict[cls])
    else:
      log.ERROR("Found a class: %s which is not present in class count dict. "
                "Exiting the process." % cls)
      sys.exit()

  return cls_weights

def construct_all_y_pred_y_actual(files_dir, session_ids=HDD_TEST_SIDS):
  """
  Constructs the tuple of predicted output and actual output for all the sessions.

  Args:
    files_dir (str): The pickle files directory,
    session_ids ([str]): A list of session IDs.

  Returns:
    {}: {epoch_num: [List of predicted classes], [List of actual classes]}.
  """
  files = os.listdir(files_dir)
  epochs_output = {}

  for f in files: # Each file corresponds to each epoch.
    if f.endswith(".p"): # i.e. it is a pickle file containing labels.
      epoch = int(f.split("_")[1])
      epoch_output = pickle.load(open(files_dir + "/%s" % f, "rb"))
      pred_clss, actual_clss, pred_scores = [], [], []
      for session_id in session_ids:
        if session_id not in epoch_output.keys():
          print("Session ID: %s output not found in pickle file: %s"
                % (session_id, f))
          continue
        session_output = epoch_output[session_id]
        if len(session_output) != 3:
          break
        for pred_cls, actual_cls, pred_score in zip(
              session_output[0], session_output[1], session_output[2]):
          pred_clss.append(pred_cls)
          actual_clss.append(actual_cls)
          pred_scores.append(pred_score)

      epochs_output[epoch] = (pred_clss, actual_clss, pred_scores)

  return epochs_output
