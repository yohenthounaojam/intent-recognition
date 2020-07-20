#
# Author: Ramashish Gaurav
#
# The HDD dataset has 30 frames per second, each frame of size 720, 1280, 3
# Saving all the frames (as jpg image) of 53 min video takes 28GB of space.
#
# The given "saved_index.pkl" by the HDD dataset authors has the required
# information for creating experimental dataset out of videos.
#

import numpy as np
import os
import pandas as pd
import subprocess
import torch

from pathlib import Path

from . import log
from .cnn_lstm_utils import get_cnn_codes_for_frames_tt
from .consts import (HDD_VIDEO_PATH, HDD_FVECS_PATH, HDD_ALL_FRAMES_ALL_VIDS_PATH,
    HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH)
from .exp_util import (get_saved_index_pkl, get_duration_of_video_in_secs,
                       get_qualified_fvecs_and_labels)

def extract_frames(video_path, start=None, end=None, fps_rate="3/1"):
  """
  Extract frames from the videos. Reads a *.mp4 video and a matrix of frames.

  Args:
    video_path (str): absolute/path/to/video.mp4
    start (float): Start of the video sequence in seconds.
    end (float): End of the video sequence in seconds.
    fps_rate (str): Number of frames to be extracted every second, e.g. "30/1" =>
        thirty frames extracted every second.

  Returns:
    numpy.ndarray : A matrix of frames of shape (# frames, 720, 1280, 3)
  """
  if start and end:
    command = ["ffmpeg",
              "-ss", str(start),
              "-i", video_path,
              "-t", str(end-start),
              # Make sure all frames are of same scale for uniformity in CNN inpt.
              "-vf", "fps=%s,scale=1280:720" % fps_rate,
              "-f", "image2pipe",
              "-pix_fmt", "rgb24",
              "-vcodec", "rawvideo",
              "-"]
  else:
    command = ["ffmpeg",
               "-i", video_path,
               "-vf", "fps=%s,scale=1280:720" % fps_rate,
               "-f", "image2pipe",
               "-pix_fmt", "rgb24",
               "-vcodec", "rawvideo",
               "-"]

  process = subprocess.Popen(
      command, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
      bufsize=10 ** 7)

  nbytes = 3*720*1280 # Number of pixels in image.
  frames = []
  frame_count = 0

  while True:
    byte_string = process.stdout.read(nbytes)
    if len(byte_string) != nbytes:
      break
    else:
      frame = np.fromstring(byte_string, dtype="uint8")
      frame = frame.reshape(720, 1280, 3) # Shape the frame as desired.
      frames.append(frame)
      frame_count += 1

  process.wait()
  del process
  if len(frames) == 0:
    log.INFO("No frames found, perhaps the video_path is wrong. Check it.")
  return np.asarray(frames, dtype="uint8")

def prepare_hdd_dataset_sequence_frames(model, max_clip_secs=13, fps_rate="3/1"):
  """
  Saves the file of frame sequence for each the video clip in HDD dataset whose
  information is provided in original `saved_index.pkl`.

  The sequence frames are stored in dictionary with keys as index value of the
  clip information (i.e. session_id, start, end) in the original dataframe in
  `saved_index.pkl` and values as the tuple of 3D torch Tensor of shape (
  1 x num_frames x size of one frame's feature vector) e.g. (1, 15, 1024) and
  the tuple of (layer, event_type). Note that no event_type in any layer is
  common.

  The frames feature vecs are saved in dict format:
  {int: (torch.Tensor, (int, int))} : where key is the index value (of the row
      in "event_pd" dataframe, value is a tuple with first element being a torch
      Tensor of 3D shape: (1 x seq_len x dimension of extracted features) and
      second element is a tuple of (layer, event_type).


  Args:
    model (torchvision.models): Torch vision pretrained models.
    max_clip_secs (int): Maximum duration of allowed clips.
    fps_rate (str): Number of frames to be extracted per second.
  """
  saved_idx_pkl = get_saved_index_pkl()
  df = saved_idx_pkl["events_pd"]
  frame_vecs_dict = {}
  frame_vecs_idx_lst = []
  for index in df.index:
    row = df.loc[index]
    layer, event_type, session_id, start, end = (row["layer"],
        row["event_type"], row["session_id"], row["start"], row["end"])
    if (end-start)/1000 <= max_clip_secs:
      files = os.listdir(HDD_VIDEO_PATH.format(session_id))
      frames = extract_frames(
          HDD_VIDEO_PATH.format(session_id)+"/"+files[0], start/1000, end/1000,
          fps_rate)
      ff_vecs = get_cnn_codes_for_frames_tt(model, frames, llyr=-1)
      frame_vecs_dict[index] = (
          ff_vecs.reshape(1, ff_vecs.shape[0], ff_vecs.shape[1]),
          (layer, event_type))
      frame_vecs_idx_lst.append(index)
    log.INFO("Index %s Done" % index)

  np.save(HDD_FVECS_PATH + "hdd_frame_vecs_dict_fps_%s_model_%s.npy" % (
      "_".join(fps_rate.split("/")), model._get_name()), frame_vecs_dict)
  np.save(HDD_FVECS_PATH + "hdd_frame_vecs_idx_lst_fps_%s_model_%s.npy" % (
      "_".join(fps_rate.split("/")), model._get_name()), frame_vecs_idx_lst)

def _get_data_and_labels(idcs_dict, ffvecs_data, cls_to_et_map):
  """
  Args:
    idcs_dict ({}): {class (int): [indices (int)]}
    ffvecs_data ({int: (torch.Tensor(3D), (layer, event_type))}): Frames Feature
    cls_to_et_map ({int: [int]}): {class: [event types]} e.g. {6: [11]}.

  Returns:
    [pytorch.Tensor(3D)], [int]: List of seq frame features, List of classes.
  """
  data, labels = [], []
  for cls, idcs_lst in idcs_dict.items():
    for idx in idcs_lst:
      ffvec_data = ffvecs_data.item().get(idx)
      if not ffvec_data:
        log.INFO("No feature vector data for index: %s found." % idx)
        continue

      if cls_to_et_map:
        assert ffvec_data[1][1] in cls_to_et_map[cls]

      data.append(ffvec_data[0])
      labels.append(cls)

  return data, labels

def get_train_test_tensors_and_labels_lst(
    train_idcs_dict, test_idcs_dict, ffvecs_data, cls_to_et_map=None):
  """
  Returns training and test tensors for a particular layer, whose info is
  mentioned in `train_idcs_dict`, `test_idcs_dict`, and `cls_to_et_map`.

  Args:
    train_idcs_dict ({}): {class (int): [indices (int)]} e.g. {6: [12530, ...]}
    test_idcs_dict ({}): {class (int): [indices (int)]}
    ffvecs_data ({int: (torch.Tensor(3D), (layer, event_type))}): Frames Feature
    cls_to_et_map ({int: [int]}): {class: [event types]} e.g. {6: [11]}.

  Returns:
    [pytorch.Tensors(3D)], [int], [pytorch.Tensors(3D)], [int]
  """
  train_data, train_labels = _get_data_and_labels(
      train_idcs_dict, ffvecs_data, cls_to_et_map)
  test_data, test_labels = _get_data_and_labels(
      test_idcs_dict, ffvecs_data, cls_to_et_map)

  return train_data, train_labels, test_data, test_labels

def get_all_frames_labels_all_vids(session_ids_lst, layer, fps):
  """
  Creates labels of each frames in all videos.

  Args:
    session_ids_lst ([int]): A list of videos' session IDs.
    layer (int): Layer of the HDD dataset.
    fps (str): Frame rate per second e.g. "3/1" => 3 frames per second.
  """
  samp_freq = int(fps.split("/")[0])
  saved_index = get_saved_index_pkl()
  events_pd = saved_index["events_pd"]
  layer_events_pd = events_pd[events_pd["layer"] == layer]

  for session_id in session_ids_lst:
    files = os.listdir(HDD_VIDEO_PATH.format(session_id))
    total_duration_secs = get_duration_of_video_in_secs(
        HDD_VIDEO_PATH.format(session_id)+"/"+files[0])
    num_frames = int(samp_freq * total_duration_secs)
    session_labels = np.zeros(num_frames)

    session_rows = layer_events_pd[layer_events_pd["session_id"] == session_id]
    for row in session_rows.iterrows():
      start, end = (int(row[1]["start"]/1000 * samp_freq),
                    int(row[1]["end"] / 1000 * samp_freq))
      # Since the background (i.e. frames with no event_type) is labelled 0,
      # and right_turn is also labelled 0, increment each event_type label by 1.
      # Make sure while interpreting results, predicted event_type labels are
      # reduced by 1. This is generalized to all the layers.
      session_labels[start:end] = row[1]["event_type"] + 1

    np.save(HDD_ALL_FRAMES_ALL_VIDS_LABELS_PATH + "/session_%s_layer_%s.npy" % (
            session_id, layer), session_labels)
    log.INFO("Labels for session: %s, layer: %s done." % (session_id, layer))

def get_all_frames_fvecs_all_vids(session_ids_lst, model, fps, batch_size=40):
  """
  Create extracted feature vecs of all frames in each video using a
  pretrained 2D CNN model. Note that you might need to lower down the
  `batch_size` in case deeper models are used like GoogLeNet.

  Args:
    session_ids_lst [int]: A list of videos' session IDs.
    model (torchvision.models): Pretrained 2D CNN model.
    fps (str): Frame rate per second e.g. "3/1" => 3 frames per second
    batch_size (int): The batch size of frames to be sent to the `model`.
  """
  for session_id in session_ids_lst:
    files = os.listdir(HDD_VIDEO_PATH.format(session_id))
    log.INFO("All frames extraction for session: %s starting...." % session_id)
    all_frames = extract_frames(
        HDD_VIDEO_PATH.format(session_id)+"/"+files[0], fps_rate=fps)
    num_frames = all_frames.shape[0]
    frames_fvecs_lst = []
    log.INFO("All frames extraction for session: %s done." % session_id)
    for start in range(0, num_frames, batch_size):
      end = min(start+batch_size, num_frames)
      frames_vecs = get_cnn_codes_for_frames_tt(model, all_frames[start:end])
      for frame_vec in frames_vecs:
        frames_fvecs_lst.append(frame_vec.view(frame_vec.shape[0]).numpy())
    log.INFO("Feature vecs of all frames for session: %s done. Now saving..."
              % session_id)
    np.save(
        HDD_ALL_FRAMES_ALL_VIDS_PATH + "/session_%s_model_%s.npy" % (
        session_id, model._get_name()), frames_fvecs_lst)
    del frames_fvecs_lst
    del all_frames
    log.INFO("Saving feature vecs for session: %s done." % session_id)

def get_sequence_wise_fvecs_and_clss(session_id, layer, seq_size,
                                     map_lbl_to_cls, model):
  """
  Returns the tuple of frames features vecs and corresponding labels of a
  session `session_id`.

  Args:
    session_id (str): Session ID of the video.
    layer (int): The layer of action recognition.
    seq_size (int): The number of frames in the considered sequence.
    map_lbl_to_cls (dict): The map of qualified labels to class.
    model (str): The 2D CNN's model name which was used to extract features.

  Returns:
    (numpy.ndarray, numpy.ndarray): Tuple of fvecs (num_frames x fvecs dim) and
                                    labels (

    Note: Both the numpy arrays are of same shape.
  """
  qualified_all_fvec, qualified_all_fcls = get_qualified_fvecs_and_labels(
      session_id, layer, map_lbl_to_cls, model=model)

  # Calculate the length of qualified data and create the sequence wise data.
  qfd_len = qualified_all_fvec.shape[0]
  assert qfd_len == qualified_all_fcls.shape[0]

  ret_fvec, ret_fcls = [], []
  for start in range(0, qfd_len, seq_size):
    end = min(start+seq_size, qfd_len)
    ret_fvec.append(qualified_all_fvec[start:end])
    ret_fcls.append(qualified_all_fcls[start:end])

  return np.array(ret_fvec), np.array(ret_fcls)

def get_batch_size_one_data_loader(session_id, layer, seq_size, map_lbl_to_cls,
                                   model):
  """
  Args:
    session_id (str): The session ID of the video.
    layer (int): The layer.
    seq_size (int): The sequence size.
    map_lbl_to_cls (dict): A dict of mapping from event_type labels to classes.
    model (str): Name of the pretrained 2D CNN model to extract features.

  Returns: <generator> of (frames features, frames class)
  """
  fvecs, clss = get_sequence_wise_fvecs_and_clss(session_id, layer, seq_size,
                                                 map_lbl_to_cls, model=model)
  for fvec, cls in zip(fvecs, clss):
    fvec = torch.from_numpy(fvec).unsqueeze(0) # [seq_size, :] -> [1, seq_size, :]
    cls = torch.from_numpy(cls).unsqueeze(0) # [seq_size] -> [1: seq_size]
    yield fvec, cls
