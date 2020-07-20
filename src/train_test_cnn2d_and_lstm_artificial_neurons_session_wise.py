#
# This file uses a 2D CNN's extracted features + LSTM for intent recognition on
# HDD dataset. This is done session wise, i.e. we train and test the models for
# every session. Artificial Neurons are employed.
#
# Author: Ramashish Gaurav
#
# TODO: Check if hidden states are passed correctly for the last batch?
# Do Class Balancing. Assert that mapping of labels to class is same in Training
# And Test Data.
#

import datetime
import numpy as np
import pickle
import sys
import torch
import torch.nn as nn

from utils import log
from utils.cnn_lstm_utils import LSTMIR, FocalLoss
from utils.consts import (LOGS_PATH, HDD_TRAIN_SIDS, HDD_TEST_SIDS,
    HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18, HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT)
from utils.data_prep_util import (get_sequence_wise_fvecs_and_clss,
                                  get_batch_size_one_data_loader)
from utils.exp_util import (
    get_map_of_unique_classes_for_layer, get_count_per_class_in_layer,
    get_accuracy_of_predicted_classes, get_sklbal_weights_per_class_in_layer,
    get_invbal_weights_per_class_in_layer)

from torch.utils.data import TensorDataset, DataLoader

def _get_data_loader(
    session_id, layer, batch_size, seq_size, map_lbl_to_cls, model):
  """
  Returns the pytorch data loader.

  Args:
    session_id (str): The session ID of the video.

  Returns:
    pytorch.DataLoader
  """
  fvecs, labels = get_sequence_wise_fvecs_and_clss(
      session_id, layer, seq_size, map_lbl_to_cls, model=model)
  # Get rid of the last list element since it may not have `seq_size` fvecs.
  fvecs = torch.stack([torch.from_numpy(fvec) for fvec in fvecs[:-1]])
  labels = torch.stack([torch.from_numpy(label) for label in labels[:-1]])
  dataset = TensorDataset(fvecs, labels)
  data_loader = DataLoader(
      dataset=dataset, batch_size=batch_size, shuffle=False)
  return data_loader

def _get_exp_metadata(layer, model, device, weight_bal, use_fl=True, gamma=2.0):
  """
  Returns the experiment's metadata.

  Args:
    layer (int): The layer.
    model (str): One of "ResNet18"|"GoogLeNet".
    device (torch.Device): Device type of GPU or CPU.
    invbal (bool): Use inverse class frequency as weights if True else sk-learn
                   balanced.
    use_fl (bool): Use FocalLoss criterion if True else use CrossEntropyLoss.
    gamma (float): Gamma paramter in FocalLoss.
  """
  num_clss, map_lbl_to_cls, map_cls_to_lbl = (
      get_map_of_unique_classes_for_layer(HDD_TRAIN_SIDS, layer))

  if weight_bal=="inv_bal":
    log.INFO("Using inverse balanced class weights.")
    cls_weights = get_invbal_weights_per_class_in_layer(
        layer, map_lbl_to_cls, num_clss, model=model)
  elif weight_bal=="skl_bal":
    log.INFO("Using sklearn balanced class weights.")
    cls_weights = get_sklbal_weights_per_class_in_layer(
        layer, map_lbl_to_cls, model=model)
  elif weight_bal=="non_bal":
    log.INFO("Using no class weights.")
    cls_weights = None
  else:
    log.ERROR("Invalid entry for `weight_bal`. Exiting...")
    sys.exit()

  if use_fl:
    if cls_weights:
      criterion = FocalLoss(weights=torch.tensor(cls_weights).float().to(device),
                            gamma=gamma)
    else:
      log.ERROR("Not Implemented")
      sys.exit()
  else:
    if cls_weights:
      criterion = nn.CrossEntropyLoss(
          weight=torch.tensor(cls_weights).float().to(device))
    else:
      criterion = nn.CrossEntropyLoss()

  return num_clss, map_lbl_to_cls, map_cls_to_lbl, cls_weights, criterion

def train_test_session_wise_2D_CNN_plus_LSTM(
    train_sids, test_sids, input_dim, exp_out_dir, batch_size=40, seq_size=90,
    dropout=0.1, num_layers=2, lrng_rate=0.0001, num_epochs=500, layer=0,
    weight_bal="inv_bal", use_fl=True, gamma=2.0, model="ResNet"):
  """
  Trains the features of all frames extracted in each session over the LSTM.

  Args:
    train_sids ([str]): Training session IDs.
    test_ids([str]): Test session IDs.
    input_dim (int): The input dimension of visual features.
    exp_out_dir (str): The experiment output dir to save results and logs.
    batch_size (int): Batch size of data.
    seq_size (int): Sequence size of the input frames.
    dropout (float): Dropout for regularization.
    num_layers (int): Number of LSTM layers.
    lrng_rate (float): Learning rate of the LSTM.
    num_epochs (int): Number of training epochs over whole training data.
    layer (int): The layer for which this model has to be trained and tested.
    invbal (bool): Use inverse class frequency as weights if True else sk-learn
                   balanced.
    use_fl (bool): Use FocalLoss criterion if True else use CrossEntropyLoss.
    gamma (float): Gamma parameter in FocalLoss.
    model (str): Pretrained 2D CNN model's name which was used to extract fvecs.
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  log.INFO("2D CNN Model: %s, Obtaining metadata for layer: %s" % (model, layer))
  # Get the total number of classes in the training session IDs and mappings for
  # the layer, class weights and loss criterion.
  num_clss, map_lbl_to_cls, map_cls_to_lbl, cls_weights, criterion = (
      _get_exp_metadata(layer, model, device, weight_bal, use_fl=use_fl,
                        gamma=gamma))
  log.INFO("Number of classes: %s" % num_clss)
  log.INFO("Mapping of label to class: %s" % map_lbl_to_cls)
  log.INFO("Mapping of class to label: %s" % map_cls_to_lbl)
  log.INFO("Class weights: %s" % cls_weights)
  log.INFO("Loss function: %s" % criterion)
  log.INFO("Batchsize: %s" % batch_size)

  # Configure the LSTM.
  log.INFO("Configuring the LSTM...")
  hidden_dim = 2*input_dim
  lstm_model = LSTMIR(
      input_dim, hidden_dim, num_layers,  num_clss, dropout, batch_size)
  lstm_model.to(device)
  optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=lrng_rate)
  log.INFO("LSTM Model: %s" % lstm_model)
  log.INFO("Back Propagation Optimizer: %s" % optimizer)

  def _execute_model(epoch, mode, session_ids, layer, batch_size, seq_size,
                     map_lbl_to_cls, model="ResNet"):
    """
    Executes the model for the given mode.

    Args:
      epoch (int): The epoch number of training or testing phase.
      mode (str): "train" or "test".
      session_ids ([str]): The session IDs of the videos.
      layer (int): The layer.
      batch_size (int): Batch size.
      seq_size (int): The sequence size of frames.
      map_lbl_to_cls (dict): The map of labels to class.
      model (str): The model name of 2D CNN used to extract frames feature vecs.
    """
    if mode == "train":
      lstm_model.train()
    else:
      lstm_model.eval()

    all_session_loss, all_session_acc = [], []
    epoch_results = {}
    for session_id in session_ids:

      if batch_size == 1:
        data_loader = get_batch_size_one_data_loader(
            session_id, layer, seq_size, map_lbl_to_cls, model)
      else:
        data_loader = _get_data_loader(
            session_id, layer, batch_size, seq_size, map_lbl_to_cls, model)

      # For each session which is one video, initialize the hidden state to None.
      hidden_state = None
      each_session_loss = []
      pred_clss, actual_clss, pred_scores = [], [], []
      for fvecs, clss in data_loader: # Batch wise iteration for each session.
        if mode == "train":
          # Manually set the gradient to 0 for each batch which is one complete
          # pass through the LSTM model.
          optimizer.zero_grad()
        if hidden_state: # Shape: [# lstm layers, batch size, hidden dim]
          hidden_state = (hidden_state[0][:, :fvecs.shape[0], :].contiguous(),
                          hidden_state[1][:, :fvecs.shape[0], :].contiguous())
        output, hidden_state = lstm_model(
            fvecs.to(device).requires_grad_(), hidden_state)
        output, clss = output.view(-1, num_clss), clss.view(-1).long()
        loss = criterion(output, clss.to(device))
        if mode == "train":
          loss.backward(retain_graph=True)
          optimizer.step()

        each_session_loss.append(loss.item())
        out_clss = torch.max(output.data, 1)[1]
        for out_cls, cls, out_score in zip(out_clss, clss, output.data):
          pred_clss.append(int(out_cls.data))
          actual_clss.append(int(cls))
          pred_scores.append(out_score.cpu().numpy())

      # Calculate accuracy for each session.
      acc = get_accuracy_of_predicted_classes(pred_clss, actual_clss)
      log.INFO("Epoch: %s, session: %s, %s accuracy: %s, %s loss: %s" % (
               epoch, session_id, mode, acc, mode, np.mean(each_session_loss)))
      all_session_acc.append(acc)
      all_session_loss.append(np.mean(each_session_loss))
      if epoch % 10 == 0:
        epoch_results[session_id] = (
            pred_clss, actual_clss, np.array(pred_scores))
    return all_session_loss, all_session_acc, epoch_results

  for epoch in range(1, num_epochs+1):
    all_session_loss, all_session_acc, _ = _execute_model(
        epoch, "train", HDD_TRAIN_SIDS, layer, batch_size, seq_size,
        map_lbl_to_cls, model=model)
    log.INFO("Epoch: %s, Average Training Loss: %s, Average Training Accuracy: %s"
             % (epoch, np.mean(all_session_loss), np.mean(all_session_acc)))
    if epoch % 10 == 0:
      all_session_loss, all_session_acc, epoch_results = _execute_model(
          epoch, "test", HDD_TEST_SIDS, layer, batch_size, seq_size,
          map_lbl_to_cls, model=model)
      log.INFO("Epoch: %s, Average Test Loss: %s, Average Test Accuracy: %s"
               % (epoch, np.mean(all_session_loss), np.mean(all_session_acc)))
      log.INFO("Saving pred classes, actual classes, and pred scores in dir: %s"
               % exp_out_dir)
      pickle.dump(epoch_results,
                  open(exp_out_dir + "/epoch_%s_results.p" % epoch, "wb"))

if __name__ == "__main__":
  # Configure the logger.
  model = "GoogLeNet"
  if model == "ResNet18":
    exp_out_dir = HDD_EXP_2DCNN_LSTM_OUT_VIDS_RES18
  elif model == "GoogLeNet":
    exp_out_dir = HDD_EXP_2DCNN_LSTM_OUT_VIDS_GGLNT

  exp_out_dir = exp_out_dir + "/weight_invbalanced_focal_loss_gamma_2_0/"
  log.configure_log_handler(
    "%s_%s.log" % (exp_out_dir + __file__, datetime.datetime.now()))
  log.INFO("Inverse weight balanced with Focal Loss. Gamma=2.0")
  train_test_session_wise_2D_CNN_plus_LSTM(
      HDD_TRAIN_SIDS, HDD_TEST_SIDS, 1024, exp_out_dir, weight_bal="inv_bal",
      use_fl=True, gamma=2.0, batch_size=1, seq_size=3600, model=model)
