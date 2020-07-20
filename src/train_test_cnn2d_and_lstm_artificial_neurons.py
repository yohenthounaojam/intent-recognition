#
# This file uses a 2D CNN's extracted features + LSTM for intent recognition
# on HDD dataset. Artificial Neurons are employed.
#
# Author: Ramashish Gaurav
#
import datetime
import numpy as np
import os
import random
import torch.nn as nn
import torch

from utils import log
from utils.cnn_lstm_utils import LSTMVR
from utils.consts import HDD_FVECS_PATH, LOGS_PATH
from utils.data_prep_util import get_train_test_tensors_and_labels_lst
from utils.exp_util import get_layer_train_test_indices_cls_map_dict

def train_test_2D_CNN_plus_LSTM(ffvecs_data, layer, div_frc=0.7, lrng_rate=0.0005,
                                num_layers=2, dropout=0.2, num_epochs=10):
  """
  Trains the features extracted from a 2D CNN over the LSTM.

  Args:
    ffvecs_data (dict): {int: (torch.Tensor, (int, int))} where key is index of
        row in "event_pd" dataframe and torch.Tensor is of shape:
        (1 x seq_len x dimension of extracted features)
    layer (int): Layer for which experiment is to be done.
    div_frc (float): The percentage of training data, rest will be test data.
    lrng_rate (float): The learning rate of the LSTM.
    num_layers (int): Number of layers in the LSTM.
    num_epochs (int): Number of epochs over whole training data.
  """
  log.INFO("Curating the HDD data...")
  # Curate the data.
  train_idcs_dict, test_idcs_dict, cls_to_et_map = (
      get_layer_train_test_indices_cls_map_dict(layer, div_frc))
  train_data, train_labels, test_data, test_labels = (
      get_train_test_tensors_and_labels_lst(
      train_idcs_dict, test_idcs_dict, ffvecs_data, cls_to_et_map))

  input_dim = train_data[0].shape[2]
  hidden_dim = 2 * input_dim
  num_clss = len(np.unique(train_labels))
  train_indices = list(range(len(train_labels)))
  test_indices = list(range(len(test_labels)))
  log.INFO("Layer: %s, Fraction of training data: %s, Learning Rate: %s, Number"
           " of layers in LSTM: %s, Dropout: %s, Number of Epochs: %s, Input "
           "features dimension: %s, LSTM hidden state dimension: %s, Number of "
           "training classes: %s" % (layer, div_frc, lrng_rate, num_layers,
           dropout, num_epochs, input_dim, hidden_dim, num_clss))

  log.INFO("Configuring the LSTM...")
  # Configure the LSTM training module.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  lstm_model = LSTMVR(input_dim, hidden_dim, num_layers, num_clss, dropout)
  lstm_model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lrng_rate)
  log.INFO("LSTM Model: %s" % lstm_model)
  log.INFO("Loss function: %s" % criterion)
  log.INFO("BP Optimizer: %s" % optimizer)

  # Start the training process.
  lstm_model.train()
  log.INFO("Starting Training...")
  for epoch in range(num_epochs):
    # Randomly sample the training indices.
    train_indices = random.sample(train_indices, len(train_indices))
    iteration = 0

    for idx in train_indices:
      # Manually set the gradients to zero to mark a new beginning of gradient
      # computation over a new sample. Else, previously computed gradients will
      # affect the computation of new gradients (as the autograd engine remembers
      # all the operations with a variable to compute the gradient, but doesn't
      # know at which step of the operations to start a fresh gradient computation.
      optimizer.zero_grad()
      output = lstm_model(train_data[idx].to(device).requires_grad_())
      print(output.shape, torch.Tensor([train_labels[idx]]).shape)
      loss = criterion(output, torch.Tensor([train_labels[idx]]).long().to(
          device))
      if iteration % 1000 == 0:
        log.INFO("Epoch: %s, Training Loss: %s" % (epoch, loss.item()))
      loss.backward()
      optimizer.step()
      iteration += 1

    log.INFO("#"*30 + "Epoch: %s Done!" % epoch + "#"*30)

  # Start the testing process.
  lstm_model.eval()
  for param in lstm_model.parameters():
    param.requires_grad = False

  pred_labels = []
  acc = 0

  for idx in test_indices:
    output = lstm_model(test_data[idx].to(device))
    out_label = torch.max(output.data, 1)[1]
    if out_label == test_labels[idx]:
      acc += 1
    pred_labels.append(out_label)

  # accuracy.
  log.INFO("Accuracy: %s" % str(acc/len(test_labels)))
  return test_labels, pred_labels

if __name__ == "__main__":
  # Configure the logger.
  log.configure_log_handler(
      "%s_%s.log" %  (LOGS_PATH + __file__, datetime.datetime.now()))
  ffvecs_data = np.load(
      HDD_FVECS_PATH + "hdd_frame_vecs_dict_fps_3_1_model_ResNet.npy",
      allow_pickle=True)
  test_labels, pred_labels = train_test_2D_CNN_plus_LSTM(
      ffvecs_data, 0, num_epochs=2)
