#
# This file implements the util functions for the CNNs and LSTMS.
#
# Author: Ramashish Gaurav
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from . import log

def get_cnn_codes_for_frames_tt(model, frames, llyr=-1):
  """
  Returns CNN feature vectors for a stack of frames. Note that this code works
  perfectly for pytorch pretrained ResNet18 and GoogleNet models. However for
  InceptionV3 model it breaks. Looks like there are some architectural
  limitations in InceptionV3 model with HDD data. TODO: Investigate?

  Args:
    model (torchvision.models): Pretrained torchvision model.
    frames (numpy.ndarray): An array 3D frames. Shape: 4D - (num_frames x 3D).
    llyr (int): Last layer index (in negative indexing) from which features
                are required.

  Returns:
    torch.Tensor: 4D Shape-> num_frames x 3D feats, 3D feats can be: 1024x1x1
  """
  to_tensor = transforms.ToTensor()
  # to_tensor also takes care of coverting all the pixel values to range [0, 1].
  # Also, Normalize the images with specific mean and std.
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  frames_tensor = torch.stack([normalize(to_tensor(frame)) for frame in frames])
  # Image dim 299 for inception_v3 model and 224 for rest of the models.
  assert frames_tensor.shape[2] >= 299
  assert frames_tensor.shape[3] >= 299

  model.eval() # Batchnorm and Dropout layers work in eval mode rather training.
  # Freeze the parameters as only features are extracted (saves GPU memory too).
  for param in model.parameters():
    param.requires_grad = False

  # Map the model and data to use GPU if available.
  if torch.cuda.is_available():
    model = model.cuda()
    frames_tensor = frames_tensor.cuda()

  feature_extractor = torch.nn.Sequential(*list(model.children())[:llyr])
  output_features = feature_extractor(frames_tensor)
  del frames_tensor
  torch.cuda.empty_cache()

  return output_features.to("cpu")

class LSTMVR(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, num_clss, dropout=0,
               batch_dim=1):
    """
    Instantiates the LSTMVR class object. TODO: Stateful and Stateless.

    Args:
      input_dim (int): The frame's feature vector dim, (same for all frames).
      hidden_dim (int): The dim of the hidden state in LSTM cells.
      num_layers (int): The number of layers in the LSTM stack.
      num_clss (int): Number of video classes.
      dropout (int): The dropout between two layers of stacked LSTM, defaul 0.
      batch_dim (int): Number of input sequences in one batch, default 1.
    """
    super().__init__()
    self._input_dim = input_dim
    self._hidden_dim = hidden_dim
    self._num_layers = num_layers
    self._dropout = dropout
    self._num_clss = num_clss
    self._batch_dim = batch_dim

    # Define the LSTM layer.
    self._lstm = nn.LSTM(
        input_size=self._input_dim, hidden_size=self._hidden_dim,
        num_layers=self._num_layers, batch_first=True, dropout=self._dropout)
    # Define the dropout layer between LSTM and FC.
    self._dptlyr = nn.Dropout(self._dropout)
    # Define the linear FC layer for class recognition. Size of input vector
    # to linear FC layer is same as the dim of the output of LSTM cell i.e. the
    # `hidden_dim`.
    self._lfc = nn.Linear(self._hidden_dim, self._num_clss)

  def forward(self, x):
    """
    Forward pass of the LSTM training process. TODO: Include a dropout between
    LSTM and FC with sigmoid on the output of FC layer.
    Note that only the input `x` is passed to the LSTM object and no initial
    hidden state or the cell state. When no initial states are passed, the LSTM
    object assumes zero states for the initial hidden state and cell state.

    Args:
      x (torch.Tensor): A 3D tensor of shape -> batch_dim x seq_len x input dim
    """
    lstm_out, (hidden_state, cell_state) = self._lstm(x)
    # lsmt_out from self._lstm() has shape: batch_dim x seq_dim x hidden_dim.
    lstm_out = self._dptlyr(lstm_out)
    # For a `batch_dim` of 1, the LSTM output has `seq_dim` length outputs, each
    # of size `hidden_dim`. For video recognition, we need only the last output
    # in `seq_dim` sized outputs for the classification purpose.
    # TODO: May add a ReLU layer here.
    lstm_out = self._lfc(lstm_out[:, -1, :])
    # Shape of lstm_out from self._lfc() will be: batch_dim x num_clss
    return lstm_out

class LSTMIR(nn.Module):
  """
  LSTM model for intent recognition.
  """
  def __init__(self, input_dim, hidden_dim, num_layers, num_clss, dropout=0,
               batch_dim=1):
    """
    Instantiates the LSTMIR class object. TODO: Stateful and Stateless.

    Args:
      input_dim (int): The frame's feature vector dim, (same for all frames).
      hidden_dim (int): The dim of the hidden state in LSTM cells.
      num_layers (int): The number of layers in the LSTM stack.
      num_clss (int): Number of intent (or event_type) classes.
      dropout (int): The dropout between two layers of stacked LSTM, defaul 0.
      batch_dim (int): Batch size, default 1.
    """
    super().__init__()
    self._input_dim = input_dim
    self._hidden_dim = hidden_dim
    self._num_layers = num_layers
    self._dropout = dropout
    self._num_clss = num_clss
    self._batch_dim = batch_dim

    # Define the LSTM layer.
    self._lstm = nn.LSTM(
        input_size=self._input_dim, hidden_size=self._hidden_dim,
        num_layers=self._num_layers, batch_first=True, dropout=self._dropout)
    # Define the dropout layer between LSTM and FC.
    self._dptlyr = nn.Dropout(self._dropout)
    # Define the linear FC layer for class recognition. Size of input vector
    # to linear FC layer is same as the dim of the output of LSTM cell i.e. the
    # `hidden_dim`.
    self._lfc = nn.Linear(self._hidden_dim, self._num_clss)

  def forward(self, x, hidden_state):
    """
    Forward pass of the LSTM training process. TODO: Include a dropout between
    LSTM and FC with sigmoid on the output of FC layer.

    Args:
      x (torch.Tensor): The input data of shape (Batch x Sequence x Feature Dim).
      hidden_and_cell_state ((torch.Tensor, torch.Tensor)): A tuple of initial
           hidden state and cell state.
    """
    #TODO: Do I need to detach the hidden_state and cell state?
    lstm_out, hidden_state = self._lstm(x, hidden_state)
    # lsmt_out from self._lstm() has shape: batch_dim x seq_dim x hidden_dim.
    # hidden_state[0] is of shape: num_layers x batch_size x hidden_dim
    # hidden_state[1] is of shape: num_layers x batch_size x hidden_dim
    lstm_out = self._dptlyr(lstm_out)
    # For a `batch_dim` of 40, the LSTM output has 40 x `seq_dim` length outputs,
    # each of size `hidden_dim`. For intent recognition we need all `seq_dim`
    # sized outputs.
    lstm_out = self._lfc(lstm_out)
    # Shape of lstm_out from self._lfc() will be: batch_dim x seq_dim x num_clss.
    return lstm_out, hidden_state

class FocalLoss(nn.Module):
  def __init__(self, weights, gamma=2.0, reduction="mean"):
    """
    Args:
      weights (pytorch.Tensor): A tensor of floats, denoting the weight of each
                                class.
      gamma (float): The gamma hyperparameter. If `gamma` == 0, FocalLoss becomes
                     CrossEntropyLoss.
      reduction (string): One of "sum"|"mean"|"none"
    """
    super().__init__()
    log.INFO("FocalLoss Gamma: %s:" % gamma)
    self._weights = weights
    self._gamma = gamma
    self._reduction = reduction

  def forward(self, inputs, targets):
    """
    Compute the loss.

    Args:
      inputs (torch.Tensor): A 2D float Tensor of shape number of instances x
                             number of classes.
      targets (torch.Tensor): A 1D long Tensor of shape number of instances,
                              where each element is the class of the instance.

    Returns:
      torch.Tensor: The computed loss.
    """
    log_probability = F.log_softmax(inputs, dim=-1)
    probability = torch.exp(log_probability)
    loss = F.nll_loss((1-probability)**self._gamma * log_probability,
                      targets, weight=self._weights, reduction=self._reduction)
    return loss
