# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import pandas as pd
import numpy as np

WEIGHT_DECAY = 1e-5
EPOCHS = 10
LEARNING_RATE = 1e-3
SPARSE_REG = 1e-3

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Sparse autoencoder for feature representation
# Uses a lot of hidden nodes for minimal activation (sharp features)
class SparseAutoEncoder(nn.Module):
  def __init__(self, feature_size, hidden_size):
    super(SparseAutoEncoder, self).__init__()
    self.feature_size = feature_size
    self.hidden_size = hidden_size

    # Encoder layers
    self.layer1 = nn.Linear(feature_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, feature_size)

  # Feedforward
  def forward(self, x):
    x = F.sigmoid(self.layer1(x))
    x = F.sigmoid(self.layer2(x))
    return x
  
# Difference in going from p to pHat tensors
def kl_divergence(p, pHat):
  pHat = torch.mean(torch.sigmoid(pHat))
  p = torch.tensor(p)
  res = torch.sum(p * torch.log(p) - p * torch.log(pHat) + (1 - p) * torch.log(1 - p) - (1 - p) * torch.log(1 - pHat))
  return res

def custom_loss(xHat, x, W, V, b1, b2):
  m = len(x)
  loss = (1 / 2 * m) * (torch.sum((x - xHat)** 2)) + ((WEIGHT_DECAY / 2) * (
    torch.sum(W** 2) + torch.sum(V** 2) + torch.sum(b1** 2) + torch.sum(b2** 2))
    + SPARSE_REG*(kl_divergence(0.3, x))
  )
  return loss

# Train
def train_encoder(save=True):
  df = pd.read_hdf('./data/clean_num_cols.h5')
  targets = pd.read_hdf('./data/clean_target_vars.h5')

  train = torch.utils.data.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(targets)))
  train_loader = torch.utils.data.DataLoader(train, batch_size=64)
  
  # Training autoencoder
  net = SparseAutoEncoder(feature_size=44, hidden_size=22)  
  if torch.cuda.is_available():
    net = net.cuda()
  
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  net.train()

  for epoch in range(EPOCHS):
    for batch_idx, (x,y) in enumerate(train_loader):
      x = Variable(x[0])
      optimizer.zero_grad()
      outputs = net.forward(x)
      mse_loss = criterion(outputs, x)
      kl_loss = custom_loss(outputs, x, net.layer1.weight.data,
       net.layer2.weight.data, net.layer1.bias.data, net.layer2.bias.data)
      loss = kl_loss
      loss.backward()
      optimizer.step()
      if batch_idx % 10000 == 0:
        print('Epoch [{}/{}] - Iter[{}/{}] \t KL Divergence: {}'.format(
          epoch+1, EPOCHS, batch_idx, len(train_loader.dataset)//64, loss.item()))
  
  # Remove decoder and add classifier
  new_classifier = nn.Sequential(*list(net.children())[:-1])
  net = new_classifier
  net.add_module('classifier', nn.Sequential(
    nn.Linear(22, 15),
    nn.Softmax(1)
  ))

  print(net)

  criterion = nn.CrossEntropyLoss()

  for epoch in range(EPOCHS):
    running_corr = 0.0
    for i, data in enumerate(train_loader):
      x, y = data
      x = Variable(x)
      y = Variable(y)
      net.zero_grad()
      outputs = net.forward(x)
      loss = criterion(outputs, y.long())
      loss.backward()
      optimizer.step()

      # Eval
      preds = torch.argmax(outputs, 1)
      running_corr += torch.sum(preds.long() == y.long()).item()
      corr_percentage = running_corr / ((i+1)*(64))
      if i % 10000 == 0:
        print("Epoch: [{}/{}] - Iter[{}/{}] - Epoch loss: {} - Epoch accuracy: {}".format(epoch+1, EPOCHS, i+1, len(train_loader.dataset)//64, loss.item(), corr_percentage))
  if save == True:
      torch.save(net.state_dict(), './saved_models/Autoencoder.pt')

if __name__ == "__main__":
  train_encoder(save=True)
  pass