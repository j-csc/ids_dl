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

    # Encoder-Decoder layers
    self.layer1 = nn.Linear(feature_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, feature_size)

  # Feedforward
  def forward(self, x):
    x = F.sigmoid(self.layer1(x))
    x = F.sigmoid(self.layer2(x))
    return x
  
# Difference in going from p to pHat tensors
def kl_divergence(p, pHat):
  funcs = nn.Sigmoid()
  pHat = torch.mean(funcs(pHat), 1)
  p_tensor = torch.Tensor([p] * len(pHat)).to(device)
  return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(pHat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - pHat))

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
  train = torch.utils.data.TensorDataset(torch.Tensor(np.array(df)))
  train_loader = torch.utils.data.DataLoader(train, batch_size=64)
  
  net = SparseAutoEncoder(feature_size=44, hidden_size=22)
  print(net)
  
  if torch.cuda.is_available():
    net = net.cuda()
  
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  net.train()

  for epoch in range(EPOCHS):
    for batch_idx, x in enumerate(train_loader):
      x = Variable(x[0])
      optimizer.zero_grad()
      outputs = net(x)
      mse_loss = criterion(outputs, x)
      kl_loss = custom_loss(outputs, x, net.layer1.weight.data,
       net.layer2.weight.data, net.layer1.bias.data, net.layer2.bias.data)
      loss = mse_loss + kl_loss * 1e-3
      loss.backward()
      optimizer.step()
      if batch_idx % 1000 == 0:
        print('Epoch [{}/{}] - Iter[{}/{}] \t Total Loss: {}'.format(
          epoch+1, EPOCHS, batch_idx, len(train_loader.dataset)//64, loss))
  if save == True:
    torch.save(net.state_dict(), './savedModels/Autoencoder.pt')

if __name__ == "__main__":
  train_encoder(save=True)
  pass