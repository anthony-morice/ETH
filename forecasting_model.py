import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
AHEAD = 1 
LAGS = 120
FEATURES = 5
HIDDEN_SIZE = 16
N_LAYERS = 3
N_HIDDEN = 32

DATA_ROOT = "../data/ETH"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

class LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_size = FEATURES
    self.hidden_size = HIDDEN_SIZE
    self.num_layers = N_LAYERS
    self.LSTM1= nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
    self.n_hidden = N_HIDDEN
    self.fc1 = nn.Linear(self.hidden_size, self.n_hidden)
    self.relu = nn.ReLU()
    self.fc2 = nn. Linear(self.n_hidden, 1)

  def forward(self, x):
    # x is of shape (BATCH_SIZE, LAGS, FEATURES)
    output, _ = self.LSTM1(x)
    output = self.fc1(output[:,-1,:])
    output = self.relu(output)
    return self.fc2(output)

class EthDataset(Dataset):
  def __init__(self, filename, n_lags=60, n_ahead=1, train=True, mean=None, std=None):
    super().__init__()
    self.lags = n_lags
    self.parse_file(filename, n_ahead)
    if train:
      self.fit()
    else:
      assert mean is not None and std is not None, "Must supply mean and std for non training data!"
      self.mean, self.label_mean = mean
      self.std, self.label_std = std
    self.normalize()

  def parse_file(self, file, n_ahead):
    # read file
    fh = open(file, "r")
    self.header = fh.readline().strip().split(',')
    raw_data = fh.readlines()
    fh.close()
    # parse data and generate labels
    for i,ln in enumerate(raw_data):
      raw_data[i] = list(map(lambda x: float(x), ln.strip().split(',')))
    raw_data.sort(key=lambda x: x[0]) # sort on timestamp
    data_labels = []
    for t in range(len(raw_data) - n_ahead):
      #data_labels.append(raw_data[t + n_ahead][4] - raw_data[t][4]) 
      data_labels.append(raw_data[t + n_ahead][4]) 
    self.data = torch.tensor(np.array(raw_data[:-n_ahead])[:, 1:], dtype=torch.float)
    self.labels = torch.tensor(data_labels, dtype=torch.float)[:, None]

  def fit(self):
    self.mean = torch.mean(self.data, axis=0)
    self.std = torch.std(self.data, axis=0)
    self.label_mean = torch.mean(self.labels)
    self.label_std = torch.std(self.labels)

  def normalize(self):
    self.data = (self.data - self.mean) / (self.std + 1e-15)
    self.labels = (self.labels - self.label_mean) / (self.label_std + 1e-15)

  def denormalize(self, label):
    return label * (self.label_std + 1e-15) + self.label_mean

  def __len__(self):
    return self.data.shape[0] - self.lags + 1

  def __getitem__(self, idx):
    return self.data[idx:idx+self.lags], self.labels[idx+self.lags - 1]

def load_data():
  training_data = EthDataset(f"{DATA_ROOT}/eth_training.csv", n_lags=LAGS, n_ahead=AHEAD)
  #training_data = EthDataset(f"{DATA_ROOT}/small_dataset.csv", n_lags=LAGS, n_ahead=AHEAD)
  training_size = int(0.8 * len(training_data))
  validation_size = len(training_data) - training_size
  training_data, validation_data = torch.utils.data.random_split(training_data, 
      [training_size, validation_size])
  #test_data = EthDataset(f"{DATA_ROOT}/eth_test.csv",
  test_data = EthDataset(f"{DATA_ROOT}/small_dataset.csv",
                         n_lags=LAGS, 
                         n_ahead=AHEAD,
                         train=False,
                         mean=(training_data.dataset.mean, training_data.dataset.label_mean),
                         std=(training_data.dataset.std, training_data.dataset.label_std))
  return training_data, validation_data, test_data

def train_model(dataloader, model, loss_fn, optimizer):
  model.train()
  #with autograd.detect_anomaly():
  for batch, (samples, labels) in enumerate(dataloader):
    samples, labels = samples.to(DEVICE), labels.to(DEVICE)
    # forward pass
    y_tilde = model(samples)
    loss = loss_fn(y_tilde, labels)
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print stats
    if batch % 100 == 0:
      sys.stdout.write("\x1b[0G" + f"batch {batch} of {len(dataloader)} -->  loss: {loss.item():>7f}")
      sys.stdout.flush()

def evaluate_model(dataloader, model, loss_fn, graph=False):
  denormalize = None
  if isinstance(dataloader.dataset, Subset):
    denormalize = dataloader.dataset.dataset.denormalize
  else:
    denormalize = dataloader.dataset.denormalize
  abs_error, loss = 0, 0
  preds = []
  lbls = []
  with torch.no_grad():
    for samples, labels in dataloader:
      samples, labels = samples.to(DEVICE), labels.to(DEVICE)
      y_tilde = model(samples)
      loss += loss_fn(y_tilde, labels).item() 
      d_labels = denormalize(labels)
      d_y_tilde = denormalize(y_tilde)
      abs_error += (d_labels - d_y_tilde).abs().mean().item()
      if graph:
        preds += list(d_y_tilde.cpu().flatten())
        lbls += list(d_labels.cpu().flatten())
  abs_error /= len(dataloader)
  loss /= len(dataloader)
  print(f"\nEvaluation Results: \n  Avg loss -> {loss:>.5f}, Mean Abs Error -> {abs_error:>.5f}\n")
  if graph:
    plt.plot(preds, label="Predicted")
    plt.plot(lbls, label="Actual")
    plt.legend()
    plt.show()
  return abs_error

if __name__ == "__main__":
  load = True
  if len(sys.argv) < 2:
    load = False
  print("Loading Data...")
  training_data, validation_data, test_data = load_data()
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
  print("...DONE\n\n")
  model = LSTM().to(DEVICE)
  loss_fn = nn.MSELoss()
  print(model)
  if load:
    model.load_state_dict(torch.load(sys.argv[1]))
  else:
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mabs, count = 0, 0
    for t in range(EPOCHS):
      print(f"************ Epoch {t} ************\n")
      train_model(train_dataloader, model, loss_fn, optimizer)
      mabs = evaluate_model(validation_dataloader, model, loss_fn)
      count = count + 1 if mabs >= best_mabs else 0
      best_mabs = min(mabs, best_mabs)
      """
      if (count > 5):
        break
      """
  mabs = evaluate_model(test_dataloader, model, loss_fn, True)
  print(f"\nTest set mean abs error of trained model: {mabs}")
  torch.save(model.state_dict(), f"model_{int(time.time())}.pth")
  print("DONE")
