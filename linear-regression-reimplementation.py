import torch.nn
from torch import nn
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

data = pd.read_csv('~/pytorch-fundamentals/insurance.csv')


features = data.drop('charges', axis = 1)
target = data['charges']


inputs = np.array(pd.get_dummies(features))
targets = np.array(target)

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size = 0.2)

print(inputs.shape, targets.shape)
train_ds = TensorDataset(inputs, targets)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(11,1)
print(model.weight.shape)
print(model.bias.shape)
print(list(model.parameters()))

preds = model(inputs.float())
print(preds)

loss_fn = F.mse_loss
loss = loss_fn(model(inputs.float()), targets)
print(loss)

opt = torch.optim.SGD(model.parameters(), lr=0.000001)

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            loss.backward()

            opt.step()
            opt.zero_grad()
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {}'.format(epoch+1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt)

loss = loss_fn(model(inputs.float()), targets)
print(loss)
