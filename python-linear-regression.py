import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('~/pytorch-fundamentals/insurance.csv')


features = data.drop('charges', axis = 1)
target = data['charges']

features.head()
target.head()

inputs = np.array(pd.get_dummies(features))
targets = np.array(target)

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

inputs.shape, target.shape

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2)

print(X_train.shape, y_train.shape)
torch.seed()
w = torch.randn(11, requires_grad=True)
b = torch.randn(1070, requires_grad=True)
print(w.t().shape)
def model(x):
    return x @ w.t() + b

preds = model(X_train.float())


y_train = torch.Tensor(y_train)

def mse(preds, true):
    diff = preds - true
    return torch.sum(diff ** 2) / diff.numel()

alpha = 1e-5
for epoch in range(100):
    preds = model(X_train.float())
    loss = mse(preds, y_train)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * alpha
        b -= b.grad * alpha
        w.grad.zero_()
        b.grad.zero_()

preds = model(X_train.float())
loss = mse(preds, y_train)
print(loss)
