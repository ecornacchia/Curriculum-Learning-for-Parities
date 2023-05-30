import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def generate_biased_parity(n, d, k, p):
    # p: prob. x_i=1
    bias = p * torch.ones(n, d, device = device)
    x = 2 * torch.bernoulli(bias) - 1
    y = torch.prod(x[:,:k], axis=1) 
    return x, y
    
## network
class twolayer_fcn(nn.Module):
    def __init__(self, h, d):  ## h: nb hidden neurons
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, h)/ d**0.5)
        self.b = nn.Parameter(torch.randn(h)/ d**0.5)
        self.beta = nn.Parameter(torch.randn(h)/h**0.5)   ##weigths of 2nd layer
        
    def forward(self, x):
        z = (x.squeeze() @ self.w + self.b) 
        y = F.relu(z) @ self.beta / self.beta.size(0)**0.5
        return y    
    
class threelayer_fcn(nn.Module):
    def __init__(self, h, d):  ## h: nb hidden neurons
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, h)/ d**0.5)
        self.v = nn.Parameter(torch.randn(h, h)/ h**0.5)
        self.b = nn.Parameter(torch.randn(h)/ d**0.5)
        self.beta = nn.Parameter(torch.randn(h)/h**0.5)   ##weigths of 2nd layer
        
    def forward(self, x):
        z = (x.squeeze() @ self.w + self.b) 
        y = F.relu(z) @ self.v / h**0.5
        p = F.relu(y) @ self.beta / h**0.5
        return p        
        
def build_model(arch, h,d):
    if arch == '2l':
        model = twolayer_fcn(h,d)   
    if arch == '3l':
        model = threelayer_fcn(h,d)
    return model.to(device)
    
## Train and test

def train_loop(dataloader, model, loss_fn, optimizer):   #dataloader=dl_train
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  #puts all gradient to zero
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    return loss


def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(torch.sign(pred)/2, y/2).item()
            
    test_loss /= num_batches
    return test_loss

### 2-steps CL
def test_error(generate_data,k,p,model,loss_fn,d,h):
    global n
    xtest,ytest = generate_data(n, d, k, p)
    dl_test = DataLoader(TensorDataset(xtest,ytest),batch_size=n)
    test = test_loop(dl_test,model,loss_fn)
    return test

def run_2steps_parity_p(k, model,epochs,d,h,p,b):
    global n
    ntest = n
    loss_fn = nn.MSELoss()
    learning_rate = 0.1
    T1 = int(epochs/2)
    
    trainvec,test_half,test_p = list(),list(),list()

    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    for t1 in tqdm(range(T1)):
        xtrain, ytrain = generate_biased_parity(n, d, k, p)
        dl_train = DataLoader(TensorDataset(xtrain,ytrain),batch_size=b)
        train = train_loop(dl_train,model,loss_fn,optimizer)
        trainvec.append(train)

        test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)
        test_half.append(test)
        testp_val = test_error(generate_biased_parity,k,p,model,loss_fn,d,h)
        test_p.append(testp_val)
        if testp_val <= 0.01:
            break
            

    for t2 in tqdm(range(epochs-T1)):       
        xtrain, ytrain = generate_biased_parity(n, d, k, 1/2)
        dl_train = DataLoader(TensorDataset(xtrain,ytrain),batch_size=b)
        train = train_loop(dl_train,model,loss_fn,optimizer)
        trainvec.append(train)    
        
        test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)
        test_half.append(test)
        testp_val = test_error(generate_biased_parity,k,p,model,loss_fn,d,h)
        test_p.append(testp_val)
        if test <= 0.01:
            break
 
    test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)        
    return model,trainvec,test_half,test_p, test, t1,t2

### Continuous CL

def test_error(generate_data,k,p,model,loss_fn,d,h):
    global n
    xtest,ytest = generate_data(n, d, k, p)
    dl_test = DataLoader(TensorDataset(xtest,ytest),batch_size=n)
    test = test_loop(dl_test,model,loss_fn)
    return test

def run_continuous_parity(k, model,epochs,d,h,b):
    global n
    ntest = n
    loss_fn = nn.MSELoss()
    learning_rate = 0.1
    pvec = np.linspace(0,1/2,epochs-10)
    
    trainvec,test_half,test_p = list(),list(),list()

    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
    
    for t in tqdm(range(epochs-10)):
        p = pvec[t]
        xtrain, ytrain = generate_biased_parity(n, d, k, p)
        dl_train = DataLoader(TensorDataset(xtrain,ytrain),batch_size=b)
        train = train_loop(dl_train,model,loss_fn,optimizer)
        trainvec.append(train)
  
        # if t1 %50 ==0:
        test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)
        test_half.append(test)
        testp_val = test_error(generate_biased_parity,k,p,model,loss_fn,d,h)
        test_p.append(testp_val)
        # if testp_val <= 0.01:
        #     break

    for t in tqdm(range(10)):
        p=1/2
        xtrain, ytrain = generate_biased_parity(n, d, k, p)
        dl_train = DataLoader(TensorDataset(xtrain,ytrain),batch_size=b)
        train = train_loop(dl_train,model,loss_fn,optimizer)
        trainvec.append(train)
  
        # if t1 %50 ==0:
        test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)
        test_half.append(test)
        testp_val = test_error(generate_biased_parity,k,p,model,loss_fn,d,h)
        test_p.append(testp_val)
    
    test = test_error(generate_biased_parity,k, 1/2,model,loss_fn,d,h)        
    return model,trainvec,test_half,test_p, test, t
    
