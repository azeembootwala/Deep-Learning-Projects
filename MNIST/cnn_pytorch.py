import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from util_pyt import getImageData

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 16, kernel_size= 5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels = 16 , out_channels=32, kernel_size = 5, stride = 1, padding=2)
        self.relu = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(32*7*7, 10)

    def forward(self, X):
        out = self.conv1(X)
        out = self.relu(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

def train(X, Y, model , loss , optimizer):
    if torch.cuda.is_available():
        inputs = Variable(X.cuda())
        labels = Variable(Y.cuda())
    else:
        inputs = Variable(X)
        labels = Variable(Y)
    optimizer.zero_grad()
    output = model(inputs)
    cost = loss(output,labels)
    cost.backward()
    optimizer.step()
    return cost

def predict(X,Y, model,loss):
    if torch.cuda.is_available():
        X = Variable(X.cuda())
        #Y = Variable(Y.cuda())
    else:
        X = Variable(X)
        #Y = Variable(Y)
    output=model(X)
    #cost = loss(output, Y)
    #pred = output.data.cpu().numpy().argmax(axis=1)
    _,pred = torch.max(output.data, 1)
    return pred

def main():
    X , Y = getImageData()
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    Xtrain = torch.from_numpy(Xtrain).float()
    Ytrain = torch.from_numpy(Ytrain).long()
    Xtest = torch.from_numpy(Xtest).float()
    Ytest = torch.from_numpy(Ytest).long()
    model = CNN()
    if torch.cuda.is_available():
        model.cuda()
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    N = Xtrain.shape[0]
    batch_size = 100
    n_batches =N//batch_size
    max_iter = 30
    for i in range(0,max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
            Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

            c =train(Xbatch, Ybatch , model , loss , optimizer)

            if j %10 ==0:
                 pred = predict(Xtest , Ytest , model , loss)
                 result = np.mean(pred.cpu().numpy() == Ytest.numpy())
                 print(" cost is %f , classification rate is %f " % (c.data[0],result))


if __name__ =="__main__":
    main()
