import pandas
import numpy as np
from utilities_AB import get_data
import matplotlib.pyplot as plt


def tar2ind(Y):
    N = len(Y)
    D = len(set(Y))
    ind = np.zeros((N,D))
    for i in range(0,len(Y)):
        ind[i,int(Y[i])]=1
    return ind
def sigmoid(Z):
    Z = np.exp(-Z)
    return 1/(1+Z)
def relu(Z):
    return Z*(Z>0)
def softmax(Z):
    Z = np.exp(Z)
    return Z /Z.sum(axis=1,keepdims=True)
def forward(X,W1,B1,W2,B2):
    Z = np.tanh(X.dot(W1)+B1)
    out = softmax(Z.dot(W2)+B2)
    return out,Z
def cross_entropy(Y,T):
    return -(T*np.log(Y)).sum()
def derivative_W2(T,Y,Z,reg,W2):
    return Z.T.dot(T-Y)+reg*W2
def derivative_b2(T,Y,reg,b2):
    return (T-Y).sum(axis = 0)+reg*b2
def derivative_W1(T,Y,Z,W2,X,reg,W1):
    dz = (T-Y).dot(W2.T)*(1-Z*Z)
    #dz = (T-Y).dot(W2.T) * (Z > 0)
    out = X.T.dot(dz)+reg*W1
    return out
def derivative_b1(T,Y,W2,Z,reg,b1):
    return ((T-Y).dot(W2.T)*(1-Z*Z)).sum(axis=0) + reg*b1
    #return ((T-Y).dot(W2.T)*(Z>0)).sum(axis=0) + reg*b1

def classification_rate(Y,T):
    return np.mean(Y==T)

def main():
    X,Y = get_data() # Loading the data
    print("Data loaded")
    X_train = X[:-1000,:] #Dividing the data in train and test
    Y_train = Y[:-1000]
    X_test = X[-1000:,:]
    Y_test = Y[-1000:]
    t_mat = tar2ind(Y_train)

    N,D = X_train.shape
    M = 200 # Number of hidden units
    K = len(set(Y_train))

    W1 = np.random.randn(D,M)/np.sqrt(D)
    b1 = np.random.randn(M)
    W2=np.random.randn(M,K)/np.sqrt(M)
    b2 = np.random.randn(K)
    lr = 10e-7
    reg = 10e-7  
    cost = []

    for i in range(0,10000):
        output,Z = forward(X_train,W1,b1,W2,b2)
        W2 = W2+lr*derivative_W2(t_mat,output,Z,reg,W2)
        b2 = b2+lr*derivative_b2(t_mat,output,reg,b2)
        W1 = W1 + lr*derivative_W1(t_mat,output,Z,W2,X_train,reg,W1)
        b1 = b1 + lr*derivative_b1(t_mat,output,W2,Z,reg,b1)
        if i%10==0:
            c =cross_entropy(output,t_mat)
            cost.append(c)
            y_pred = np.argmax(output,axis=1)
            r = classification_rate(y_pred,Y_train)
            print("i: ",i,"Cost: ",c,"classification_rate: ",r)
    plt.plot(cost)
    plt.show()




if __name__=="__main__":
    main()
