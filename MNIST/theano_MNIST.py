#This is the implementation of the the famous MNIST dataset using theano
# I have also incorporated momentum into the algorithm
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os
import pandas as pd 

def relu(a):
    return a * (a>0)
def get_normalized_data():
    print("Reading in and normalized data...")

    if not os.path.exists('../large_files/train.csv'):
        print('Looking for ../large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std # normalize the data
    Y = data[:, 0]
    return X, Y

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(a,b):
    return np.mean(a!=b)

def main():
    X,Y = get_normalized_data()
    # We will now devided our dataset into test and train
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000,]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:,]

    # We will now convert our target vector into an indicator matrix using one hot encoding

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N,D = Xtrain.shape
    M = 300 # Hidden layer size
    K = 10  # output layer size
    lr = 0.00004 # Learning rate (Step size)
    reg = 0.01 # l2 regularization parameter
    max_iter = 20
    batch_sz = 500
    n_batches= N/batch_sz

    W1_init  = np.random.randn(D,M)/np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K)/np.sqrt(M)
    b2_init = np.zeros(K)
    dW1_init = np.zeros((D, M))
    db1_init = np.zeros(M)
    dW2_init = np.zeros((M, K))
    db2_init = np.zeros(K)
    mu = 0.9 # momentum



    # Theano implementation starts now !!!
    thX = T.matrix("X")
    thT = T.matrix("T")
    # We declare shared variables as in theano only shared variables can be updated
    W1 = theano.shared(W1_init,"W1")
    b1 = theano.shared(b1_init,"b1")
    W2 = theano.shared(W2_init,"W2")
    b2 = theano.shared(b2_init,"b2")
    dW1 = theano.shared(dW1_init,"dW1")# dw needs to be a shared variable to be updated
    dW2 = theano.shared(dW2_init,"dW2")
    db1 = theano.shared(db1_init,"db1")
    db2 = theano.shared(db2_init,"db2")

    # Running the feed forward step

    thZ = relu(thX.dot(W1)+ b1)
    thY = T.nnet.softmax(thZ.dot(W2)+b2)

    cost = -(thT*T.log(thY)).sum() + reg*((W1*W1).sum()+(b1*b1).sum()+(W2*W2).sum() + (b2*b2).sum())

    prediction = T.argmax(thY, axis = 1)

    # So now we have our cost and prediction functions
    # We will now write our update functions

    update_dW2 = mu*dW2 - lr*T.grad(cost, W2)

    update_W2 = W2 + mu*dW2 - lr*T.grad(cost, W2)

    update_db2 = mu*db2 - lr*T.grad(cost, b2)

    update_b2 = b2 + mu*db2 - lr*T.grad(cost, b2)

    update_dW1 = mu*dW1 - lr*T.grad(cost, W1)

    update_W1 = W1 + mu*dW1 - lr*T.grad(cost, W1)

    update_db1 = mu*db1 - lr*T.grad(cost, b1)

    update_b1 = b1 + mu*db1 - lr*T.grad(cost, b1)


    train = theano.function(inputs = [thX,thT],
                            updates =[(W1,update_W1),(W2,update_W2),(b1,update_b1),(b2,update_b2)]
                            + [(dW1, update_dW1), (dW2, update_dW2),(db2, update_db2), (db1, update_db1)]
                            ,)
    get_prediction = theano.function(inputs=[thX,thT],outputs=[cost,prediction],)
    ll = []

    for j in range(0,max_iter):
        for i in range(0,int(n_batches)):
            Xbatch = Xtrain[i*batch_sz:(i*batch_sz+batch_sz),]
            Ybatch = Ytrain_ind[i*batch_sz:(i*batch_sz+batch_sz),]

            train(Xbatch,Ybatch)
            if i % 10 == 0:
                cost_test , prediction_test = get_prediction(Xtest,Ytest_ind)
                err = error_rate(prediction_test,Ytest)
                ll.append(cost_test)
                print("cost/prediction at i:%d, j:%d is %.3f/%.3f" % (i,j,cost_test,err))

    plt.plot(ll)
    plt.show()




if __name__ == "__main__":
    main()
