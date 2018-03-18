import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import matplotlib.pyplot as plt
from util_AB import getdata
from sklearn.utils import shuffle

def error_rate(a,b):
    return np.mean(a!=b)
def relu(a):
    return a*(a>0)
def y2indicator(y):
    N = len(y)
    D = len(set(y))
    ind = np.zeros((N,D))
    for i in range(0,N):
        ind[i,y[i]]=1
    return ind
def rearrange(X):
    N = X.shape[-1]
    out = np.zeros((N,3,X.shape[0],X.shape[1]),dtype=np.float32)
    for i in range(0,N):
        for j in range(0,3):
            out[i,j,:,:]=X[:,:,j,i]
    return out/255
def filter_init(shape,pool_sz = (2,2)):
    w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(pool_sz)))
    #Not understood this line so get it checked
    return w.astype(np.float32)
def convpool(X,W,b,pool_sz=(2,2)):
    conv_out = conv2d(input=X,filters=W)
    pooled = pool.pool_2d(
    input = conv_out,
    ws = pool_sz,
    mode = "max",
    ignore_border = True)
    return relu(pooled + b.dimshuffle("x",0,"x","x"))


def main():
    train , test = getdata()
    Xtrain = rearrange(train["X"])
    Ytrain = train["y"].flatten() -1


    Xtest = rearrange(test["X"])
    Ytest = test["y"].flatten() -1
    Ytest_ind = y2indicator(Ytest)

    Xtrain, Ytrain = shuffle(Xtrain,Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    W1_shape = (20,3,5,5)
    W1_init = filter_init(W1_shape)
    b1_init = np.zeros(W1_shape[0],dtype=np.float32)

    W2_shape=(50,20,5,5)
    W2_init = filter_init(W2_shape)
    b2_init = np.zeros(W2_shape[0],dtype=np.float32)

    M = 500 # hidden layer size
    K = 10 # output size
    #weights for our vanilla network
    W3_init = np.random.randn(W2_shape[0]*5*5,M)/np.sqrt((W2_shape[0]*5*5)+M)
    b3_init = np.zeros(M,dtype = np.float32)
    W4_init = np.random.randn(M,K)/np.sqrt(M+K)
    b4_init = np.zeros(K, dtype=np.float32)
    # Now our weights are initialized
    #Defining training parameters
    N = Xtrain.shape[0]
    max_iter = 8
    print_period = 10
    batch_sz = 500
    n_batches= int(N/batch_sz)
    lr = np.float32(0.00001)
    reg = np.float32(0.01)
    mu = np.float32(0.99) # defining our momentum

    # WE will define our theano variables
    X = T.tensor4("X",dtype="float32")
    Y = T.matrix("T")

    W1 = theano.shared(W1_init,"W1")
    b1 = theano.shared(b1_init,"b1")
    W2 = theano.shared(W2_init,"W2")
    b2 = theano.shared(b2_init,"b2")
    W3 = theano.shared(W3_init.astype(np.float32),"W3")
    b3 = theano.shared(b3_init,"b3")
    W4 = theano.shared(W4_init.astype(np.float32),"W4")
    b4 = theano.shared(b4_init,"b4")

    #Initializing momentum weight changes
    dW1 = theano.shared(np.zeros(W1_init.shape,dtype=np.float32),"dW1")
    db1 = theano.shared(np.zeros(b1_init.shape,dtype=np.float32),"db1")
    dW2 = theano.shared(np.zeros(W2_init.shape,dtype=np.float32),"dW2")
    db2 = theano.shared(np.zeros(b2_init.shape,dtype=np.float32),"db2")
    dW3 = theano.shared(np.zeros(W3_init.shape,dtype=np.float32),"dW3")
    db3 = theano.shared(np.zeros(b3_init.shape,dtype=np.float32),"db3")
    dW4 = theano.shared(np.zeros(W4_init.shape,dtype=np.float32),"dW4")
    db4 = theano.shared(np.zeros(b4_init.shape,dtype=np.float32),"db4")

    #Feed-forward
    Z1 = convpool(X,W1,b1)
    Z2 = convpool(Z1,W2,b2)
    Z3 = relu(Z2.flatten(ndim=2).dot(W3)+b3)
    Py = T.nnet.softmax(Z3.dot(W4)+b4)

    #Now we shall have an estimate of the cost
    params = (W1,b1,W2,b2,W3,b3,W4,b4)
    reg_cost = np.sum((params*params).sum()for params in params)

    cost = -(Y * T.log(Py)).sum() + reg_cost
    prediction = T.argmax(Py,axis = 1)

    #We will now define our update equations
    update_W1 = W1+mu*dW1- lr*T.grad(cost,W1)
    update_b1 = b1+mu*db1 - lr * T.grad(cost,b1)
    update_W2 = W2+mu*dW2 - lr * T.grad(cost,W2)
    update_b2 = b2+mu*db2 - lr * T.grad(cost,b2)
    update_W3 = W3+mu*dW3 - lr * T.grad(cost,W3)
    update_b3 = b3+mu*db3 - lr * T.grad(cost,b3)
    update_W4 = W4+mu*dW4 - lr * T.grad(cost,W4)
    update_b4 = b4+mu*db4 - lr * T.grad(cost,b4)

    #Defining updates for our Momentum variable
    update_dW1 = dW1 * mu - lr*T.grad(cost,W1)
    update_db1 = mu*db1 - lr*T.grad(cost, b1)
    update_dW2 = mu*dW2 - lr*T.grad(cost, W2)
    update_db2 = mu*db2 - lr*T.grad(cost, b2)
    update_dW3 = mu*dW3 - lr*T.grad(cost, W3)
    update_db3 = mu*db3 - lr*T.grad(cost, b3)
    update_dW4 = mu*dW4 - lr*T.grad(cost, W4)
    update_db4 = mu*db4 - lr*T.grad(cost, b4)

    train = theano.function(
                    inputs = [X,Y],
                    updates=[
                    (W1,update_W1),
                    (b1,update_b1),
                    (W2,update_W2),
                    (b2,update_b2),
                    (W3,update_W3),
                    (b3,update_b3),
                    (W4,update_W4),
                    (b4,update_b4),
                    (dW1,update_dW1),
                    (db1,update_db1),
                    (dW2,update_dW2),
                    (db2,update_db2),
                    (dW3,update_dW3),
                    (db3,update_db3),
                    (dW4,update_dW4),
                    (db4,update_db4),
                    ],
    )

    get_prediction = theano.function(inputs=[X,Y],
                    outputs = [cost, prediction],
            )

    LL =[]
    for j in range(0,max_iter):
        for i in range(0,n_batches):
            Xbatch = Xtrain[i*batch_sz:i*batch_sz+batch_sz,]
            Ybatch = Ytrain_ind[i*batch_sz:i*batch_sz+batch_sz,]

            train(Xbatch,Ybatch)

            if i%print_period ==0:
                cost_val, prediction_val = get_prediction(Xtest,Ytest_ind)
                LL.append(cost_val)
                err = error_rate(prediction_val,Ytest)
                print("cost / error at iteration i:%d , j :%d , is %.3f/ %.3f "%(i,j,cost_val,err))

    plt.plot(LL)
    plt.show()






if __name__=="__main__":
    main()
