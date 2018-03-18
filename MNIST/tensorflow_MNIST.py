#Implementation of the MNIST dataset using CNN
# Architecture :- LeNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import get_data

def error_rate(a,b):
    return np.mean(a!=b)

def y2indicator(y):
    N=len(y)
    D=len(set(y))
    ind = np.zeros((N,D),dtype=np.float32)
    for i in range(N):
        ind[i,y[i]]=1
    return ind
def init_filter(shape, pool_size=(2,2)):
    W = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2]/np.prod(pool_size)))
    return W

#converting the input matrix NxD into a tensor of shape NxWxHxC for tensorflow
def rearrange(X):
    N = X.shape[0]
    out = np.zeros((N,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])),1),dtype=np.float32)
    for i in range(N):
        out[i,:,:,0] = X[i,:].reshape(int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
    return out

def convpool(X,W,b,padding_type):
    conv_out = tf.nn.conv2d(input=X,filter=W,strides=[1,1,1,1],padding=padding_type)
    conv_out = tf.nn.bias_add(conv_out, b)
    conv_out = tf.nn.relu(conv_out)
    out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    out = tf.tanh(out)
    return out

#Normalization not done yet
def main():
    X,Y = get_data("train")
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000,]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:,]
    Xtest = rearrange(Xtest)/255
    Ytest_ind = y2indicator(Ytest)
    Ytrain_ind = y2indicator(Ytrain)
    Xtrain = rearrange(Xtrain)/255

    W1_shape=(5,5,1,6)
    W1_init = init_filter(W1_shape)
    b1_init = np.zeros((W1_shape[-1]),dtype=np.float32)
    W2_shape=(5,5,6,16)
    W2_init = init_filter(W2_shape)
    b2_init = np.zeros((W2_shape[-1]),dtype=np.float32)

    M = 500 # Number of hidden layers
    K = 10
    #Weights for fully connected layer
    W3_init = np.random.randn(W2_shape[-1]*5*5,M)/np.sqrt(W2_shape[-1]*5*5 + M)
    b3_init = np.zeros(M,dtype=np.float32)

    W4_init = np.random.randn(M,K)/np.sqrt(M+K)
    b4_init = np.zeros(K,dtype=np.float32)

    max_iter = 10
    N = Xtrain.shape[0]
    batch_size = 100
    n_batches = int(N/batch_size)

    X = tf.placeholder(tf.float32,shape=(batch_size, 28,28,1),name="X")
    T = tf.placeholder(tf.float32,shape=(batch_size,K),name = "T")

    W1 = tf.Variable(W1_init.astype(np.float32), name = "W1")
    b1 = tf.Variable(b1_init, name = "b1")
    W2 = tf.Variable(W2_init.astype(np.float32), name = "W2")
    b2 = tf.Variable(b2_init, name = "b2")
    W3 = tf.Variable(W3_init.astype(np.float32), name =  "W3")
    b3 = tf.Variable(b3_init,name ="b3")
    W4 = tf.Variable(W4_init.astype(np.float32), name =  "W4")
    b4 = tf.Variable(b4_init,name="b4")

    #Feed forward
    Z1 = convpool(X,W1,b1,"SAME")
    Z2 = convpool(Z1,W2,b2,"VALID")
    Z2_shape = Z2.get_shape().as_list()
    Z2_f = tf.reshape(Z2,[Z2_shape[0],np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2_f, W3)+b3)
    yish = tf.matmul(Z3,W4)+b4

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=yish,labels=T))

    training_op = tf.train.AdamOptimizer(0.01).minimize(cost)

    prediction_op = tf.argmax(yish,axis = 1)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        LL = []
        for j in range(0,max_iter):
            for i in range(0,n_batches):
                Xbatch = Xtrain[i*batch_size:(i+1)*batch_size,]
                Ybatch = Ytrain_ind[i*batch_size:(i+1)*batch_size,]

                session.run(training_op,feed_dict={X:Xbatch,T:Ybatch})
                if i % 10 ==0:
                    test_cost = 0
                    pred = np.zeros(len(Xtest))
                    for k in range(len(Xtest)//batch_size):
                        Xtest_b = Xtest[k*batch_size:(k+1)*batch_size,]
                        Ytest_b = Ytest_ind[k*batch_size:(k+1)*batch_size,]
                        test_cost += session.run(cost,feed_dict={X:Xtest_b,T:Ytest_b})
                        pred[k*batch_size:(k+1)*batch_size] =session.run(prediction_op, feed_dict={X:Xtest_b,T:Ytest_b})
                    err = error_rate(pred,Ytest)
                    LL.append(test_cost)
                    print("Cost / err at iteration j=%d, i=%d: %.3f / %.3f" % (j, i, test_cost, err))
    """
    result0=np.arange(1,Xtest.shape[0]+1)
    result1 = pred
    result = np.vstack((result0.T,result1.T))
    result = result.T

    df_final =pandas.DataFrame({"ImageId":result[:,0],"Label":result[:,1]},dtype=int)
    df_final.to_csv("submission1.csv",sep=",",index=False)
    """

    plt.plot(LL)
    plt.show()



if __name__ == "__main__":
    main()
