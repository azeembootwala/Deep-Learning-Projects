#This is a tensor flow implementation of the Street view house number dataset
#Using Convolutional Neural networks
#Author Azeem Bootwala
#Date:- 13th September,2017
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat

def error_rate(a,b):
    return np.mean(a!=b)
def y2indicator(y):
    N=len(y)
    K = len(set(y))
    ind = np.zeros((N,K),dtype=np.float32)
    for i in range(0,N):
        ind[i,y[i]]=1
    return ind

def reshape(X):
    N = X.shape[-1]
    out = np.zeros((N,X.shape[0],X.shape[1],3),dtype = np.float32)
    for j in range(0,N):
        for i in range(0,3):
            out[j,:,:,i] = X[:,:,i,j]
    return out
def init_filters(shape,poolsize=(2,2)):
    W = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2]/np.prod(poolsize)))
    return W

def convpool(X,W,b,poolsize=(2,2)):
    conv_out = tf.nn.conv2d(input=X,filter=W,strides=[1,1,1,1],padding="SAME")
    conv_out = tf.nn.bias_add(conv_out,b)
    pool_out = tf.nn.max_pool(
                conv_out,
                ksize=[1,2,2,1],
                strides=[1,2,2,1],
                padding="SAME"
    )
    return pool_out

def main():
    train = loadmat("../large_files/train_32x32.mat")
    test = loadmat("../large_files/test_32x32.mat")
    #The data matrix is in the format (W * H* C* N) but tensorflow needs 4-D tensors
    #In the format N * H * W * C
    Xtrain = reshape(train["X"])
    Ytrain = train["y"].flatten() -1 # Conversion from MATLAB indices to python indices
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = reshape(test["X"])
    Ytest = test["y"].flatten() -1
    Ytest_ind = y2indicator(Ytest)
    #Added part
    Xtrain = Xtrain[:73000,]
    Ytrain = Ytrain[:73000]
    Xtest = Xtest[:26000,]
    Ytest = Ytest[:26000]
    Ytest_ind = Ytest_ind[:26000,]
    #Added part

    #Now that our data is ready we can create our weight matrices
    #Since we are using a LeNet model we will create our weights acording to the LeNet paper

    W1_shape = (5,5,3,20) # Tensorflow has a filter format of (w,h,c,maps)
    W1_init = init_filters(W1_shape).astype(np.float32)
    b1_init = np.zeros(W1_shape[-1],dtype=np.float32)

    W2_shape = (5,5,20,50) # Tensorflow has a filter format of (w,h,c,maps)
    W2_init = init_filters(W2_shape).astype(np.float32)
    b2_init = np.zeros(W2_shape[-1],dtype=np.float32)

    #Now our our input matrix after 2 conv and 2 pooling will become N* 8 * 8 * 50
    #Hence our weight matrix for our FC network has to compy with this change

    M = 500 # Hidden layer size
    K = 10 # 10 output classes
    max_iter = 8
    W3_init = np.random.randn(W2_shape[-1]*8*8,M)/np.sqrt(W2_shape[-1]*8*8 + M)
    b3_init = np.zeros(M, dtype=np.float32)

    W4_init = np.random.randn(M,K)/np.sqrt(K + M)
    b4_init = np.zeros( K , dtype=np.float32)

    #Defining our Grad descent parameters
    max_iter = 20
    N = Xtrain.shape[0]
    batch_size = 500
    n_batches = int(N/batch_size)

    # Defining our tensorflow graph
    X = tf.placeholder(tf.float32,shape=(batch_size,32,32,3),name = "X")
    T = tf.placeholder(tf.float32,shape=(batch_size,K),name="T") #todo

    W1 = tf.Variable(W1_init,"W1")
    b1 = tf.Variable(b1_init,"b1")
    W2 = tf.Variable(W2_init,"W2")
    b2 = tf.Variable(b2_init,"b2")
    W3 = tf.Variable(W3_init.astype(np.float32),"W3")
    b3 = tf.Variable(b3_init,"b3")
    W4 = tf.Variable(W4_init.astype(np.float32),"W4")
    b4 = tf.Variable(b4_init,"b4")

    # WE have to now define our convpool function to genereate a feed forward graph
    Z1 = convpool(X,W1,b1)
    Z2 = convpool(Z1,W2,b2)
    # Now we have to flatten our output to modify it for the FC
    Z2_shape = Z2.get_shape().as_list()
    Z2_r = tf.reshape(Z2,[Z2_shape[0],np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2_r,W3)+b3)
    Yish = tf.matmul(Z3,W4)+b4

    #lets define our cost
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish,labels=T))

    training_op = tf.train.RMSPropOptimizer(0.0001,decay=0.99,momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish,1)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        LL = []


        for j in range(0,max_iter):
            for i in range(0,n_batches):
                Xbatch = Xtrain[i*batch_size:i*batch_size+batch_size,]
                Ybatch = Ytrain_ind[i*batch_size:i*batch_size+batch_size,]

                #if len(Xbatch)==batch_size:
                session.run(training_op,feed_dict={X:Xbatch,T:Ybatch})

                if i%10 ==0:
                    test_cost = 0
                    prediction = np.zeros(len(Xtest))
                    for k in range(len(Xtest) // batch_size):
                        Xtestbatch = Xtest[k*batch_size:(k*batch_size + batch_size),]
                        Ytestbatch = Ytest_ind[k*batch_size:(k*batch_size + batch_size),]
                        test_cost += session.run(cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                        prediction[k*batch_size:(k*batch_size + batch_size)] = session.run(
                                    predict_op, feed_dict={X: Xtestbatch})
                    LL.append(test_cost)
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration j=%d, i=%d: %.3f / %.3f" % (j, i, test_cost, err))

        plt.plot(LL)
        plt.show()

if __name__=="__main__":
    main()
