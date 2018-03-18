import numpy as np
import pandas

def get_image_data(V):
    if V =="train":
        df = pandas.read_csv("../large_files/train.csv")
        Y = np.asarray(df.ix[:,0])
        X = np.asarray(df.ix[:,1:])
        N , D = X.shape
        d = int(np.sqrt(D))
        X = X.reshape(N,d,d,1)
        return X,Y
    else:
        df = pandas.read_csv("../large_files/test.csv")
        X = np.asarray(df.ix[:,1:])
        N , D = X.shape
        d = int(np.sqrt(D))
        X = X.reshape(N,d,d,1)
        return X

def init_weights_bias(M1, M2):
    W = np.random.randn(M1, M2)/np.sqrt(M1+M2)
    b = np.zeros(M2,dtype = np.float32)
    return W.astype(np.float32),b.astype(np.float32)

def init_filter(shape, pool_size =(2,2)):
    W = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2]/np.prod(pool_size)))
    b = np.zeros(shape[-1])
    return W.astype(np.float32),b.astype(np.float32)

def error_rate(a,b):
    return np.mean(a!=b)

def y2indicator(y):
    N = len(y)
    D = len(set(y))
    out = np.zeros((N,D))
    for i in range(0,N):
        out[i,y[i]]=1
    return out
