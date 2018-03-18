import pandas
import numpy as np
from sklearn.utils import shuffle
def get_data(balanced=False):
    X_class_3 =[]
    Y_class_3=[]
    df = pandas.read_csv("balance-scale.csv",header=None)
    Y = np.asarray(df.ix[:,0])
    #for i in range(1,df.shape[1]):
        #df.ix[:,i]/=df.ix[:,i].max() # Normalization by replacing each value with the max in that array
    X = np.asarray(df.ix[:,1:])
    for i in range(0,Y.shape[0]):
        if Y[i]=="L":
            Y[i]=0
        elif Y[i]=="R":
            Y[i]=1
        else:
            Y[i]=2
            X_class_3.append(X[i,:])
            Y_class_3.append(Y[i])
    if balanced:
        X_class_3=np.asarray(X_class_3)
        Y_class_3=np.asarray(Y_class_3)
        X_class_3= np.repeat(X_class_3,4,axis=0)
        Y_class_3=np.repeat(Y_class_3,4)
        Y = np.hstack((Y.T,Y_class_3.T))
        X =np.vstack((X,X_class_3))
        X,Y = shuffle(X,Y)

        return X,Y
    else:
        X,Y = shuffle(X,Y)
        return X,Y
