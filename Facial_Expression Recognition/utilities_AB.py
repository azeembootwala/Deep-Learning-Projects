import numpy as np
from sklearn.utils import shuffle

def get_data(balanced=True):

    first = True
    Y =[]
    X =[]
    X_class1=[]
    for line in open("../large_files/fer2013.csv"):
        if first:
            first=False
        else:
            row = line.split(",")
            if int(row[0])==1:
                X_class1.append([int (p) for p in row[1].split(" ")])
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split(" ")])
            else:
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split(" ")])


    X_class1 = np.array(X_class1) / 255.0
    X ,Y = np.array(X) / 255.0 , np.array(Y)

    if balanced:
        X_class1 = np.repeat(X_class1,9,axis=0)
        X = np.vstack([X,X_class1])
        Y_class1 = np.ones(X_class1.shape[0],)
        Y = np.concatenate((Y,Y_class1))
        X,Y = shuffle(X,Y)
    return X,Y
