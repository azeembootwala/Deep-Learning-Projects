from scipy.io import loadmat

#Please change the path from where your file can be loaded , the dataset is not in this repository 
# You can download the dataset from Kaggle  

def getdata():
    train = loadmat("../large_files/train_32x32.mat")
    test = loadmat("../large_files/test_32x32.mat")
    return train,test
