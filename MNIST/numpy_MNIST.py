#This is a code that runs a deep learning backpropagation algorithm in a class based format using different activation functions.
# This project has been made explictly using numpy for the purpose of demonstrating the backpropagation chain rule.

import numpy as np
import matplotlib.pyplot as plt
from util import (get_data,view_image,forward,cross_entropy,sigmoid,softmax,classification_rate,
                     derivative_W2,derivative_b2,derivative_b1,derivative_W1,tar2ind,relu)
import pandas
import datetime
from sklearn.utils import shuffle

class MNIST(object):
    def __init__(self,M):
        self.M = M
    def train(self,X,Y,activation=1,lr=10e-7,reg=10e-7,epoch = 10):
        N,D = X.shape #Diamentionality of our data
        batch_size = 500
        n_batches = int(N/batch_size)
        ind = tar2ind(Y) # WE convert our target array into indicator matrix using one hot encoding
        _,K = ind.shape

        self.W1 = np.random.randn(D,self.M)/np.sqrt(D) #Input to hidden weight
        self.W2 = np.random.randn(self.M,K)/np.sqrt(self.M)#Hidden to output weights
        self.b1 = np.random.randn(self.M)
        self.b2 = np.random.randn(K)
        dW2 =0
        db2=0
        dW1=0
        db1=0
        mu = 0.9 # Momentum
        decay_rate = 0.99


        cost = []
        for n in range(0,200):
            #tempx , tempy = shuffle(X, ind)
            for i in range(0,n_batches):
                X_tr = X[i*batch_size:(i*batch_size+batch_size),:]
                Y_tr = Y[i*batch_size:(i*batch_size+batch_size),]
                ind = tar2ind(Y_tr)
                output,hidden = forward(X_tr,activation,self.W1,self.b1,self.W2,self.b2)


                #Performing backpropagation now
                dW2 = mu*dW2 + lr*(derivative_W2(ind,output,hidden,reg,self.W2))
                self.W2 = self.W2 + dW2
                db2= mu * db2 + lr*(derivative_b2(ind,output,reg,self.b2))
                self.b2 = self.b2 + db2
                dW1 = mu*dW1 + lr*(derivative_W1(ind,output,hidden,self.W2,X_tr,activation,reg,self.W1))
                self.W1 = self.W1 + dW1
                db1 = mu * db1 +lr *(derivative_b1(ind,output,hidden,self.W2,activation,reg,self.b1))
                self.b1 = self.b1 + db1
                c = cross_entropy(ind,output)
                cost.append(c)

                if i %10 ==0:
                    result = np.argmax(output,axis=1)
                    r = classification_rate(Y_tr,result)
                    print("iteration:- ",i,"cost:- ",c,"classification rate:- ",r)

    def predict(self,X,activation=1):
        output,_ = forward(X,activation,self.W1,self.b1,self.W2,self.b2)
        return np.argmax(output,axis = 1)



def main():   #This is a main function where we will import the data and call train and predict functions.

    X,Y = get_data("train")
    model = MNIST(200)
    activation = int(input("Enter the activation function eg. 1 or 2 \n 1.Sigmoid \n 2.tanh \n "))
    a = datetime.datetime.now()
    #attr = vars(model)

    model.train(X,Y,activation)
    X = get_data("test")
    result0=np.arange(1,X.shape[0]+1)

    result1 = model.predict(X,activation)  #This is a main function where we will import the data and call train and predict functions.
    result = np.vstack((result0.T,result1.T))
    result = result.T



    df_final =pandas.DataFrame({"ImageId":result[:,0],"Label":result[:,1]},dtype=int)
    df_final.to_csv("submission.csv",sep=",",index=False,dtype=int)
    

if __name__ == "__main__":
    main()
