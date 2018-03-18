import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle

from cnn_util import y2indicator, error_rate, init_weights_bias, init_filter, get_image_data
class ConvPoolLayer(object):
    def __init__(self, mi , mo, fw , fh , pool_size=(2,2)):
        self.shape = (fw,fh,mi,mo)
        W , b = init_filter(self.shape)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W , self.b]

    def forward(self, X, padding_type):
        conv_out = tf.nn.conv2d(X,self.W , strides = [1,1,1,1], padding=padding_type)
        conv_out = tf.nn.bias_add(conv_out,self.b)
        pool_out = tf.nn.max_pool(conv_out, ksize = [1,2,2,1], strides =[1,2,2,1], padding = "SAME")
        return tf.nn.relu(pool_out)

class HiddenLayer(object):
    def __init__(self, M1, M2,an_count):
        W , b =init_weights_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
        self.count = an_count

    def forward(self,X):
        return tf.nn.relu(tf.matmul(X, self.W)+self.b)

class CNN(object):
    def __init__(self, conv_pool_layer_sizes, hidden_layer_sizes):
        self.conv_pool_layer_sizes = conv_pool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate = 0.01,epoch = 4 , batch_size = 100, reg=10e-3,show_fig=True):
        X,Y = shuffle(X,Y)
        K = len(set(Y))
        Xvalid = X[-1000:]
        Yvalid_flat= Y[-1000:]
        X = (X[:-1000]).astype(np.float32)
        Y = y2indicator(Y[:-1000]).astype(np.float32)
        Yvalid = y2indicator(Yvalid_flat).astype(np.float32)

        mi = X.shape[3]
        self.conv_layer = []
        for fw, fh , mo in self.conv_pool_layer_sizes:
            conv_obj = ConvPoolLayer(mi,mo,fw,fh)
            self.conv_layer.append(conv_obj)
            mi = mo

        M1 = np.prod(self.conv_layer[-1].shape[:2])*self.conv_layer[-1].shape[-1]
        count = 0
        self.hidden_layer = []
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2 , count)
            self.hidden_layer.append(h)
            M1 = M2
            count+=1
        W , b = init_weights_bias(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W , self.b]

        for h in self.conv_layer:
            self.params +=h.params
        for h in self.hidden_layer:
            self.params +=h.params

        tfX = tf.placeholder(tf.float32,shape=(None,X.shape[1],X.shape[2],X.shape[3]), name="X")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name ="T")

        act = self.forward(tfX)
        N = X.shape[0]

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=tfT)) + rcost

        training_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        predict_op = self.predict(tfX)

        n_batches = N // batch_size

        init = tf.global_variables_initializer()
        LL = []

        with tf.Session() as session:
            session.run(init)

            for i in range(0, epoch):
                for j in range(0, n_batches):
                    Xbatch = X[j*batch_size:(j+1)*batch_size]
                    Ybatch = Y[j*batch_size:(j+1)*batch_size]

                    session.run(training_op,feed_dict={tfX:Xbatch,tfT:Ybatch})

                    if j % 10 == 0:
                        c = session.run(cost, feed_dict={tfX:Xvalid, tfT:Yvalid})
                        LL.append(c)
                        p = session.run(predict_op, feed_dict={tfX:Xvalid})
                        e = error_rate(p , Yvalid_flat)

                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(LL)
            plt.show()


    def forward(self,X, first=True):
        Z = X
        for c in self.conv_layer:
            if first:
                Z = c.forward(Z,"SAME")
                first=False
            else:
                Z = c.forward(Z,"VALID")
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z,[-1, np.prod(Z_shape[1:])])
        for h in self.hidden_layer:
            Z = h.forward(Z)
        return tf.matmul(Z,self.W)+self.b

    def predict(self,X):
        X = self.forward(X)
        return tf.argmax(X,axis=1)

def main():
    X, Y = get_image_data("train")
    model = CNN([(5,5,20),(5,5,50)],
                [500,300,200,100])
    model.fit(X,Y, show_fig=True)

if __name__ == "__main__":
    main()
