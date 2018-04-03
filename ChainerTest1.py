'''
Created on 2018/04/02

@author: hiroy
'''
from chainer import Link, Chain, ChainList, optimizers, Variable, cuda
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import numpy as np

class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            cn1=L.Convolution2D(1,20,5),
            cn2=L.Convolution2D(20,50,5),
            l1=L.Linear(800, 500),
            l2=L.Linear(500, 10)
        )

    def __call__(self, x, t, is_train):
        if is_train:
            return F.softmax_cross_entropy(self.fwd(x), t),  F.accuracy(self.fwd(x), t)
        else:
            return F.accuracy(self.fwd(x), t)


    def fwd(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)),2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)),2)
        h3 = F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)


if __name__ == "__main__":

    N = 1000
    batchsize = 1000
    epochs = 5

    mnist = fetch_mldata('MNIST original', data_home=".")
    X = mnist.data
    Y = mnist.target

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    x_train = x_train.astype(np.float32)  #学習データはfloat32
    y_train = y_train.astype(np.int32)

    x_test = x_test.astype(np.float32)  #学習データはfloat32
    y_test = y_test.astype(np.int32)

    x_train = x_train.reshape((len(x_train), 1, 28, 28))
    x_test = x_test.reshape((len(x_test), 1, 28, 28))

    datasize = len(x_train)
    print(len(x_train))
    #print(y_train)

    model = MyChain()
    gpu_device = 0
    cuda.get_device(gpu_device).use()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    print("abe")

    sum_loss, sum_accuracy = 0, 0

    # Learning loop
    for epoch in range(1, epochs + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, datasize, batchsize):
            x = Variable(np.asarray(x_train[i:i + batchsize]))
            t = Variable(np.asarray(y_train[i:i + batchsize]))

            # Pass the loss function (Classifier defines it) and its arguments
            model.zerograds()
            loss, acc = model(x, t, True)
            loss.backward()
            optimizer.update()

            print ("acc  ", acc.data)

            sum_loss += float(loss.data) * batchsize
            sum_accuracy += float(acc.data) * batchsize

    print('train mean loss={}, accuracy={}'.format(sum_loss / datasize, sum_accuracy / datasize))

    acc = model(x_test, y_test, train=False)
    print ("acc test ", acc.data)
