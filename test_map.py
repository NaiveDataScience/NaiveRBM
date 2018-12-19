# python: 3.7
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class RBM:
    """Restricted Boltzmann Machine."""
    
    def __init__(self, n_hidden=500, n_observe=784, learning_rate = 0.01):
        """Initialize model."""
        
        # 请补全此处代码
        self.a = np.random.normal(0, 1, [n_observe,1])
        self.b = np.random.normal(0, 1, [n_hidden, 1])
        self.W = np.random.normal(0, 1, size=(n_observe, n_hidden))
        self.v_0 = np.zeros([n_observe, 1])
        self.h_0 = np.zeros([n_hidden, 1])
        self.v_1 = np.zeros([n_observe, 1])
        self.h_1 = np.zeros([n_hidden, 1])
        self.learning_rate = learning_rate
    
    def train(self, data, time_steps):
        """Train model using data."""
        data = np.reshape(data, (data.shape[0], data.shape[1]))
        
        # 请补全此处代码
        for epoch in range(time_steps):
            N = data.shape[0]
            for i in range(N):
                self.v_0 = data[i].reshape(28 * 28, -1)
                # import pdb;pdb.set_trace()
                self.h_0 = sigmoid(self.b + np.dot(np.transpose(self.W), self.v_0))

                self.v_1 = sigmoid(self.a + np.dot(self.W, self.h_0))
                self.h_1 = sigmoid(self.b + np.dot(np.transpose(self.W), self.v_1))

                self.W = self.W + self.learning_rate * (np.dot(self.v_0, np.transpose(self.h_0))
                                              - np.dot(self.v_1, np.transpose(self.h_1)))

                self.a = self.a + self.learning_rate * (self.v_0 - self.v_1)
                self.b = self.b + self.learning_rate * (self.h_0 - self.h_1)
                if (i+1) % 1000 == 0:
                    print("epoch:%s, step:%s" % (epoch, i+1))
        np.save('w.npy', self.W)
        np.save('a.npy', self.a)
        np.save('b.npy', self.b)

    def sample(self):
        """Sample from trained model."""
        
        # 请补全此处代码
        mnist = np.load('mnist_bin.npy')

        index = np.random.randint(mnist.shape[0])
        im = mnist[index].reshape(28*28, -1)
        v = mnist[index].reshape(28*28, -1)
    
        h = sigmoid(self.b + np.dot(np.transpose(self.W), v))
        for i in range(5):
            v = sigmoid(self.a + np.dot(self.W, h))
            h = sigmoid(self.b + np.dot(np.transpose(self.W), v))
        return im, v

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':

    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy').reshape(-1, 28*28)  # 60000x784
    
    n_imgs, img_size = mnist.shape
    print(mnist.shape)

    # construct rbm model
    rbm = RBM(500, img_size)

    time_steps = 3
    # train rbm model using mnist
    rbm.train(mnist, time_steps)
    
    # sample from rbm model
    im, new_im = rbm.sample()
    im = np.reshape(im, (28, 28))
    new_im = np.reshape(new_im, (28, 28))
    plt.subplot(1, 2, 1)
    plt.imshow(im.astype("float"), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(new_im.astype("float"), cmap='gray')
    plt.show()


