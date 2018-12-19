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
        self.a = np.random.normal(0, 1, n_observe)
        self.b = np.random.normal(0, 1, n_hidden)
        self.W = np.random.normal(0, 1, size=(n_observe, n_hidden))
        self.v_0 = np.zeros(n_observe)
        self.h_0 = np.zeros(n_hidden)
        self.v_1 = np.zeros(n_observe)
        self.h_1 = np.zeros(n_hidden)
        self.learning_rate = learning_rate
    
    def train(self, data, time_steps):
        """Train model using data."""
        data = np.reshape(data, (data.shape[0], data.shape[1]))
        
        # 请补全此处代码
        for epoch in range(time_steps):
            N = data.shape[0]
            for i in range(N):
                self.v_0 = data[i]
                self.h_0 = sigmoid(self.b + np.dot(np.transpose(self.W), self.v_0))

                self.v_1 = sigmoid(self.a + np.dot(self.W, self.h_0))
                self.h_1 = sigmoid(self.b + np.dot(np.transpose(self.W), self.v_1))

                self.W = self.W + self.learning_rate * (np.dot(self.v_0[:, None], self.h_0[None, :])
                                              - np.dot(self.v_1[:, None], self.h_1[None, :]))

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
        mnist = np.load('mnist_bin2.npy')
        w = np.load('w.npy')
        a = np.load('a.npy')
        b = np.load('b.npy')
        index = np.random.randint(mnist.shape[0])
        im = mnist[index].copy()
        v = mnist[index].copy()
        import pdb;pdb.set_trace()
        h = sigmoid(b + np.dot(np.transpose(w), v))
        for i in range(3):
            v = sigmoid(a + np.dot(w, h))
            h = sigmoid(b + np.dot(np.transpose(w), v))
        return im, v

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':

    # load mnist dataset, no label
    mnist = np.load('mnist_bin2.npy')  # 60000x784
    n_imgs, img_size = mnist.shape
    print(mnist.shape)

    # construct rbm model
    rbm = RBM(500, img_size)

    time_steps = 3
    # train rbm model using mnist
    # rbm.train(mnist, time_steps)
    
    # sample from rbm model
    im, new_im = rbm.sample()
    im = np.reshape(im, (28, 28))
    new_im = np.reshape(new_im, (28, 28))
    plt.subplot(1, 2, 1)
    plt.imshow(im.astype("float"), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(new_im.astype("float"), cmap='gray')
    plt.show()