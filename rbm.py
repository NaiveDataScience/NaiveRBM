# python: 2.7
# encoding: utf-8

import numpy as np
import math

class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""
        self.v = np.random.rand(n_observe, 1) # hidden variables [n_observe, 1]
        self.h = np.random.rand(n_hidden, 1) # hidden variables [n_hidden, 1]
        self.W = np.random.rand(n_observe, n_hidden) # Weight matrix
        self.a = np.random.rand(n_observe, 1) # bias for the visual variables
        self.b = np.random.rand(n_hidden, 1) # bias for the hidden variables
        # 请补全此处代码

        pass

    def train(self, data):
        """Train model using data."""

        # 请补全此处代码
        pass

    def sigmod(x):
        return 1 / (1 + math.exp(-x))

    def sample(self, iter_times=10):
        """Sample from trained model."""
        # 请补全此处代码

        for iter in range(iter_times):
            p_h = self.sigmod(np.matmul( \
                    self.W.traponse(), self.v) + self.b)

            slef.v = self._sample_binary(p_h)


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print mnist.shape

    # construct rbm model
    rbm = RBM(2, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    s = rbm.sample()
