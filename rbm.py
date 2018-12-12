# python: 2.7
# encoding: utf-8

import numpy as np
import math

def new_map(func, tensor):
    if type(tensor) != np.ndarray:
        return func(tensor) 
    ret = []
    for iter in tensor:
        ret.append(new_map(func, iter))

    return np.array(ret)

def new_compare(func, tensor1, tensor2):
    if type(tensor1) != np.ndarray:
        # import pdb;pdb.set_trace()
        return func(tensor1, tensor2)

    ret = []
    for (iter1, iter2) in zip(tensor1, tensor2):
        ret.append(new_compare(func, iter1, iter2))

    return np.array(ret)

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

    def train(self, data, max_epoch=10):
        """Train model using data."""

        # 请补全此处代码
        self.visible = data.reshape(-1, self.n_observe)
        for epoch in range(max_epoch):
            np.random.shuffle(data)

            ## CD loss

    def sigmod(x):
        return 1 / (1 + math.exp(-x))

    def _sample_binary(self, p):
        """
        Sample the binary vector by given probability tensor
        """
        def get_random(s):
            return np.random.uniform(0, 1)

        def compare_func(a, b):
            return 0 if a < b else 1

        return new_compare(compare_func, \
                    new_map(get_random, p), p)


    def sample(self, iter_times=10):
        """Sample from trained model."""
        # 请补全此处代码
        new_v = np.random(self.n_observe, 1)
        for iter in range(iter_times):
            p_h = self.sigmod(np.matmul( \
                    self.W.traponse(), new_v) + self.b)

            new_h = self._sample_binary(p_h)
            p_v = self.sigmod(np.matmul( \
                    self.W, new_h) + self.a)
            new_v = self._sample_binary(p_v)

        return new_v

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print (mnist.shape)

    # construct rbm model
    rbm = RBM(2, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    s = rbm.sample()
    import pdb;pdb.set_trace()


