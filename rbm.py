import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer', '-y', required=False, type=int, default=100)
parser.add_argument('--learning_rate', '-l', required=False, type=float, default=0.01)
args = parser.parse_args()


def new_map(func, tensor):
    if type(tensor) != np.ndarray:
        return func(tensor) 
    ret = []
    for iter in tensor:
        ret.append(new_map(func, iter))

    return np.array(ret)

def new_compare(func, tensor1, tensor2):
    if type(tensor1) != np.ndarray:
        return func(tensor1, tensor2)

    ret = []
    for (iter1, iter2) in zip(tensor1, tensor2):
        ret.append(new_compare(func, iter1, iter2))

    return np.array(ret)

def compare_img(raw_img, gen_img):
    plt.subplot(1,2,1)
    plt.imshow(raw_img.reshape((28, 28)), cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(gen_img.reshape((28, 28)), cmap="gray")
    plt.savefig("result.png")

def sigmoid(x):
    try:
        ret = 1.0 / (1.0 + math.exp(-x))
        return ret
    except:
        return 0

class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=100, n_observe=784, alpha=0.01):
        """Initialize model."""
        self.n_observe = n_observe
        self.n_hidden = n_hidden
        self.v = np.random.normal(0, 1, [n_observe, 1]) # visual variables [n_observe, 1]
        self.h = np.random.normal(0, 1, [n_hidden, 1]) # hidden variables [n_hidden, 1]
        self.W = np.random.normal(0, 1, [n_observe, n_hidden]) # Weight matrix
        self.a = np.zeros([n_observe, 1]) # bias for the visual variables
        self.b = np.zeros([n_hidden, 1]) # bias for the hidden variables
        # 请补全此处代码

        self.alpha = alpha # learning rate


    def train(self, data, max_epoch=3):
        """Train model using data."""

        # 请补全此处代码
        self.visible = data.reshape(-1, self.n_observe)
        for epoch in range(max_epoch):
            np.random.shuffle(data)
            for v in data:
                ## CD loss
                v = v.reshape(-1, 1)
                h_dist = new_map(sigmoid, \
                    np.matmul(np.transpose(self.W), v) + self.b)

                # h_sample = self._sample_binary(h_dist)
                h_sample = h_dist
                v_dist = new_map(sigmoid, \
                    np.matmul(self.W, h_sample) + self.a)

                # v_sample = self._sample_binary(v_dist)
                v_sample = v_dist
                h_dist2 = new_map(sigmoid, \
                    np.matmul(np.transpose(self.W), v_sample) + self.b)
                
                # h_sample2 = self._sample_binary(h_dist2)
                h_sample2 = h_dist2
                ## Update weight
                self.W += self.alpha * \
                    (np.matmul(v, np.transpose(h_sample)) -
                        np.matmul(v_sample, np.transpose(h_sample2)))

                self.a += self.alpha * (v - v_sample)
                self.b += self.alpha * (h_sample - h_sample2)

            print (self._energy(v, h_sample2))

            np.save("w.npy", W)
            np.save("a.npy", a)
            np.save("b.npy", b)
    

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

    def _energy(self, visible, hidden):
        return -np.inner(self.a.flatten(), visible.flatten()) - np.inner(self.b.flatten(), hidden.flatten()) \
            -np.matmul(np.matmul(np.transpose(visible), self.W), hidden)



    def sample(self, new_v, iter_times=3):
        """Sample from trained model."""
        # 请补全此处代码

        self.W = np.load("w.npy")
        self.a = np.load("a.npy").reshape(-1, 1)
        self.b = np.load("b.npy").reshape(-1, 1)
        for iter in range(iter_times):
            p_h = new_map(sigmoid, np.matmul(np.transpose(self.W), new_v) + self.b)
            
            # new_h = self._sample_binary(p_h)
            new_h = p_h
            p_v = new_map(sigmoid, np.matmul(self.W, new_h) + self.a)
            # new_v = self._sample_binary(p_v)
            new_v = p_v
            # import pdb;pdb.set_trace()
        return new_v

# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    
    num_layers = args.hidden_layer
    lr = args.learning_rate
    # load mnist dataset
    mnist = np.load('mnist_bin.npy')  # 60000x784
    n_imgs, img_size = mnist.shape

    # construct rbm model
    rbm = RBM(num_layers, img_size, lr)

    # train rbm model using mnist
    # rbm.train(mnist[0:6000], 3)

    # sample from rbm model
    random_index = np.random.randint(len(mnist))
    raw_img = mnist[random_index].reshape(-1, 1)
    gen_img = rbm.sample(raw_img, 3)
    compare_img(raw_img, gen_img)
    
