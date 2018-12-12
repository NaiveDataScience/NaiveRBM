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

def add1(x):
	return x + 1

def _sample_binary(p):
    """
    Sample the binary vector by given probability tensor
    """
    def get_random(s):
        return np.random.uniform(0, 1)

    def compare_func(a, b):
        return 0 if a < b else 1

    mapping = new_map(get_random, p)
    return new_compare(compare_func, \
                mapping, p)


if __name__ == '__main__':
	s = _sample_binary(np.array([[0.3,0.4],[0.5, 0.8]]))
	import pdb;pdb.set_trace()

