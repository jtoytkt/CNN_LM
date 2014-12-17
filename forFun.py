import theano
import theano.tensor as T
from theano import  config
from theano import pp
import numpy as np


arr = np.array([1, 3, 2, 4, 5])

print arr.argsort()[-3:][::-1]

