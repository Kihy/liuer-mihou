# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import sys
import numpy
from KitNET.utils import *
import json

def squeeze_features(fv, precision):
    """rounds features to siginificant figures

    Args:
        fv (array): feature vector.
        precision (int): number of precisions to use.

    Returns:
        array: rounded array of floats.

    """

    return numpy.around(fv, decimals=precision)

def quantize(x, k):
    n = 2**k - 1
    return numpy.round(numpy.multiply(n, x))/n


def quantize_weights(w, k):
    x = numpy.tanh(w)
    q = x / numpy.max(numpy.abs(x)) * 0.5 + 0.5
    return 2 * quantize(q, k) - 1


class dA_params:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, gracePeriod=10000, hiddenRatio=None, normalize=True, input_precision=None, quantize=None):
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio
        self.normalize = normalize
        self.quantize=quantize
        self.input_precision=input_precision
        if quantize:
            self.q_wbit,self.q_abit=quantize


class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(
                self.params.n_visible * self.params.hiddenRatio))

        # for 0-1 normlaization
        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf
        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
        self.n = 0

        self.rng = numpy.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))

        #quantize weights
        if self.params.quantize:
            self.W=quantize_weights(self.W, self.params.q_wbit)

        self.hbias = numpy.zeros(self.params.n_hidden)  # initialize h bias 0
        self.vbias = numpy.zeros(self.params.n_visible)  # initialize v bias 0
        # self.W_prime = self.W.T

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W.T) + self.vbias)

    def train(self, x):
        self.n = self.n + 1

        if self.params.normalize:
            # update norms
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max -
                                       self.norm_min + 0.0000000000000001)

        if self.params.input_precision:
            x=squeeze_features(x,self.params.input_precision)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x

        y = self.get_hidden_values(tilde_x)
        if self.params.quantize:
            y=quantize(y, self.params.q_abit)

        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.vbias += self.params.lr * L_vbias

        if self.params.quantize:
            self.W=quantize_weights(self.W, self.params.q_wbit)
            self.hbias=quantize_weights(self.hbias, self.params.q_wbit)
            self.vbias=quantize_weights(self.vbias, self.params.q_wbit)
        # the RMSE reconstruction error during training
        return numpy.sqrt(numpy.mean(L_h2**2))

    def reconstruct(self, x):
        y = self.get_hidden_values(x)

        try:
            if self.params.quantize:
                y=quantize(y, self.params.q_abit)
        except AttributeError as e:
            pass
            
        z = self.get_reconstructed_input(y)
        return z

    def get_params(self):
        params={
        "W":self.W,
        "hbias":self.hbias,
        "vbias":self.vbias
        }
        return params

    def set_params(self, new_param):
        self.W=new_param["W"]
        self.hbias=new_param["hbias"]
        self.vbias=new_param["vbias"]

    def execute(self, x):  # returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            # 0-1 normalize
            try:
                if self.params.normalize:
                    x = (x - self.norm_min) / (self.norm_max -
                                               self.norm_min + 0.0000000000000001)

                if self.params.input_precision:
                    x=squeeze_features(x,self.params.input_precision)

            except AttributeError as e:
                pass

            z = self.reconstruct(x)
            rmse = numpy.sqrt(((x - z) ** 2).mean())  # MSE
            return rmse

    def inGrace(self):
        return self.n < self.params.gracePeriod
