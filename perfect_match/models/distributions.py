"""
Copyright (C) 2018  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import tensorflow as tf


SQRT_CONST = 1e-10


# SOURCE: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py, MIT-License
def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


# SOURCE: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py, MIT-License
def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def calculate_distance(i, it, X, Xc, nc, b, t, p=0.5, lam=10, its=10, sq=False, backpropT=False):
    Xt = tf.gather(X, it)
    nt = tf.to_float(tf.shape(Xt)[0])

    # Marginal vector for treatment.
    a = tf.concat([p * tf.ones(tf.shape(tf.where(tf.equal(t, i))[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))], 0)

    # Xt = tf.Print(Xt, [Xt], message="Xt=")
    # Xc = tf.Print(Xc, [Xc], message="Xc=")

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat([delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))], 0)

    # M = tf.Print(M, [M], message="M=")
    # col = tf.Print(col, [col], message="col=")
    # row = tf.Print(row, [row], message="row=")

    Mt = tf.concat([M, row], 0)
    Mt = tf.concat([Mt, col], 1)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)
    return D


def calculate_distances(X, t, ic, p=0.5, lam=10, its=10, sq=False, backpropT=False, num_treatments=2):
    Xc = tf.gather(X, ic)
    nc = tf.to_float(tf.shape(Xc)[0])

    # Marginal vector for control.
    b = tf.concat([(1 - p) * tf.ones(tf.shape(tf.where(tf.equal(t, 0))[:, 0:1])) / nc, p * tf.ones((1, 1))], 0)

    # Gather the treatment distributions.
    total_D = tf.zeros((1,))
    for i in range(1, num_treatments):
        it = tf.where(tf.equal(t, i))[:, 0]
        is_empty = tf.equal(tf.size(it), 0)

        D = tf.cond(
            is_empty,
            lambda: tf.zeros((1,)),
            lambda: calculate_distance(i, it, X, Xc, nc, b, t, p, lam, its, sq, backpropT)
        )
        total_D += D

    return total_D


# SOURCE: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py, MIT-License
def wasserstein(X, t, p=0.5, lam=10, its=10, sq=False, backpropT=True, num_treatments=2):
    """ Returns the Wasserstein distance between treatment groups """

    # Gather the control distribution.
    ic = tf.where(tf.equal(t, 0))[:, 0]
    is_empty = tf.equal(tf.size(ic), 0)
    total_D = tf.cond(
        is_empty,
        lambda: tf.zeros((1,)),
        lambda: calculate_distances(X, t, ic, p, lam, its, sq, backpropT, num_treatments)
    )
    return total_D[0]
