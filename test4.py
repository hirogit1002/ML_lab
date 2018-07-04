import numpy as np
from imgproc import*
import matplotlib.pyplot as plt
import argparse
import kpca
import pandas as pd
import cv2
import glob

perm = np.random.permutation(10)[:5]
print(perm)


def student(Y):
    N = Y.shape[0]
    Q = 1 / (1 + scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean'))
    Q /= np.sum(Q) - N
    return np.log(Q)

def student(Y):
    D = scipy.spatial.distance.cdist(Y, Y, 'sqeuclidean')
    D_inv = 1/(D+1)
    unter = (D_inv.sum(0)-1)[np.newaxis,:]
    logQ = np.log(D_inv) - np.log(unter)
    return logQ


def objective(logP,logQ):
    return (np.exp(logP)*(logP - logQ)).sum()

def gradient(logP,Y):
    logQ = student(Y)
    dist = squareform(pdist(Y)) ** 2
    D = 1. / (1. + dist)
    Yij = np.array([y - Y for y in Y])
    PQ = (np.exp(logP) - np.exp(logQ))
    gradY = 4. * np.array([(PQ) * (Yij[:, :, i]) * (D) for i in range(len(Y[0]))]).sum(1).T
    return -gradY


def TSNE(X, Y0,color, perplexity=25, learningrate=1.0, nbiterations=250, steps=False, silent=False):
    if (not silent):
        print('get affinity matrix')
    logP = utils.getaffinity(X, perplexity)
    Y = Y0 * 1
    dY = Y * 0
    if not silent:
        print('run t-SNE')
    for t in range(nbiterations):
        logQ = student(Y)
        if t % 50 == 0:
            if not silent:
                print('%3d %.3f' % (t, objective(logP, logQ)))
            if steps:
                plt.scatter(*Y.T, c=color)
                plt.title('t=' + str(t))
                plt.show()
        dY = (0.5 if t < 100 else 0.8) * dY + learningrate * gradient(logP, Y)
        Y = Y - dY
    return Y