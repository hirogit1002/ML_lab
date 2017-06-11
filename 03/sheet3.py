import numpy as np
import numpy.testing as npt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
import scipy.spatial, scipy.linalg
import scipy.sparse.linalg
from scipy.misc import logsumexp
from scipy.cluster.hierarchy import linkage, dendrogram
import itertools
#%matplotlib inline

class krr(): 
    
    def __init__(self, kernel, kernelparameter, regularization):
        self.kernel = kernel
        self.kp = kernelparameter
        self.c = regularization

    def getkernel(self, X, Y=None):
        n= len(X)
        n2 = n
        if(Y==None):
            X2=np.array(X)
            x2=len(X2)
            n2 = n
        else: 
            X2 = Y
            n2 = len(X2)
            
        if(self.kernel =='gaussian'):
            w = self.kp
            X1 = (X**2).sum(1).reshape(n,1)*np.ones((n,n2))
            U1 = (X2**2).sum(1).reshape(1,n2) * np.ones([n,n2])
            D = X1 - 2*(X.dot(X2.T)) + U1
            K = np.exp(-D/(2*w**2))
        elif(self.kernel =='polynomial'):
            p= self.kp
            K = (np.dot(X,X2.T)+1)**p
        elif(self.kernel =='linear'):
            K = np.dot(X,X2.T)
        else:
            raise AssertionError("Choose from ['gaussian','polynomial','linear']")
        return K
    
    def fit(self, X, y):
        self.X_fit = X
        n= len(X)
        K = self.getkernel(X)
        self.K = K
        if(self.c==0):
            D, U =np.linalg.eigh(K)
            c = np.random.uniform(0.01,0.10,100)
            err = np.empty(len(c))
            for i in range(len(c)):
                LCI = np.diag(D)+c[i]*np.eye(n)
                LCI_inv=np.linalg.solve(LCI,np.eye(n))
                S = np.dot(np.dot(np.dot(U,np.diag(D)),LCI_inv),U.T)
                UY = np.dot(U.T,y)
                SY = np.dot(np.dot(np.dot(U,np.diag(D)),LCI_inv),UY)
                err[i] = (((y-SY)/(1-np.diag(S)))**2).mean(0)
            cidx=np.argmin(err)
            self.c = c[cidx]
        print(self.c)
        
        KK = self.K + self.c *np.eye(n)
        inv = np.linalg.solve(KK,np.eye(n))
        self.alpha= np.dot(inv,y.reshape(n,1))
        
    def predict(self, X):
        K = self.getkernel(X, self.X_fit)
        return np.dot(K, self.alpha)


