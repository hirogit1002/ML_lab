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

def zero_one_loss(y_true, y_pred):
    assert(len(y_true) == len(y_pred))
    pred = np.array(y_pred[:,0])
    true = np.array(y_true[:,0])
    true[np.where(true==-1)]=0
    b = np.mean(pred)
    pred[np.where(pred>=b)] = 1
    pred[np.where(pred<b)] = 0
    right = (true == pred).astype(np.int64).sum()
    loss = (len(true)-right)/len(true)
    return loss

def cv(X, y, method, parameters,nfolds=10, nrepetitions=5,loss_function=zero_one_loss):
    n=len(X)
    d = len(X[0])
    e = n % nfolds
    div = n-e
    nom = int(div/nfolds)
    knl = parameters['kernel']
    reg = parameters['regularization']
    kp = parameters['kernelparameter']
    losssum =np.zeros(len(reg)*len(kp))
    krrset = [method(knl,kp[i],reg[j]) for i in range(len(kp)) for j in range(len(reg))]
    for i in range(nrepetitions):
        partidx=np.append((np.ones((nfolds,nom))*np.arange(nom)).reshape(div),np.arange(e))
        np.random.shuffle(partidx)
        for j in range(nfolds):
            [krrset[a].fit(X[np.where(partidx!=j)],y[np.where(partidx!=j)]) for a in range(len(krrset))]
            yy = [krrset[a].predict(X[np.where(partidx!=j)]) for a in range(len(krrset))]
            loss = [loss_function(y[np.where(partidx!=j)].reshape(len(y[np.where(partidx!=j)]),1), np.array(yy[a])) for a in range(len(yy)) ]
            losssum = losssum +np.array(loss)
    D = (losssum -losssum.mean())**2
    return krrset[np.argmin(D)]




class krr(): 
    
    def __init__(self, kernel, kernelparameter, regularization):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
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
 
        if((self.kernel =='gaussian') or (self.kernel ==['gaussian'])):
            w = self.kp
            X1 = (X**2).sum(1).reshape(n,1)*np.ones((n,n2))
            U1 = (X2**2).sum(1).reshape(1,n2) * np.ones([n,n2])
            D = X1 - 2*(X.dot(X2.T)) + U1
            K = np.exp(-D/(2*w**2))
        elif((self.kernel =='polynomial') or (self.kernel ==['polynomial'])):
            p= self.kp
            K = (np.dot(X,X2.T)+1)**p
        elif((self.kernel =='linear') or (self.kernel ==['linear'])):
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
            c = np.logspace(-4,4,10)
            cc = c.reshape(len(c),1,1)
            br = np.ones((len(c),1,1))
            I = np.eye(len(K))*br
            UL =np.dot(U,np.diag(D))
            UtY =np.dot(U.T,y.reshape(len(y),1))
            L=br*np.diag(D)
            LCI=L +(I*cc)
            LCI_inv=1/LCI
            ULCI = np.dot(LCI_inv.transpose(0,2,1),K.T).transpose(0,2,1)
            S = np.dot(ULCI,U.T)
            SY = np.dot(ULCI,UtY)
            diag=np.dot(I*S,np.ones((len(y),1)))-1
            err = (((y.reshape(len(y),1)*br-SY)/diag)**2).mean(1)
            cidx=np.argmin(err[:,0])
            self.c = c[cidx]
            self.regularization =  c[cidx]


        """
        if(self.c==0):
            D, U =np.linalg.eigh(K)
            c = np.logspace(-2,2,10)
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
            self.regularization =  c[cidx]
        """


        KK = self.K + self.c *np.eye(n)+np.eye(n)*0.000000001
        inv = np.linalg.solve(KK,np.eye(n))
        self.alpha= np.dot(inv,y.reshape(n,1))
        
    def predict(self, X):
        K = self.getkernel(X, self.X_fit)
        return np.dot(K, self.alpha)


