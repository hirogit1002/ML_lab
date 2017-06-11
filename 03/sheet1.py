import numpy as np
import numpy.testing as npt
from scipy.linalg import expm
import matplotlib.pyplot as plt
import scipy.spatial, scipy.linalg
import scipy.sparse.linalg

class PCA():
    def __init__(self, X):
        # ...
        Xt = X.T
        C = np.cov(Xt)
        D, U = np.linalg.eig(C)
        self.U = U.T
        self.D = D
        
    
    def project(self, X, m):
        # ...
        U = self.U.T
        M = U[:,:m] 
        Z = np.dot(M.T,X.T)
        Z = Z.T 
        return Z
    
    def denoise(self, X, m):
        Z = self.project(X,m)
        U = self.U.T
        Y = np.dot(U[:,:m],Z.T)
        Y = Y + np.mean(X,axis=1).reshape(len(Y),1)
        return Y.T
    
def gammaidx(X, k):
    # ...
    n = len(X)
    
    X2 = (X**2).sum(1) + np.zeros([n,1])
    D = X2 - 2*(X.dot(X.T)) + X2.T
    tmp = np.sort(D,axis=1)
    ktmp = tmp[:,1:1+k]
    ktmp = np.sqrt(ktmp)
    y = np.mean(ktmp,axis=1)
    

    return y

def auc(y_true, y_val, plot=False):
    d = len(y_true)
    one = np.ones((d,1))
    datei = np.array((y_val,y_true))
    num = np.argsort(datei[0])
    dateit = datei.T
    dateit = dateit[num]
    datei = dateit.T
    datei[1][np.where(datei[1]==-1)] = 0
    tsum = np.dot(datei[1],one)
    fsum = d - tsum
    roc = np.empty((2,d+1))
    for i in range(d+1):
        count = one
        count[np.arange(0,i)] = 0
        tpr = np.dot(datei[1],count)/tsum
        fpr = (d-i-np.dot(datei[1],count))/fsum
        roc[0][i]=fpr
        roc[1][i]=tpr
    numroc = np.argsort(roc[0])
    roct = roc.T
    roct = roct[numroc]
    roc = roct.T
    
    for i in roc[0]:
        tmp = roc[1][np.where(roc[0]==i)]
        tmp = np.sort(tmp)
        roc[1][np.where(roc[0]==i)] = tmp
            
    c = 0.
    for j in range(d):
        c = c + (abs(roc[0][j]-roc[0][j+1]))*roc[1][j]
        c = c + (abs(roc[1][j+1]-roc[1][j]))*(abs(roc[0][j]-roc[0][j+1]))/2


    if(plot==True):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(roc[0],roc[1])
        ax.set_title('ROC curve')
        ax.set_xlabel('false postive rate')
        ax.set_ylabel('true postive rate')
        fig.show()

    return c

def lle(X, m, tol, n_rule='knn', k=5, epsilon=1.):
    data = X
    N = len(data)
    W = np.zeros([N,N]) # matrix for storing reconstruction weights
    M = np.zeros([N,N]) # matrix M of which eigenvectors are computed
    E = np.zeros([N,2]) # eigenvectors of M forming the embedding

    # compute distance matrix 
    X2 = (data**2).sum(1) + np.zeros([N,1])
    D = X2 - 2*(data.dot(data.T)) + X2.T
    #D = np.sqrt(D)
    if(n_rule=='knn'):

        # Iterate over all data points to find their associated reconstruction weights
        for i in range(N):
            x = data[i]
            mins = np.argsort(D[i])[1:k+1]
            X = data[mins]
            X = x - X
            C_inv = np.linalg.inv((X).dot(X.T) + tol*np.eye(k))
            W[i,mins] = C_inv.sum(1)/C_inv.sum()

        IM = np.eye(N) - W
        M = IM.T.dot(IM)
        vals, vecs = scipy.sparse.linalg.eigsh(M, k=m+1, sigma=0)
        E = vecs[:,-m:]
        
    if(n_rule=='eps-ball'):
        for i in range(N):
            x = data[i] 
            mins = np.argsort(D[i])[1:len(D[i])]
            srtd = D[i][mins]
            X = data[mins]
            dd = np.sum(srtd<epsilon**2)
            #dd = len(srtd[np.where(srtd<epsilon**2)])
            
            if(dd==0):
                raise ValueError("One of X has no neighbars")
            X = X[0:dd]
            C_inv = np.linalg.inv((X).dot(X.T) + tol*np.eye(len(X)))
            W[i,mins[0:dd]] = C_inv.sum(1)/C_inv.sum()
            
            
        IM = np.eye(N) - W
        M = IM.T.dot(IM)
        vals, vecs = scipy.sparse.linalg.eigsh(M, k=m+1, sigma=0)
        E = vecs[:,-m:]
    return E