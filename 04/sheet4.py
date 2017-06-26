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
import itertools as it
import time
#%matplotlib inline

class svm_smo(): 
    
    def __init__(self, kernel,C):
        self.kernel = kernel
        self.c = C

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
    
    
    def fx(self,X1,X2,Y):
        K=self.getkernel(X1,X2)
        alpY = (Y*self.alpha).reshape(len(self.alpha),1)
        return np.dot(K,alpY)-self.b#np.sign(np.dot(K,alpY)+self.b)
    
    def _compute_box_constraints(self, i, j, Y, alpha, C):
        
        if(Y[i]==Y[j]):
            L = np.max([0,alpha[i]+alpha[j]-C])
            H = np.min([C,alpha[i]+alpha[j]])
        else:
            L = np.max([0,alpha[j]-alpha[i]])
            H = np.min([C,C+alpha[j]-alpha[i]])
        return L, H 
    
    
    def _update_parameters(self, E_i, E_j, i, j, K, Y, alpha, b, C):
        L, H = self._compute_box_constraints( i, j, Y, alpha, C)
        if(L==H):
            return alpha, b, 0
        kappa = 2*K[i,j] -K[i,i] -K[j,j]
        if(kappa>=0):
            return alpha, b, 0
        
        aph2_new= alpha[j]-Y[j]*(E_i-E_j)/kappa
        
        if(aph2_new>H):
            aph2_new=H
        if(aph2_new<L):
            aph2_new=L
           
        aph1_new = alpha[i]+Y[i]*Y[j]*(alpha[j]-aph2_new)
        if(np.abs(alpha[j]-aph2_new)<0.0005):
            return alpha, b, 0
        
        alpha_new = np.array(alpha)
        alpha_new[i]= aph1_new
        alpha_new[j]= aph2_new
        
        new_b = self._compute_updated_b( E_i, E_j, i, j, K, Y, alpha, alpha_new, b, C)
            
        return alpha_new, new_b, 1
    
    
    def _compute_updated_b(self, E_i, E_j, i, j, K, Y, alpha_old, alpha_new, b_old, C):
        b1 = b_old+E_i+Y[i]*(alpha_new[i]-alpha_old[i])*K[i,i]+Y[j]*(alpha_new[j]-alpha_old[j])*K[i,j]
        b2 = b_old+E_j+Y[i]*(alpha_new[i]-alpha_old[i])*K[i,j]+Y[j]*(alpha_new[j]-alpha_old[j])*K[j,j]
        new_b=(b1+b2)/2
        
        if((alpha_new[i]>0)and(alpha_new[i]<C)):
            new_b = b1
        if((alpha_new[j]>0)and(alpha_new[j]<C)):
            new_b = b2
        
        return new_b
     
        
    def fit(self, X, Y):
        #self.X_fit = X
        N= len(X)
        rang=np.arange(N)
        K = self.getkernel(X)
        self.alpha = np.zeros(N)
        self.b = 0
        P=1000
        tol = 0.03
        p=0
        
        while(p<P):
            a=0
            for i in range(N):
                Ei = self.fx(X[i].reshape(1,len(X[i])),X,Y)-Y[i]
                if (((Y[i]*Ei < -tol) and (self.alpha[i] < self.c)) or ( (Y[i]*Ei > tol) and (self.alpha[i] > 0))):
                    j=np.random.choice(np.setdiff1d(rang, np.array([i])),1)[0]
                    Ej = self.fx(X[j].reshape(1,len(X[i])),X,Y)-Y[j]
                    self.alpha, self.b, updated=self._update_parameters( Ei[0,0], Ej[0,0], i, j, K, Y, self.alpha, self.b, self.c)
                    a = a+updated
                
            if(a==0):        
                p=p+1
            else:
                p=0
        
        f=self.fx(X,X,Y)*Y.reshape(len(Y),1)
        self.SV= X[np.where(np.abs(f[:,0]-1)<0.0001)[0]]
        self.y = Y[np.where(np.abs(f[:,0]-1)<0.0001)[0]]
        self.alpha = self.alpha[np.where(np.abs(f[:,0]-1)<0.0001)[0]]
        
    def predict(self, X):
        return np.sign(self.fx(X,self.SV,self.y)[:,0])
    
    
def plot_svm_2d(X, y, model):
    n=len(X)
    plt.figure(figsize = (6, 6))
    plt.plot(X[y ==  1,0], X[y ==  1,1], 'ro')
    plt.plot(X[y == -1,0], X[y == -1,1], 'bo')
    plt.plot(model.SV[:,0],model.SV[:,1],'kx')
    a = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, n)
    b = np.linspace(np.min(X[:,1])-1, np.max(X[:,1])+1, n)
    A, B = np.meshgrid(a,b)
    mesh_z = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            mesh_z[y, x] = model.fx(np.array([a[x], b[y]]).reshape(1,2),C.SV,C.y)
    plt.contour(A, B, mesh_z, 0)