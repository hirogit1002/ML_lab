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

def kmeans(X,k,maxiter=100):
    n=len(X)
    d=len(X[0])
    mu = (np.var(X,axis=0)**(1/2))*np.random.randn(k,d)+ np.mean(X,axis=0)
    ite =0
    X2 = (X**2).sum(1).reshape(n,1)*np.ones((n,k))
    U2 = (mu**2).sum(1).reshape(1,k) * np.ones([n,k])
    D = X2 - 2*(X.dot(mu.T)) + U2
    muidx = np.ones(D.shape)*np.arange(k)
    r = np.argmin(D,axis=1)
    r = r.reshape(len(r),1)
    member = (muidx==r).sum(0)
    members = np.array(member).reshape(1,k)

    while(ite<maxiter):
        idx=((np.ones((n,k))*np.arange(k))==r).astype(np.int64)
        division = (member).reshape(k,1)
        zero = np.where(division==0)[0]
        division[zero]=1
        mu = ((idx.T.reshape(k,n,1)*X).sum(1))/division 
        mu[zero] = (np.var(X,axis=0)**(1/2))*np.random.randn(len(zero),d)+ np.mean(X,axis=0)
        X2 = (X**2).sum(1).reshape(n,1)*np.ones((n,k))
        U2 = (mu**2).sum(1).reshape(1,k) * np.ones([n,k])
        D = X2 - 2*(X.dot(mu.T)) + U2
        rr = np.argmin(D,axis=1).reshape(len(r),1)
        boolvec=(r == rr)
        if (len(boolvec)==sum(boolvec)):
            break
        r = rr  
        member = ((np.ones(D.shape)*np.arange(k))==r).sum(0)
        members = np.append(members,member.reshape(1,k),axis=0)

        ite = ite +1        
    loss = np.sum(np.sort(D,axis=1)[:,0])
    print("The number of iterations performed: ",ite)
    print("The number of cluster memberships: ")
    print("k= ",np.arange(1,k+1))
    print(members)
    print(" The loss function value: ",loss)
    return mu,r.T[0],loss


def kmeans_agglo(X, r):
    n = len(X)
    d = len(X[0])
    mxidx=np.max(r)
    K=len(set(r))
    if(K<2):
        raise AssertionError('Number of cluster shuld be more than 1')
    kmloss = np.zeros(K)
    R = np.zeros((K-1,n)).astype(np.int64)
    R[0] = np.array(r)
    RR = np.zeros((K-1,n)).astype(np.int64)
    RR[0] = np.array(r)
    ite = 1
    checkbox = (np.arange(0,K)*np.ones((n,1))).T
    bb =(R[0]==checkbox).astype(np.float64)
    mmm=np.sum(bb,axis=1).reshape(K,1)
    BB = bb[:,:,np.newaxis]
    C = bb[:,:,np.newaxis]*X
    C = np.sum(C,axis=1)
    C = C/mmm
    clabel = np.arange(K)
    kmloss[0]=np.linalg.norm(BB*C[:,np.newaxis,:]-bb[:,:,np.newaxis]*X,axis=2).sum()
    mergeidx=np.zeros((K-1,2)).astype(np.int64)
    while(K>2):
        seq = np.arange(0,K)
        label=np.array(list(itertools.combinations(seq,2)))
        combo=(C[label[:,0]]+C[label[:,1]])/2
        cmblabel = np.array((clabel[label[:,0]],clabel[label[:,1]])).T
        f = len(cmblabel)#number of combo
        newlab=cmblabel.reshape(len(cmblabel),2,1)
        newlab=np.ones((2,n))*newlab
        inridx = (newlab==R[ite-1]).astype(np.int64)
        inridx = inridx[:,0,:]+inridx[:,1,:]
        inX = inridx[:,:,np.newaxis]*X
        resizC = inridx[:,:,np.newaxis]*combo[:,np.newaxis,:]
        inD = np.linalg.norm(inX-resizC,axis=2).sum(1)
     
        outidx = np.ones(inridx.shape).astype(np.int64) -inridx
        outX = outidx[:,:,np.newaxis]*X
        g= lambda x: np.where(clabel==x)
        adress = np.array(list(map(g,R[ite-1])))
        outC=outidx[:,:,np.newaxis]*C[adress].reshape(n,2)
        outD = np.linalg.norm(outX-outC,axis=2).sum(1)
        D = inD + outD    
     
        minidx=np.argmin(D)
        bestCom = combo[minidx]
        bClabel= cmblabel[minidx]
        kmloss[ite] = D[minidx]
        rr=np.array(R[ite-1])
        rr[np.where(rr==max(bClabel))]=ite+mxidx
        rr[np.where(rr==min(bClabel))]=ite+mxidx
        R[ite]=rr
        C[ np.where(min(bClabel))]=bestCom
        C = np.delete(C, np.where(clabel==max(bClabel)), 0)
        clabel[np.where(clabel==min(bClabel))]=ite+mxidx
        clabel = np.delete(clabel, np.where(clabel==max(bClabel)), 0)
        mergeidx[ite-1][1] =  max(bClabel)
        mergeidx[ite-1][0] =  min(bClabel)  
        ite = ite +1
        K=K-1
    lstmerge=(C[0]+C[1])/2
    kmloss[len(kmloss)-1]=np.linalg.norm(X-lstmerge,axis=1).sum()
    lstlbl=set(R[len(R)-1])
    mergeidx[len(mergeidx)-1][1]=max(lstlbl)
    mergeidx[len(mergeidx)-1][0]=min(lstlbl)
    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    k = len(mergeidx)
    num=np.ones(k*2+1)
    for i in range(k):
        num[i+k+1]=num[mergeidx[i][0]] +num[mergeidx[i][1]]
    for i in range(k+1):
        num = np.delete(num,0)
    num= num.reshape(k,1)
    kmloss2 = np.delete(kmloss,0).reshape(len(kmloss)-1,1)
    ergb=np.append(np.append(mergeidx,kmloss2,1),num,1)
    scipy.cluster.hierarchy.dendrogram(ergb)
    plt.show()
    
    
def norm_pdf(X, mu, C):
    k =mu.shape[0]
    n=len(X)
    d=len(X[0])
    C_inv= np.linalg.solve(C,np.ones((k,1,1))*np.eye(d))
    c1 = C_inv[:,:,0][:,np.newaxis,:]*np.ones((k,n,d))
    c2 = C_inv[:,:,1][:,np.newaxis,:]*np.ones((k,n,d))
    XX=np.ones((k,1,1))*X
    bmu = mu[:,np.newaxis,:]*np.ones((k,n,d))
    XX=XX-bmu
    X1=(XX*c1).sum(2)
    X2=(XX*c2).sum(2)
    X3=np.append(X1[:,:,np.newaxis],X2[:,:,np.newaxis],2)
    y =np.exp(-((XX*X3).sum(2))/2)/((np.sqrt((2*np.pi)**d)*np.sqrt(np.linalg.det(C))).reshape(k,1))
    return y


def em_gmm(X, k, max_iter=100, init_kmeans=False, tol=1e-5):
    n = len(X)
    d = len(X[0])
    pi = np.ones((k,1))/k
    num = np.random.choice(len(X),k,replace=False)
    mu = X[num]
    if(init_kmeans==True):
        mu, r, loss = kmeans(X, k)
    C = np.ones((k,1,1))*np.eye(d)
    ite = 0
    eps = 0.001
    while( ite<max_iter):
        tloglik=np.log(norm_pdf(X,mu,C))
        y = norm_pdf(X,mu,C)
        loggamma = np.log(np.pi*y)-np.log((np.pi*y).sum(0))
        ###
        #loggamma =  logsumexp(loggamma)
        ###
        gamma = np.exp(loggamma)
        Nk = gamma.sum(1)
        pi = Nk/n
        mu =(((X.T*np.ones((k,1,1)))*gamma[:,np.newaxis,:]).sum(2))/Nk.reshape(k,1)
        XX=np.ones((k,1,1))*X
        bmu = mu[:,np.newaxis,:]*np.ones((k,n,d))
        XX=XX-bmu
        S=XX[:,:,:,np.newaxis].transpose(0,1,3,2)*XX[:,:,:,np.newaxis]
        S=S*gamma[:,:,np.newaxis,np.newaxis]
        C = S.sum(1)/Nk[:,np.newaxis,np.newaxis]
        C = C+np.ones((k,1,1))*(tol*np.eye(d))
        loglik=np.log(norm_pdf(X,mu,C))
        error = np.linalg.norm(tloglik -loglik)
        if(error<eps):
            break
        ite = ite+1
    
    print("Number of iterations: ",ite)
    print("Log likelihood ",loglik)
    return pi, mu, C, loglik 

def to_transform(mu, sigma):
    val, vec = np.linalg.eigh(sigma)
    trans = np.diag(np.sqrt(val)).dot(vec)
    return Affine2D.from_values(*trans.flatten(), e=mu[0], f=mu[1])

def plot_gmm_solution(X, mu, sigma):
    plt.clf()
    plt.scatter(X[:,0],X[:,1])
    plt.scatter(mu[:,0],mu[:,1],s=80,c='red',marker='+')
    ax = plt.gca()
    circles = [ plt.Circle( (0, 0), radius=3, transform=to_transform(m, sgm)) for m, sgm in zip(mu, sigma)]
    ax.add_collection(PatchCollection(circles, alpha=0.2))