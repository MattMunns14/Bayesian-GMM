import numpy as np 
import random as rand
from scipy.special import digamma 
import math

data = np.genfromtxt('data2.txt', delimiter=',')


def Variational_GMM(data, k, n_iter):


    dimX = 1
    Z = np.array([np.random.dirichlet(np.full(k,10)) for _ in range(len(data))])
    variationalInference( data, k, n_iter, dimX, Z)

def variationalInference( data, k, n_iter, dimX, Z):
    #Initializations and priors
    beta_0 = 1
    v_0 = 1
    mu_0 = .1
    w_0  = 10
    a_0 = 10
    nk, bk, mk, wk, vk, xk, sk = mStep(Z, v_0, mu_0, w_0, beta_0, k,data)

    count = 0
    while count < n_iter:
        mu, phi, rho, Z = eStep(a_0, nk, bk, mk, wk, vk, xk, sk, k, data)    
        nk, bk, mk, wk, vk, xk, sk = mStep(Z, v_0, mu_0, w_0, beta_0, k,data)
        count+=1
    print(mk)
    print(sk)
    print(nk)
def mStep( Z, v_0, mu_0, w_0, beta_0, k, data):
    nk = calcNk(Z,k)
    bk = calcBk(nk,k, beta_0)
    xk = calcXk(nk,Z, k, data)
    mk = calcMk(bk, beta_0, mu_0, nk, xk, k)
    vk = calcVk(v_0, nk, k)
    sk = calcSk(nk, Z, xk, data, k)
    wk = calcWk(w_0, nk, sk, beta_0, xk, mu_0,vk)

    return nk, bk, mk, wk, vk, xk, sk

def eStep(a_0, nk, bk, mk, wk, vk, xk, sk, k, data):
    rho = calcRho(a_0, nk, k)
    mu = calcMu(bk, vk, mk, wk, data, k)
    phi = calcPhi(vk, wk, k)
    Z = calcZ(mu,phi, rho, data, k, bk, vk,mk,wk)
    return mu,phi, rho, Z

def calcZ( mu, phi, rho, data, k, bk, vk,mk,wk):

    Z = np.zeros((len(data),k))
    for n in range(len(data)):
        for i in range(k):
            Z[n,i] = rho[i]*phi[i]**(1/2)*math.exp(-1/(2*bk[i])-vk[i]/2*(data[n]-mk[i])*wk[i]*(data[n]-mk[i]))
    row_sums = Z.sum(axis=1)
    Z_norm = Z/row_sums[:,np.newaxis]
    return Z_norm

def calcRho(a_0, nk, k):
    ak = [nk[i]+a_0 for i in range(k)]
    rho = [digamma(ak[i])-digamma(sum(ak)) for i in range(k)]
    rho = [math.exp(i) for i in rho]
    return rho

def calcPhi(vk, wk,k):
    phi = [digamma(vk[i]/2)+np.log(2)+np.log(wk[i]) for i in range(k)]
    phi = [math.exp(i) for i in phi]
    return phi

def calcMu(bk, vk, mk, wk, data, k):
    mu = np.zeros((len(data),k))
    for n in range(len(data)):
        for i in range(k):
            mu[n,i]=1/bk[i]+vk[i]*(data[n]-mk[i])*wk[i]*(data[n]-mk[i])
    return mu

def calcWk(w_0, nk, sk, beta_0, xk, mu_0,vk):

    invWk = [1/w_0+nk[i]*sk[i]+(beta_0*nk[i])/(beta_0+nk[i])*(xk[i]-mu_0)**2 for i in range(k)]
    Wk = [(1/invWk[i]) for i in range(k)]

    return Wk


def calcSk(nk, Z, xk, data, k):
    sk = []
    for i in range(k):
        sumK = 0
        for n in range(len(data)):
            diffMat = data[n]-xk[i]
            sumK+=Z[n,i]*(diffMat)**2 
        sk.append(1/nk[i]*sumK)
    return sk

def calcVk(v_0, nk, k):
    vk = [v_0 + nk[i]+1 for i in range(k)]
    return vk

def calcMk(bk, beta_0, m_0, nk, xk, k):
    mk = [1/bk[i]*(beta_0*m_0+nk[i]*xk[i]) for i in range(k)]
    return mk

def calcXk(nk, Z, k, data):
    xk = []
    for j in range(k):
        sumk = 0 
        for i in range(len(data)):
            sumk+=data[i]*Z[i,j]
        sumk = sumk*1/nk[j]
        xk.append(sumk)
    return xk

def calcBk(nk, k, beta_0):
    bk = [beta_0 + nk[i] for i in range(k)]
    return bk

def calcNk(Z, k):
    nk = [np.sum(Z[:,i],axis=0)for i in range(k)]
    return nk
            
        
k=2
iterations = 1000

var_GMM=Variational_GMM(data, k, iterations)
