#This script generates the plots in Figure 2: Dynamics on a two-dimensional type space with four types

import sys,os
#lib_path = os.path.abspath(os.path.join('..')) #specifying path for player module
sys.path.append('coevo/')
from player import LiteralPlayer, GriceanPlayer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
import datetime



def normalize(m):
    """returns row-normalized matrix m"""
    return m / m.sum(axis=1)[:, np.newaxis]

def get_u(types):
    """returns typeXtype utility-matrix"""
    out = np.zeros([len(types), len(types)]) #matrix to store results in
    for i in xrange(len(types)):
        for j in xrange(len(types)):
            out[i,j] = (np.sum(types[i].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5 +\
                       (np.sum(types[j].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5
    return out

def obs_counts(obs):
    """helper function: recodes observations as counts to pass to Q-matrix computation"""
    out = [] #matrix to store results in
    for i in xrange(len(obs)):
        out.append([obs[i].count(j) for j in xrange(4)])
    return out

def get_likelihood(obs,likelihoods):
    """returns lhs(obs | type)"""
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in xrange(len(likelihoods)):
        for o in xrange(len(obs)):
            flat_lhi = likelihoods[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))]) 
    return out

def get_q(lexica_prior,learning_parameter,k,types):
    """compute mutation matrix Q based on lexica_prior and k-length observations, with a learning_parameter gamma, for all types"""
    likelihoods = [i.sender_matrix for i in types] #production probabilities for types: pr(d | t)
    atomic_obs = [0,1,2,3] #0,1 upper-row; 2,3 are lower row
    D = list(product(atomic_obs,repeat=k))  #generate observations
    D = obs_counts(D) #recode observations as counts
    
    lhs = get_likelihood(D,likelihoods) #get likelihood of data given types
    post = normalize(lexica_prior * np.transpose(lhs)) #compute posterior
    parametrized_post = normalize(post**learning_parameter) #parametrize posterior by learning_parameter gamma
    return np.dot(normalize(lhs),parametrized_post)


def learning_prior(types,c):
    """An illustrative prior that favors simpler lexica over more complex ones"""
    out = np.zeros(len(types)) # matrix to store results in
    for i in xrange(len(types)):
        lx = types[i].lexicon
        if np.sum(lx) == 2.: #lexicalized upper-bound
            out[i] = 1
        else:
            out[i] = 1*c
    return out / np.sum(out)

def rmd(p,u,q,types):
    """The replicator-mutator dynamic for population p, utility matrix u and mutation matrix q"""
    pPrime = p * [np.sum(u[t,] * p)  for t in xrange(len(types))] #replicator D
    pPrime = pPrime / np.sum(pPrime) 
    return np.dot(pPrime, q) #mutator D

def rep(p,u,types):
    """The replicator dynamic for population p and utility matrix u"""
    pPrime = p * [np.sum(u[t,] * p) for t in xrange(len(types))]
    return pPrime / np.sum(pPrime)

def mut(p,q):
    """The mutator dynamic for population p and muation matrix q"""
    return np.dot(p,q) 


def quiver_filled(lam,bias_para,post_para,k):
    
    lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
    types = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
    lexica_prior = learning_prior(types,bias_para)
    Q = get_q(lexica_prior,post_para,k,types)
    U = get_u(types)
    
    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)
    
    ax_n = [[0,0],[0,1],[1,0],[1,1]]
    
    x = np.linspace(0,1,20)
    y = np.linspace(0,1,20)
    
    for i in xrange(len(ax_n)-1):
        u,v = np.zeros((len(x),len(y))), np.zeros((len(x),len(y)))
        for tx in x:
            for ty in y:
                p = np.zeros(len(types))
                p[0] = (1-tx) * (1-ty)
                p[1] = (1-tx) * ty
                p[2] = tx * (1-ty)
                p[3] = tx * ty
                if i == 0:
                    r = rep(p,U,types)
                    lbl = 'RD'
                elif i == 1:
                    r = mut(p,Q)
                    lbl = 'MD'
                else:
                    r = rmd(p,U,Q,types)
                    lbl = 'RMD'
        
                change_along_x =  r[2] + r[3] - tx #prag types
                change_along_y =  r[1] + r[3] - ty #lack types
                #u.append(change_along_x)
                #v.append(change_along_y)
                txidx = np.argwhere(x == tx)[0][0]
                tyidx = np.argwhere(y == ty)[0][0]
                u[tyidx,txidx] = change_along_x
                v[tyidx,txidx] = change_along_y
        
        if i == 0:
            axs[ax_n[i][0],ax_n[i][1]].quiver(x,y,u,v)#,scale=1./2.)
            axs[ax_n[i][0],ax_n[i][1]].set_xlim(-0.1,1.1)
            axs[ax_n[i][0],ax_n[i][1]].set_ylim(-0.1,1.1)
            axs[ax_n[i][0],ax_n[i][1]].annotate('Lit.', xy=(-0.05,-0.1), xycoords='data')
            axs[ax_n[i][0],ax_n[i][1]].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].annotate(lbl, xy=(0.45,1.1), xycoords='data',fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].axis('off')
    
    
        else:
            axs[ax_n[i][0],ax_n[i][1]].quiver(x,y,u,v)
            axs[ax_n[i][0],ax_n[i][1]].set_xlim(-0.1,1.1)
            axs[ax_n[i][0],ax_n[i][1]].set_ylim(-0.1,1.1)
            axs[ax_n[i][0],ax_n[i][1]].annotate('Lit.', xy=(-0.05,-0.1), xycoords='data')
            axs[ax_n[i][0],ax_n[i][1]].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].annotate(lbl, xy=(0.45,1.1), xycoords='data',fontsize=12)
            axs[ax_n[i][0],ax_n[i][1]].axis('off')
    
    axs[1,1].annotate(r'$\lambda$ = %d, bias = %.2f, l = %d, k = %d' % (lam,bias_para,post_para,k), xy=(0.0,0.5), xycoords='data', fontsize=10)
    
    axs[1,1].axis('off')
    
    print U
    print '#'
    print Q
    plt.tight_layout()
    plt.show()

#quiver_filled(5,1.05,15,3)


def quiver_contour(lam_lst,bias_para,post_para_lst,k,cont):
    print '#Drawing quiver contour, ', datetime.datetime.now()
    lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
    fig,axs = plt.subplots(ncols=3,nrows=2,sharex=True,sharey=True)
    ax_n = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
    #for quiver
    xq = np.linspace(0,1,8)
    yq = np.linspace(0,1,8)
    #for contour
    xc = np.linspace(0,1,250)
    yc = np.linspace(0,1,250)
    X,Y = np.meshgrid(xc,yc)

    for i in xrange(len(ax_n)):
        print '#Computing parameter configuration ', i, ' out of ', len(ax_n), ' at ', datetime.datetime.now()

        print('')
        Z = np.zeros((len(xc),len(yc)))
        u,v = np.zeros((len(xq),len(yq))),np.zeros((len(xq),len(yq)))

        lam = lam_lst[i]
        post_para = post_para_lst[i]
        types = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
        lexica_prior = learning_prior(types,bias_para)
        Q = get_q(lexica_prior,post_para,k,types)
        U = get_u(types)
        
        if cont:
            for tx in xc:
                for ty in yc:
                    p = np.zeros(len(types))
                    p[0] = (1-tx) * (1-ty)
                    p[1] = (1-tx) * ty
                    p[2] = tx * (1-ty)
                    p[3] = tx * ty
                    if i in [0,1,2]:
                        r = rep(p,U,types)
                    else: 
                        r = mut(p,Q)
                    txidx = np.argwhere(xc == tx)[0][0]
                    tyidx = np.argwhere(yc == ty)[0][0]
                    Z[tyidx,txidx] = r[3] - p[3]
        
        for tx in xq:
            for ty in yq:
                p = np.zeros(len(types))
                p[0] = (1-tx) * (1-ty)
                p[1] = (1-tx) * ty
                p[2] = tx * (1-ty)
                p[3] = tx * ty
                if i in [0,1,2]:
                    r = rep(p,U,types)
                else:
                    r = mut(p,Q)
                change_along_x = r[2] + r[3] - tx
                change_along_y = r[1] + r[3] - ty
                txidx = np.argwhere(xq == tx)[0][0]
                tyidx = np.argwhere(yq == ty)[0][0]
                u[tyidx,txidx] = change_along_x
                v[tyidx,txidx] = change_along_y

        if cont:
            axx = axs[ax_n[i][0],ax_n[i][1]].contourf(X,Y,Z,cmap='plasma',alpha=0.5,vmin=-1,vmax=1.)
        axs[ax_n[i][0],ax_n[i][1]].quiver(xq,yq,u,v, angles='xy', scale_units='xy')
        axs[ax_n[i][0],ax_n[i][1]].set_xlim(-0.1,1.1)
        axs[ax_n[i][0],ax_n[i][1]].set_ylim(-0.1,1.1)
        axs[ax_n[i][0],ax_n[i][1]].annotate('Lit.', xy=(-0.1,-0.1), xycoords='data')
        axs[ax_n[i][0],ax_n[i][1]].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
        axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
        axs[ax_n[i][0],ax_n[i][1]].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
       
        if i in [0,1,2]:
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$\lambda = %d$' % (lam), xy=(0.42,1.05), xycoords='data',fontsize=10)
        else:
            axs[ax_n[i][0],ax_n[i][1]].annotate(r'$\lambda = %d, l = %d$' % (lam,post_para), xy=(0.27,1.05), xycoords='data',fontsize=10)

        axs[ax_n[i][0],ax_n[i][1]].axis('off')
    fig.text(0.06, 0.69, '(a)', va='center', fontsize=17)#, rotation='vertical')
    fig.text(0.06, 0.27, '(b)', va='center', fontsize=17)#, rotation='vertical')
    if cont:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85,0.15,0.0275,0.7])
        fig.colorbar(axx,cax=cbar_ax)
    else:
        plt.tight_layout() #doesn't work well with subplots colorbar
    plt.show()


lam_lst = [1,5,20,1,5,20] #list of lambda values to consider
post_para_lst = [1,10,20,1,10,20] #list of gammas to consider 

quiver_contour(lam_lst,1.05,post_para_lst,5,cont=True) #If True then contour, otherwise no contour background

