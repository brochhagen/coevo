import sys,os
lib_path = os.path.abspath(os.path.join('..')) #specifying path for player module
sys.path.append(lib_path) #specifying path for player module
from player import LiteralPlayer, GriceanPlayer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def get_u(types):
    out = np.zeros([len(types), len(types)])
    for i in xrange(len(types)):
        for j in xrange(len(types)):
            out[i,j] = (np.sum(types[i].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5 +\
                       (np.sum(types[j].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5
    return out

def obs_counts(obs):
    out = []
    for i in xrange(len(obs)):
        out.append([obs[i].count(j) for j in xrange(4)])
    return out


def get_likelihood(obs,likelihoods):
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            flat_lhi = likelihoods[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))]) 
    return out


def get_q(lexica_prior,learning_parameter,k,types):
    likelihoods = [i.sender_matrix for i in types]
    atomic_obs = [0,1,2,3] #0,1 upper-row; 2,3 are lower row
    D = list(product(atomic_obs,repeat=k))  
    D = obs_counts(D)
    
    lhs = get_likelihood(D,likelihoods)
    post = normalize(lexica_prior * np.transpose(lhs))
    parametrized_post = normalize(post**learning_parameter)
    return normalize(np.dot(lhs,parametrized_post))


def learning_prior(types,c):
    out = np.zeros(len(types))
    for i in xrange(len(types)):
        lx = types[i].lexicon
        if np.sum(lx) == 2.: #lexicalized upper-bound
            out[i] = 1
        else:
            out[i] = 1*c
    return out / np.sum(out)

def rmd(p,u,q,types):
    pPrime = p * [np.sum(u[t,] * p)  for t in xrange(len(types))]
    pPrime = pPrime / np.sum(pPrime)
    return np.dot(pPrime, q)
#
def rep(p,u,types):
    pPrime = p * [np.sum(u[t,] * p) for t in xrange(len(types))]
    return pPrime / np.sum(pPrime)

def mut(p,q):
    return np.dot(p,q) 

def quiver_filled(lam,bias_para,post_para,k,n):
    
    lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
    types = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
    lexica_prior = learning_prior(types,bias_para)
    Q = get_q(lexica_prior,post_para,k,types)
    U = get_u(types)
    
    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)
    
    ax_n = [[0,0],[0,1],[1,0],[1,1]]
    
    x = np.linspace(0,1,25)
    y = np.linspace(0,1,25)
    #(X,Y) = np.meshgrid(x,y)
    #u,v = [], []
    
    
    al = 1./n*3.
    hw = 5
    
    for i in xrange(len(ax_n)-1):
        n_store = []
        for iters in xrange(n):
            u,v = np.zeros((len(x),len(y))), np.zeros((len(x),len(y)))
            for tx in xrange(len(x)):
                for ty in xrange(len(y)):
                    p = np.zeros(len(types))
                    x_coord = x[tx].tolist() 
                    y_coord = y[ty].tolist()
                    if x_coord == 0.:
                        theta0,theta1 = 1-y_coord, y_coord
                        theta2,theta3 = 0., 0.
                    elif x_coord == 1.:
                        theta0,theta1 = 0.,0.
                        theta2, theta3 = 1-y_coord, y_coord
                    elif y_coord == 0:
                        theta0,theta2 = 1-x_coord,x_coord
                        theta1,theta3 = 0.,0.
                    elif y_coord == 1.:
                        theta1,theta3 = 1-x_coord,x_coord
                        theta0,theta2 = 0.,0.
                    else:
                        theta0 = np.random.uniform(0,min({1-x_coord,1-y_coord})) 
                        theta1 = min({1-x_coord-theta0, y_coord})
                        if theta0 > 0.:
                            theta2 = min({x_coord,1-y_coord-theta0})
                        elif theta1 > 0:
                            theta2 = np.random.uniform(0,min({x_coord,1-y_coord-theta0}))
                        else:
                            theta2 = min({x_coord,1-y_coord-theta0})
                        theta3 = min({x_coord-theta2,y_coord-theta1})
                    p[0] = theta0
                    p[1] = theta1
                    p[2] = theta2
                    p[3] = theta3
        #            print np.isclose(x_coord, p[2]+p[3], rtol=1e-05, atol=1e-08, equal_nan=False)
        #            print np.isclose(y_coord, p[1]+p[3], rtol=1e-05, atol=1e-08, equal_nan=False)
                    p = p / np.sum(p)
                    if i == 0:
                        r = rep(p,U,types)
                        lbl = 'RD'
                    elif i == 1:
                        r = mut(p,Q)
                        lbl = 'MD'
                    else:
                        r = rmd(p,U,Q,types)
                        lbl = 'RMD'
        
                    change_along_x =  r[2] + r[3] - x_coord #prag types
                    change_along_y =  r[1] + r[3] - y_coord #lack types
                    u[ty,tx] = change_along_x
                    v[ty,tx] = change_along_y
            n_store.append([u,v])
        
        for iters in xrange(n):
            u,v = n_store[iters][0],n_store[iters][1]
    #        plt.quiver(x,y,u,v,alpha=1./(n/3.),headwidth=5)#,scale=5/1.)
            axs[ax_n[i][0],ax_n[i][1]].quiver(x,y,u,v,alpha=al,headwidth=hw)
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
               
        
    plt.tight_layout()
    plt.savefig('multi-quiver-lam%d-c%d-k%d-l%d.png' %(lam,bias_para,k,post_para))

    plt.show()

lam = 1
bias_para = 1.05
post_para = 1
k = 8
n = 30

quiver_filled(lam,bias_para,post_para,k,n)

sys.exit()
