import sys,os
lib_path = os.path.abspath(os.path.join('..')) #specifying path for player module
sys.path.append(lib_path) #specifying path for player module
from player import LiteralPlayer, GriceanPlayer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

#lam = 20
#bias_para = 2
#k = 5
#post_para = 10



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
    
#    out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q
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






def coords_x1(edge,pops):
    u = [] #x-tip (amount of prag)
    v = [] #y-tip (amount of amb)
    for i in xrange(len(pops)):
        u.append( (pops[i][2] + pops[i][3]) - edge[i])
        v.append(pops[i][1] + pops[i][3])
    return [u,v]

def coords_x2(edge,pops):
    u = [] #x-tip (amount of prag)
    v = [] #y-tip (amount of amb)
    for i in xrange(len(pops)):
        u.append( (pops[i][2] + pops[i][3]) - edge[i])
        v.append( (pops[i][1] + pops[i][3]) - 1)
    return [u,v]

def coords_y1(edge,pops):
    u = [] #x-tip (amount of prag)
    v = [] #y-tip (amount of amb)
    for i in xrange(len(pops)):
        u.append( (pops[i][2] + pops[i][3]))
        v.append( (pops[i][1] + pops[i][3]) - edge[i])
    return [u,v]

def coords_y2(edge,pops):
    u = [] #x-tip (amount of prag)
    v = [] #y-tip (amount of amb)
    for i in xrange(len(pops)):
        u.append( (pops[i][2] + pops[i][3]) - 1)
        v.append( (pops[i][1] + pops[i][3]) - edge[i])
    return [u,v]


#def quiver_filled(lam,bias_para,post_para,k):
lam = 20
bias_para = 2
post_para = 10
k = 8

lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
types = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
lexica_prior = learning_prior(types,bias_para)
Q = get_q(lexica_prior,post_para,k,types)
U = get_u(types)

fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)

ax_n = [[0,0],[0,1],[1,0],[1,1]]

x = np.linspace(0,1,20)
y = np.linspace(0,1,20)
#(X,Y) = np.meshgrid(x,y)
#u,v = [], []

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
                rmd(p,U,Q,types)
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

plt.savefig('quiver-lam%d-c%d-k%d-l%d.png' %(lam,bias_para*100,k,post_para))

plt.tight_layout()
plt.show()



sys.exit()


def quiver_plot(lam,bias_para,post_para,k):
    lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
    types = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
    lexica_prior = learning_prior(types,bias_para)
    q = get_q(lexica_prior,post_para,k,types)
    u = get_u(types)

    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)
    
    edge_x1 = np.linspace(0,1,22)
    edge_x2 = np.linspace(0,1,22)
    edge_y1 = np.linspace(0,1,22)
    edge_y2 = np.linspace(0,1,22)
    
    rep_x1,rep_x2,rep_y1,rep_y2 = [], [],[],[]
    mut_x1,mut_x2,mut_y1,mut_y2 = [], [],[],[]
    rmd_x1,rmd_x2,rmd_y1,rmd_y2 = [], [],[],[]
    
    for i in xrange(len(edge_x1)):
        pop_x1 = np.zeros(4)
        pop_x1[2], pop_x1[0] = edge_x1[i], 1-edge_x1[i]
        rep_x1.append(rep(pop_x1,u,types))
        mut_x1.append(mut(pop_x1,q))
        rmd_x1.append(rmd(pop_x1,u,q,types))
        rep_tip, mut_tip, rmd_tip = coords_x1(edge_x1,rep_x1), coords_x1(edge_x1,mut_x1), coords_x1(edge_x1,rmd_x1)
        rep_x1_u, rep_x1_v = rep_tip[0], rep_tip[1]
        mut_x1_u, mut_x1_v = mut_tip[0], mut_tip[1]
        rmd_x1_u, rmd_x1_v = rmd_tip[0], rmd_tip[1]
        
        pop_x2 = np.zeros(4)
        pop_x2[3], pop_x2[1] = edge_x1[i], 1-edge_x1[i]
        rep_x2.append(rep(pop_x2,u,types))
        mut_x2.append(mut(pop_x2,q))
        rmd_x2.append(rmd(pop_x2,u,q,types))
        rep_tip, mut_tip, rmd_tip = coords_x2(edge_x2,rep_x2), coords_x2(edge_x2,mut_x2), coords_x2(edge_x2,rmd_x2)
        rep_x2_u, rep_x2_v = rep_tip[0], rep_tip[1]
        mut_x2_u, mut_x2_v = mut_tip[0], mut_tip[1]
        rmd_x2_u, rmd_x2_v = rmd_tip[0], rmd_tip[1]
    
        
        pop_y1 = np.zeros(4)
        pop_y1[1], pop_y1[0] = edge_x1[i], 1-edge_x1[i]
        rep_y1.append(rep(pop_y1,u,types))
        mut_y1.append(mut(pop_y1,q))
        rmd_y1.append(rmd(pop_y1,u,q,types))
        rep_tip, mut_tip, rmd_tip = coords_y1(edge_y1,rep_y1), coords_y1(edge_y1,mut_y1), coords_y1(edge_y1,rmd_y1)
        rep_y1_u, rep_y1_v = rep_tip[0], rep_tip[1]
        mut_y1_u, mut_y1_v = mut_tip[0], mut_tip[1]
        rmd_y1_u, rmd_y1_v = rmd_tip[0], rmd_tip[1]
    
        
        pop_y2 = np.zeros(4)
        pop_y2[3], pop_y2[2] = edge_x1[i], 1-edge_x1[i]
        rep_y2.append(rep(pop_y2,u,types))
        mut_y2.append(mut(pop_y2,q))
        rmd_y2.append(rmd(pop_y2,u,q,types))
        rep_tip, mut_tip, rmd_tip = coords_y2(edge_y2,rep_y2), coords_y2(edge_y2,mut_y2), coords_y2(edge_y2,rmd_y2)
        rep_y2_u, rep_y2_v = rep_tip[0], rep_tip[1]
        mut_y2_u, mut_y2_v = mut_tip[0], mut_tip[1]
        rmd_y2_u, rmd_y2_v = rmd_tip[0], rmd_tip[1]
    
    scaling = 1
    axs[0,0].quiver(edge_x1,np.zeros(len(edge_x1)), rep_x1_u, rep_x1_v , scale=1./100)#, units='xy', angles='xy', scale=1/ scaling)
    axs[0,0].quiver(edge_x2,np.ones(len(edge_x2)),  rep_x2_u, rep_x2_v )#, units='xy', angles='xy', scale=1/  scaling)
    axs[0,0].quiver(np.zeros(len(edge_y1)),edge_y1, rep_y1_u, rep_y1_v )#, units='xy', angles='xy', scale=1/ scaling)
    axs[0,0].quiver(np.ones(len(edge_y2)),edge_y2,  rep_y2_u, rep_y2_v )#, units='xy', angles='xy', scale=1/  scaling)
    axs[0,0].set_xlim(-0.1,1.1)
    axs[0,0].set_ylim(-0.1,1.1)
    axs[0,0].annotate('Lit.', xy=(-0.05,-0.1), xycoords='data')
    axs[0,0].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
    axs[0,0].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
    axs[0,0].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
    axs[0,0].annotate('RD', xy=(0.45,0.5), xycoords='data',fontsize=12)

    axs[0,0].axis('off')

    
    scaling=0.2
    
    axs[0,1].quiver(edge_x1,np.zeros(len(edge_x1)), mut_x1_u, mut_x1_v )#, units='xy', angles='xy', scale=1/scaling)
    axs[0,1].quiver(edge_x2,np.ones(len(edge_x2)), mut_x2_u, mut_x2_v  )#, units='xy', angles='xy', scale=1/ scaling)
    axs[0,1].quiver(np.zeros(len(edge_y1)),edge_y1, mut_y1_u, mut_y1_v )#, units='xy', angles='xy', scale=1/scaling)
    axs[0,1].quiver(np.ones(len(edge_y2)),edge_y2, mut_y2_u, mut_y2_v  )#, units='xy', angles='xy', scale=1/ scaling)
    axs[0,1].set_xlim(-0.1,1.1)
    axs[0,1].set_ylim(-0.1,1.1)
    axs[0,1].annotate('Lit.', xy=(-0.05,-0.1), xycoords='data')
    axs[0,1].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
    axs[0,1].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
    axs[0,1].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
    axs[0,1].annotate('M', xy=(0.45,0.5), xycoords='data',fontsize=12)

    axs[0,1].axis('off')

    
    
    
    axs[1,0].quiver(edge_x1,np.zeros(len(edge_x1)), rmd_x1_u, rmd_x1_v )#, units='xy', angles='xy', scale=1/scaling)
    axs[1,0].quiver(edge_x2,np.ones(len(edge_x2)), rmd_x2_u, rmd_x2_v  )#, units='xy', angles='xy', scale=1/ scaling)
    axs[1,0].quiver(np.zeros(len(edge_y1)),edge_y1, rmd_y1_u, rmd_y1_v )#, units='xy', angles='xy', scale=1/scaling)
    axs[1,0].quiver(np.ones(len(edge_y2)),edge_y2, rmd_y2_u, rmd_y2_v  )#, units='xy', angles='xy', scale=1/ scaling)
    axs[1,0].set_xlim(-0.1,1.1)
    axs[1,0].set_ylim(-0.1,1.1)
    axs[1,0].annotate('Lit.', xy=(-0.05,-0.1), xycoords='data')
    axs[1,0].annotate('Prag.', xy=(0.95,-0.1), xycoords='data')
    axs[1,0].annotate(r'$L_{b}$', xy=(-0.1,0.05), xycoords='data',fontsize=12)
    axs[1,0].annotate(r'$L_{l}$', xy=(-0.1,0.95), xycoords='data', fontsize=12)
    axs[1,0].annotate('RMD', xy=(0.45,0.5), xycoords='data',fontsize=12)
    axs[1,0].axis('off')

    
    axs[1,1].annotate(r'$\lambda$ = %d, bias = %d, l = %d, k = %d' % (lam,bias_para,post_para,k), xy=(0.0,0.5), xycoords='data', fontsize=12)

    axs[1,1].axis('off')
    
    plt.savefig('quiver-lam%d-c%d-k%d-l%d.png' %(lam,bias_para,k,post_para))

    plt.tight_layout()
    plt.show()

quiver_plot(20,3,10,5)





##############

#COLORED CMAP FOR QUIVERS ACCORDING TO INTENSITY
#import matplotlib
#norm = matplotlib.colors.Normalize()
##norm.autoscale(v/np.sum(v))
#cm = matplotlib.cm.cool #colormap name
#sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
#sm.set_array([])

#plt.quiver(x,y,u,u,angles='xy',scale_units='xy',scale=1,color=cm(norm(v))) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.

#plt.quiver(x,y,line,line,angles='xy',scale_units='xy',scale=1) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.

#plt.colorbar(sm)

















sys.exit()
##################
from scipy import optimize
def find_rep(x_prop):
    return abs(replicator_step(x_prop) - x_prop)

def find_mut(x_prop):
    return abs(mutator_step(x_prop) - x_prop)

def find_rmd(x_prop):
    return abs(mutator_step(replicator_step(x_prop)) - x_prop)

#
#
x0 = 0.2
##res1 = optimize.fmin_cg(replicator_step,x0)
#res1 = optimize.minimize(find_rep,x0,bounds=((0.,1.),))
#res2 = optimize.minimize(find_mut,x0,bounds=((0.,1.),))
#res3 = optimize.minimize(find_rmd,x0,bounds=((0.,1.),))

minimizer_kwargs = dict(method='L-BFGS-B',bounds=((0.,1.),)) #, 
##
#def print_fun(x,f,accepted):
#    print "at minima %.4f accepted %d" % (f, int(accepted))
#
#def mybounds(**kwargs):
#    x = kwargs["x_new"]
#    tmax = bool(np.all(x <= 1.0))
#    tmin = bool(np.all(x >= 0.0))
#    return tmax and tmin
#
res1 = optimize.basinhopping(find_rep,x0,minimizer_kwargs = minimizer_kwargs)#,callback=print_fun)
#
print res1.fun
#
res2 = optimize.basinhopping(find_mut,x0,minimizer_kwargs = minimizer_kwargs)
print res2.fun #needs to be 0, to access best guess: res2.x
#
res3 = optimize.basinhopping(find_rmd,x0, minimizer_kwargs = minimizer_kwargs)

print res3.fun
#
#

