import sys,os
lib_path = os.path.abspath(os.path.join('..')) #specifying path for player module
sys.path.append(lib_path) #specifying path for player module
from player import LiteralPlayer, GriceanPlayer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
#import seaborn as sns

lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]


def get_u(types):
    out = np.zeros([len(types), len(types)])
    for i in xrange(len(types)):
        for j in xrange(len(types)):
            out[i,j] = (np.sum(types[i].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5 +\
                       (np.sum(types[j].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5
    return out

def fitness(x_prop,U):
    return np.sum((np.array([x_prop, 1-x_prop]) * U)[0])

def overall_fitness(x_prop,U):
    return (x_prop * np.sum((np.array([x_prop,1-x_prop]) * U)[0])) +\
           ((1.-x_prop) * np.sum((np.array([x_prop,1-x_prop])* U)[1]))

def replicator_step(x_prop,U):
    return (x_prop * fitness(x_prop,U)) / overall_fitness(x_prop,U)

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


def get_q(types,lexica_prior,learning_parameter,k):
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


def mutator_step(x_prop,q):
    return x_prop * q[0,0] + (1-x_prop) * q[1,0]

def get_theta0(q): #find x for which m(x) = x
    return 1- (q[0,1] / (q[0,1] + q[1,0]))


#Now plot direction of change and stationary points

#typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
#types = [typeList[x] for x in [2,3]]
#u = get_u(types)
#lexica_prior = learning_prior(types,2)
#q = get_q(types,lexica_prior,5,5)
#
#theta0 = theta0(q)
#
#print theta0
#print mutator_step(theta0,q)
#print mutator_step(mutator_step(theta0,q),q)
#
#sys.exit()


def characterize(X,XPrime):
    dir_left = []
    dir_right = []
    attr = []
    no_attr = []

    for i in xrange(len(X)):
        if i < (len(X)-1) and i > 0: #if there is a change in directionality, there may be a rest point
            if X[i] < XPrime[i] and X[i+1] > XPrime[i+1]:
                attr.append(X[i]) #designate former point as attractor (difference is negligeble)
        if X[i] == XPrime[i]: #if D(x) = x
            if i == 0:
                if X[i+1] <= XPrime[i+1]:
                    no_attr.append(X[i])
                else:
                    attr.append(X[i])
            elif 0 < i < (len(X)-1):
                if X[i-1] < XPrime[i-1] and X[i+1] > XPrime[i+1]:
                    attr.append(X[i])
                else:
                    no_attr.append(X[i])
            else:
                if X[i-1] < XPrime[i-1]:
                    attr.append(X[i])
                else:
                    no_attr.append(X[i])
        #Add directionality toward stationary if no change in directionality detected
    if len(dir_right) == len(dir_left) == 0 and len(attr) == len(no_attr) == 1:
        min_v = min({attr[0],no_attr[0]})
        max_v = max({attr[0],no_attr[0]})
        h = (max_v - min_v) / 2.
        hh = h/2.
        if max_v in attr and min_v in no_attr:
            dir_right.append(h-hh)
            dir_right.append(h+hh)
        elif min_v in attr and max_v in no_attr: 
            dir_left.append(h-hh)
            dir_left.append(h+hh)
    if len(attr) == 1 and len(no_attr) == 0:
        v = attr[0]
        if v / 2. > 0:
            dir_right.append(v / 2)
        if 1 - v > 0:
            dir_left.append(1- (1 - v) / 2)
    return [dir_left,dir_right,attr,no_attr]





def get_labels(type_indices):
    labels = []
    for i in type_indices:
        if i == 0:
            b = 'lit.'
            l = 'bound'
        elif i == 1:
            b = 'lit.'
            l = 'lack'
        elif i == 2:
            b = 'prag.'
            l = 'bound'
        else:
            b = 'prag.'
            l = 'lack'
        labels.append(b)
        labels.append(l)
    return labels


def plot_phases(lam,bias_para,post_para,k):
    typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)

    cAttr = 'black'
    cNoAttr = 'indianred'
    aAttr = 1.0
    aNoAttr = 1.0

    plt_order = [[0,1],  #litbound vs. litlack
                 [2,3],  #pragbound vs. praglack
                 [0,2],  #litbound vs. pragbound
                 [1,3]] #litlack vs. praglack
    yax = [0,0,1,1]
    xax = [0,1,0,1]


    for tt in xrange(len(plt_order)):
        X = np.linspace(0,1,1000)
        types = [typeList[x] for x in plt_order[tt]]
        u = get_u(types)
        lexica_prior = learning_prior(types,bias_para)
        q = get_q(types,lexica_prior,post_para,k)
        theta0 = get_theta0(q)
        print theta0
        X[np.abs(X-theta0).argmin()] = theta0 #smuggle in theta0 for mutator stat
        print np.argwhere(X == theta0)
        rep = np.array([replicator_step(prop,u) for prop in X])
        mut = np.array([mutator_step(prop,q) for prop in X])
        rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])

        r_chara = characterize(X,rep)
        m_chara = characterize(X,mut)
        rmd_chara = characterize(X,rmd)

        rep_left,rep_right, rep_attr, rep_no_attr = r_chara[0], r_chara[1], r_chara[2], r_chara[3]
        mut_left,mut_right, mut_attr, mut_no_attr = m_chara[0], m_chara[1], m_chara[2], m_chara[3]
        rmd_left,rmd_right, rmd_attr, rmd_no_attr = rmd_chara[0], rmd_chara[1], rmd_chara[2], rmd_chara[3]

        axs[yax[tt],xax[tt]].plot(X,np.ones(len(X)), color='black')
        axs[yax[tt],xax[tt]].plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
        axs[yax[tt],xax[tt]].plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
        axs[yax[tt],xax[tt]].scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color=cNoAttr, alpha=aNoAttr)
        axs[yax[tt],xax[tt]].scatter(rep_attr,np.ones(len(rep_attr)),s=80,color=cAttr, alpha=aAttr)
        axs[yax[tt],xax[tt]].annotate('RD', xy=(0.5,1.1), xycoords='data')
        
        axs[yax[tt],xax[tt]].plot(X,np.zeros(len(X)), color='black')
        axs[yax[tt],xax[tt]].plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
        axs[yax[tt],xax[tt]].plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
        axs[yax[tt],xax[tt]].scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color=cAttr, alpha=aAttr)
        axs[yax[tt],xax[tt]].scatter(mut_no_attr,np.zeros(len(mut_no_attr)),s=80,color=cNoAttr, alpha=aNoAttr)
        axs[yax[tt],xax[tt]].annotate('M', xy=(0.5,0.1), xycoords='data')
        
        axs[yax[tt],xax[tt]].plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
        axs[yax[tt],xax[tt]].plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
        axs[yax[tt],xax[tt]].plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
        axs[yax[tt],xax[tt]].scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color=cAttr, alpha=aAttr)
        axs[yax[tt],xax[tt]].scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color=cNoAttr, alpha=aNoAttr)
        axs[yax[tt],xax[tt]].annotate('RMD', xy=(0.5,-0.9), xycoords='data')

        labels = get_labels(plt_order[tt])
        axs[yax[tt],xax[tt]].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10,y=1.05)
        axs[yax[tt],xax[tt]].axis('off')
    
    fig.suptitle(r'($\lambda$ = %d, bias = %.2f, l = %d, k = %d)' % (lam,bias_para,post_para,k))#, y=1.29)
    plt.tight_layout()
    plt.show()

#plot_phases(1,1.05,1,8)
plot_phases(20,1.05,1,8)









########################## Snippets
#def get_points_of_change_heuristic(X,d_diff,u):
#    no_change_at = np.where(d_diff==0)[0] #stationary points
#    attractor = []
#    no_attractor = []
#    for i in xrange(len(no_change_at)):
#        if X[no_change_at[i]] == 0:
#            if X[1] > u[1]:
#                attractor.append(X[no_change_at[i]])
#            else:
#                no_attractor.append(X[no_change_at[i]])
#        elif X[no_change_at[i]] == X[-1]:
#            if X[-2] < u[-2]:
#                attractor.append(X[no_change_at[i]])
#            else:
#                no_attractor.append(X[no_change_at[i]])
#        else:
#            if X[no_change_at[i]-1] < u[no_change_at[i]-1] and X[no_change_at[i]+1] < u[no_change_at[i]+1]:
#                attractor.append(X[no_change_at[i]])
#            else:
#                no_attractor.append(X[no_change_at[i]])
#    return [attractor,no_attractor]
#
#def get_directionality_heuristic(X,d_diff,attractor,no_attractor):
#    no_change_at = np.where(d_diff==0)[0] #stationary points
#    to_compare = list(combinations([X[i] for i in no_change_at],2)) #pairwise groupings
#    markers_left = []
#    markers_right = []
#    if not(len(to_compare) == 0):
#         for i in to_compare:
#            h = (max(i) - min(i)) / 2.
#            hh = h/2.
#            if max(i) in attractor and min(i) in no_attractor:
#                markers_right.append(h-hh)
#                markers_right.append(h+hh)
#            elif min(i) in attractor and max(i) in no_attractor:
#                markers_left.append(h-hh)
#                markers_left.append(h+hh)
#    else:
#        steps = len(d_diff) * 1/8.
#        for i in xrange(0,8+1):
#            idx = int(i * steps-1)
#            if not(idx > 0):
#                idx = 0
#            if d_diff[idx] > 0:
#                markers_right.append(X[idx])
#            else:
#                markers_left.append(X[idx])
#    return [markers_left,markers_right]



#def plot_phase_portraits(lam,bias_para,post_para,k):
#    typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
#
#    X = np.linspace(0,1,1000)
#    X[np.where(X[np.abs(X-theta0).argmin()])[0][0]] = theta0 #smuggle in theta0 for mutator stat
#
#    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)
#
####litbound vs. litlack###
#    types = [typeList[x] for x in [0,1]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    rep_diff = rep - X
#    line = np.zeros(len(X)) #no change in x
#    
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    mut_diff = mut - X
#    
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#    rmd_diff = rmd - X
#
#    rep_stat = get_points_of_change_heuristic(X,rep_diff,rep)
#    rep_attr,rep_no_attr = rep_stat[0],rep_stat[1]
#    rep_dir = get_directionality_heuristic(X,rep_diff,rep_attr,rep_no_attr)
#    rep_left,rep_right = rep_dir[0],rep_dir[1]
#    
#    
#    mut_stat = get_points_of_change_heuristic(X,mut_diff,mut)
#    mut_attr,mut_no_attr = mut_stat[0],mut_stat[1]
#    mut_dir = get_directionality_heuristic(X,mut_diff,mut_attr,mut_no_attr)
#    mut_left,mut_right = mut_dir[0],mut_dir[1]
#    
#    rmd_stat = get_points_of_change_heuristic(X,rmd_diff,rmd)
#    rmd_attr,rmd_no_attr = rmd_stat[0],rmd_stat[1]
#    rmd_dir = get_directionality_heuristic(X,rmd_diff,rmd_attr,rmd_no_attr)
#    rmd_left,rmd_right = rmd_dir[0],rmd_dir[1]
#
#    axs[0,0].plot(X,np.ones(len(X)), color='black')
#    axs[0,0].plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
#    axs[0,0].plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
#    axs[0,0].scatter(rep_attr,np.ones(len(rep_attr)),s=80,color='black')
#    axs[0,0].scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color='white',edgecolor='black')
#    axs[0,0].annotate('RD', xy=(0.5,1.1), xycoords='data')
#    
#    axs[0,0].plot(X,np.zeros(len(X)), color='black')
#    axs[0,0].plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
#    axs[0,0].plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
#    axs[0,0].scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color='black')
#    axs[0,0].scatter(mut_no_attr,np.zeros(len(mut_attr)),s=80,color='white',edgecolor='white')
#    axs[0,0].annotate('M', xy=(0.5,0.1), xycoords='data')
#    
#    axs[0,0].plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
#    axs[0,0].plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
#    axs[0,0].plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
#    axs[0,0].scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color='black')
#    axs[0,0].scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color='white',edgecolor='white')
#    axs[0,0].annotate('RMD', xy=(0.5,-0.9), xycoords='data')
#    labels = get_labels([0,1])
#    axs[0,0].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10,y=1.10)
#    axs[0,0].axis('off')
#
#
####pragbound vs. praglack###
#    types = [typeList[x] for x in [2,3]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    rep_diff = rep - X
#    line = np.zeros(len(X)) #no change in x
#    
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    mut_diff = mut - X
#    
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#    rmd_diff = rmd - X
#
#    rep_stat = get_points_of_change_heuristic(X,rep_diff,rep)
#    rep_attr,rep_no_attr = rep_stat[0],rep_stat[1]
#    rep_dir = get_directionality_heuristic(X,rep_diff,rep_attr,rep_no_attr)
#    rep_left,rep_right = rep_dir[0],rep_dir[1]
#    
#    
#    mut_stat = get_points_of_change_heuristic(X,mut_diff,mut)
#    mut_attr,mut_no_attr = mut_stat[0],mut_stat[1]
#    mut_dir = get_directionality_heuristic(X,mut_diff,mut_attr,mut_no_attr)
#    mut_left,mut_right = mut_dir[0],mut_dir[1]
#    
#    rmd_stat = get_points_of_change_heuristic(X,rmd_diff,rmd)
#    rmd_attr,rmd_no_attr = rmd_stat[0],rmd_stat[1]
#    rmd_dir = get_directionality_heuristic(X,rmd_diff,rmd_attr,rmd_no_attr)
#    rmd_left,rmd_right = rmd_dir[0],rmd_dir[1]
#
#    axs[0,1].plot(X,np.ones(len(X)), color='black')
#    axs[0,1].plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
#    axs[0,1].plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
#    axs[0,1].scatter(rep_attr,np.ones(len(rep_attr)),s=80,color='black')
#    axs[0,1].scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color='white',edgecolor='black')
#    axs[0,1].annotate('RD', xy=(0.5,1.1), xycoords='data')
#    
#    axs[0,1].plot(X,np.zeros(len(X)), color='black')
#    axs[0,1].plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
#    axs[0,1].plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
#    axs[0,1].scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color='black')
#    axs[0,1].scatter(mut_no_attr,np.zeros(len(mut_attr)),s=80,color='white',edgecolor='white')
#    axs[0,1].annotate('M', xy=(0.5,0.1), xycoords='data')
#    
#    axs[0,1].plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
#    axs[0,1].plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
#    axs[0,1].plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
#    axs[0,1].scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color='black')
#    axs[0,1].scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color='white',edgecolor='white')
#    axs[0,1].annotate('RMD', xy=(0.5,-0.9), xycoords='data')
#    labels = get_labels([2,3])
#    axs[0,1].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10,y=1.10)
#    axs[0,1].axis('off')
#
#
####litbound vs. pragbound###
#    types = [typeList[x] for x in [0,2]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    rep_diff = rep - X
#    line = np.zeros(len(X)) #no change in x
#    
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    mut_diff = mut - X
#    
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#    rmd_diff = rmd - X
#
#    rep_stat = get_points_of_change_heuristic(X,rep_diff,rep)
#    rep_attr,rep_no_attr = rep_stat[0],rep_stat[1]
#    rep_dir = get_directionality_heuristic(X,rep_diff,rep_attr,rep_no_attr)
#    rep_left,rep_right = rep_dir[0],rep_dir[1]
#    
#    
#    mut_stat = get_points_of_change_heuristic(X,mut_diff,mut)
#    mut_attr,mut_no_attr = mut_stat[0],mut_stat[1]
#    mut_dir = get_directionality_heuristic(X,mut_diff,mut_attr,mut_no_attr)
#    mut_left,mut_right = mut_dir[0],mut_dir[1]
#    
#    rmd_stat = get_points_of_change_heuristic(X,rmd_diff,rmd)
#    rmd_attr,rmd_no_attr = rmd_stat[0],rmd_stat[1]
#    rmd_dir = get_directionality_heuristic(X,rmd_diff,rmd_attr,rmd_no_attr)
#    rmd_left,rmd_right = rmd_dir[0],rmd_dir[1]
#
#    axs[1,0].plot(X,np.ones(len(X)), color='black')
##    axs[1,0].plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
##    axs[1,0].plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
#    axs[1,0].scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color=c_attr,alpha=alpha_noAttr)
#    axs[1,0].scatter(rep_attr,np.ones(len(rep_attr)),s=80,color=c_noAttr,alpha=alpha_attr)
#
#    axs[1,0].annotate('RD', xy=(0.5,1.1), xycoords='data')
#    
#    axs[1,0].plot(X,np.zeros(len(X)), color='black')
#    axs[1,0].plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
#    axs[1,0].plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
#    axs[1,0].scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color='black')
#    axs[1,0].scatter(mut_no_attr,np.zeros(len(mut_attr)),s=80,color='white',edgecolor='white')
#    axs[1,0].annotate('M', xy=(0.5,0.1), xycoords='data')
#    
#    axs[1,0].plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
#    axs[1,0].plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
#    axs[1,0].plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
#    axs[1,0].scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color='black')
#    axs[1,0].scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color='white',edgecolor='white')
#    axs[1,0].annotate('RMD', xy=(0.5,-0.9), xycoords='data')
#
#    labels = get_labels([0,2])
#    axs[1,0].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10,y=1.10)
#    axs[1,0].axis('off')
#
#
#
####litlack vs. praglack###
#    types = [typeList[x] for x in [1,3]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    rep_diff = rep - X
#    line = np.zeros(len(X)) #no change in x
#    
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    mut_diff = mut - X
#    
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#    rmd_diff = rmd - X
#
#    rep_stat = get_points_of_change_heuristic(X,rep_diff,rep)
#    rep_attr,rep_no_attr = rep_stat[0],rep_stat[1]
#    rep_dir = get_directionality_heuristic(X,rep_diff,rep_attr,rep_no_attr)
#    rep_left,rep_right = rep_dir[0],rep_dir[1]
#    
#    
#    mut_stat = get_points_of_change_heuristic(X,mut_diff,mut)
#    mut_attr,mut_no_attr = mut_stat[0],mut_stat[1]
#    mut_dir = get_directionality_heuristic(X,mut_diff,mut_attr,mut_no_attr)
#    mut_left,mut_right = mut_dir[0],mut_dir[1]
#    
#    rmd_stat = get_points_of_change_heuristic(X,rmd_diff,rmd)
#    rmd_attr,rmd_no_attr = rmd_stat[0],rmd_stat[1]
#    rmd_dir = get_directionality_heuristic(X,rmd_diff,rmd_attr,rmd_no_attr)
#    rmd_left,rmd_right = rmd_dir[0],rmd_dir[1]
#
#    axs[1,1].plot(X,np.ones(len(X)), color='black')
#    axs[1,1].plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
#    axs[1,1].plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
#    axs[1,1].scatter(rep_attr,np.ones(len(rep_attr)),s=80,color='black')
#    axs[1,1].scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color='white',edgecolor='black')
#    axs[1,1].annotate('RD', xy=(0.5,1.1), xycoords='data')
#    
#    axs[1,1].plot(X,np.zeros(len(X)), color='black')
#    axs[1,1].plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
#    axs[1,1].plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
#    axs[1,1].scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color='black')
#    axs[1,1].scatter(mut_no_attr,np.zeros(len(mut_attr)),s=80,color='white',edgecolor='white')
#    axs[1,1].annotate('M', xy=(0.5,0.1), xycoords='data')
#    
#    axs[1,1].plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
#    axs[1,1].plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
#    axs[1,1].plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
#    axs[1,1].scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color='black')
#    axs[1,1].scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color='white',edgecolor='white')
#    axs[1,1].annotate('RMD', xy=(0.5,-0.9), xycoords='data')
#
#    labels = get_labels([1,3])
#    axs[1,1].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10,y=1.10)
#    axs[1,1].axis('off')
#
#
#    fig.suptitle(r'($\lambda$ = %d, bias = %d, l = %d, k = %d)' % (lam,bias_para,post_para,k))#, y=1.29)
#    plt.tight_layout()
#    plt.show()
#
#
#plot_phase_portraits(20,2,10,5)
#
#
#sys.exit()
#def simple_plot(lam,bias_para,post_para,k):
#    typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
#
#    X = np.linspace(0,1,1000)
#
#    fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)#,sharey=True,sharex=True)
#
####litbound vs. litlack###
#    types = [typeList[x] for x in [0,1]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#
#    axs[0,0].plot(X,rep, color='indianred',linestyle=':')
#    axs[0,0].plot(X,mut,color='blue', linestyle='--')
#    axs[0,0].plot(X,rmd, color='purple')
#    labels = get_labels([0,1])
#    axs[0,0].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10)
#    axs[0,0].set_ylim(0,1)
#    axs[0,0].set_xlim(0,1)
#    axs[0,0].legend(('RD', 'M', 'RMD'),frameon=True)
#
#    types = [typeList[x] for x in [2,3]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#
#    axs[0,1].plot(X,rep, color='indianred',linestyle=':')
#    axs[0,1].plot(X,mut,color='blue', linestyle='--')
#    axs[0,1].plot(X,rmd, color='purple')
#    labels = get_labels([2,3])
#    axs[0,1].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10)
#    axs[0,1].set_ylim(0,1)
#    axs[0,1].set_xlim(0,1)
#    axs[0,1].legend(('RD', 'M', 'RMD'),frameon=True)
#
#    types = [typeList[x] for x in [0,2]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#
#    axs[1,0].plot(X,rep, color='indianred',linestyle=':')
#    axs[1,0].plot(X,mut,color='blue', linestyle='--')
#    axs[1,0].plot(X,rmd, color='purple')
#    labels = get_labels([0,2])
#    axs[1,0].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10)
#    axs[1,0].set_ylim(0,1)
#    axs[1,0].set_xlim(0,1)
#    axs[1,0].legend(('RD', 'M', 'RMD'),frameon=True)
#
#    types = [typeList[x] for x in [1,3]]
#    u = get_u(types)
#    lexica_prior = learning_prior(types,bias_para)
#    q = get_q(types,lexica_prior,post_para,k)
#
#    rep = np.array([replicator_step(prop,u) for prop in X])
#    mut = np.array([mutator_step(prop,q) for prop in X])
#    rmd = np.array([mutator_step(replicator_step(prop,u),q) for prop in X])
#
#    axs[1,1].plot(X,rep, color='indianred',linestyle=':')
#    axs[1,1].plot(X,mut,color='blue', linestyle='--')
#    axs[1,1].plot(X,rmd, color='purple')
#    labels = get_labels([1,3])
#    axs[1,1].set_title(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=10)
#    axs[1,1].set_ylim(0,1)
#    axs[1,1].set_xlim(0,1)
#    axs[1,1].legend(('RD', 'M', 'RMD'),frameon=True)
#
#    fig.suptitle(r'($\lambda$ = %d, bias = %d, l = %d, k = %d)' % (lam,bias_para,post_para,k))
#    plt.tight_layout()
#    plt.show()
#
#simple_plot(1,2,1,5)
#
#
#        
#       
#
#
#
#sys.exit()
#
#
#
#
#
#
#plt.suptitle(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=14)
#plt.title(r'($\lambda$ = %d, bias = %d, l = %d, k = %d)' % (lam,bias_para,post_para,k), y=1.29)
#
#plt.axis('off')
#plt.tight_layout()
##plt.show()
#plt.savefig('%d%d-lam%d-c%d-k%d-l%d.png' %(type_indices[0],type_indices[1],lam,bias_para,k,post_para))
#
#
##plt.plot(x,np.zeros(len(x)),color='black', marker='o',markevery=[int(attr) for attr in no_attractor],fillstyle='full',markerfacecolor='white',markeredgecolor='black',markersize=8)
##plt.plot(markers_left,np.zeros(len(markers_left)), marker='3',color='black',markersize=10)
##plt.plot(markers_right,np.zeros(len(markers_right)), marker='4',color='black',markersize=10)
##plt.scatter(attractor,np.zeros(len(attractor)),s=80,color='black')
###plt.scatter(no_attractor,np.zeros(len(no_attractor)), s=80, facecolors='white',edgecolors='black',color='white')
##plt.axis('off')
##plt.tight_layout()
##plt.show()
#
#
#
###############
#
##COLORED CMAP FOR QUIVERS ACCORDING TO INTENSITY
##import matplotlib
##norm = matplotlib.colors.Normalize()
###norm.autoscale(v/np.sum(v))
##cm = matplotlib.cm.cool #colormap name
##sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
##sm.set_array([])
#
##plt.quiver(x,y,u,u,angles='xy',scale_units='xy',scale=1,color=cm(norm(v))) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.
#
##plt.quiver(x,y,line,line,angles='xy',scale_units='xy',scale=1) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.
#
##plt.colorbar(sm)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#sys.exit()
###################
#from scipy import optimize
#def find_rep(x_prop):
#    return abs(replicator_step(x_prop) - x_prop)
#
#def find_mut(x_prop):
#    return abs(mutator_step(x_prop) - x_prop)
#
#def find_rmd(x_prop):
#    return abs(mutator_step(replicator_step(x_prop)) - x_prop)
#
##
##
#x0 = 0.2
###res1 = optimize.fmin_cg(replicator_step,x0)
##res1 = optimize.minimize(find_rep,x0,bounds=((0.,1.),))
##res2 = optimize.minimize(find_mut,x0,bounds=((0.,1.),))
##res3 = optimize.minimize(find_rmd,x0,bounds=((0.,1.),))
#
#minimizer_kwargs = dict(method='L-BFGS-B',bounds=((0.,1.),)) #, 
###
##def print_fun(x,f,accepted):
##    print "at minima %.4f accepted %d" % (f, int(accepted))
##
##def mybounds(**kwargs):
##    x = kwargs["x_new"]
##    tmax = bool(np.all(x <= 1.0))
##    tmin = bool(np.all(x >= 0.0))
##    return tmax and tmin
##
#res1 = optimize.basinhopping(find_rep,x0,minimizer_kwargs = minimizer_kwargs)#,callback=print_fun)
##
#print res1.fun
##
#res2 = optimize.basinhopping(find_mut,x0,minimizer_kwargs = minimizer_kwargs)
#print res2.fun #needs to be 0, to access best guess: res2.x
##
#res3 = optimize.basinhopping(find_rmd,x0, minimizer_kwargs = minimizer_kwargs)
#
#print res3.fun
##
##
#
