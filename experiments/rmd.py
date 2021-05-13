## Functions for replicator-mutator dynamics with iterated learning as mutator dynamics
import sys 
import numpy as np
sys.path.append('coevo/')
from random import sample
from itertools import product
from player import LiteralPlayer,GriceanPlayer
from lexica import get_lexica,get_prior,get_lexica_bins

import datetime
import csv
import os.path

def normalize(m):
    """returns row-normalized matrix m"""
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst,states,messages):
    """summarize counts for tuples of k-states and k-messages""" 
    counter = [0 for _ in xrange(states*messages)]
    for i in xrange(len(lst)):
        s,m = lst[i][0] *messages, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(s_amount,m_amount,k,likelihoods,state_freq,sample_amount):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k"""
    if k == 1 and s_amount == 3 and m_amount == 3: #If k = 1 then no sampling, compute Q with full set of possible observations
        all_data = [[0,0,0,0,0,0,0,0,1],\
                [0,0,0,0,0,0,0,1,0],\
                [0,0,0,0,0,0,1,0,0],\
                [0,0,0,0,0,1,0,0,0],\
                [0,0,0,0,1,0,0,0,0],\
                [0,0,0,1,0,0,0,0,0],\
                [0,0,1,0,0,0,0,0,0],\
                [0,1,0,0,0,0,0,0,0],\
                [1,0,0,0,0,0,0,0,0]]
        return [all_data for _ in xrange(432)] #all data for each parent type possible
    else:
        s = list(xrange(s_amount))
        m = list(xrange(m_amount))
        atomic_observations = list(product(s,m))
   
        obs = [] #store all produced k-length (s,m) sequences 
        for t in xrange(len(likelihoods)):
            produced_obs = [] #store k-length (s,m) sequences of a type
            production_vector = likelihoods[t].flatten()
            doubled_state_freq = np.column_stack(tuple(state_freq for _ in xrange(m_amount))).flatten() # P(s)
            sample_vector = production_vector * doubled_state_freq #P(s) * P(m|s,t_i)
            for i in xrange(sample_amount):
                sample_t = [np.random.choice(len(atomic_observations),p=sample_vector) for _ in xrange(k)]
                sampled_obs = [atomic_observations[i] for i in sample_t]
                produced_obs.append(summarize_counts(sampled_obs,s_amount,m_amount))
            obs.append(produced_obs)
        return obs

def get_likelihood(obs,likelihoods):
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            flat_lhi = likelihoods[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))]) 
    return out

def get_mutation_matrix(s_amount,m_amount,state_freq, likelihoods,lexica_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity):
    """Either load or compute mutation matrix"""
    if os.path.isfile('coevo/matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(s_amount,m_amount,lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity))):
        print '#Loading mutation matrix, ', datetime.datetime.now()
        return np.genfromtxt('coevo/matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(s_amount,m_amount,lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)), delimiter=',')
    else:
        print '#Computing mutation matrix, ', datetime.datetime.now()
        obs = get_obs(s_amount,m_amount,k,likelihoods,state_freq,sample_amount) #get production data from all types
        out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q
    
        for parent_type in xrange(len(likelihoods)):
            type_obs = obs[parent_type] #Parent production data
            lhs = get_likelihood(type_obs,likelihoods) #P(parent data|t_i) for all types
            post = normalize(lexica_prior * np.transpose(lhs)) #P(t_j|parent data) for all types; P(d|t_j)P(t_j)
            parametrized_post = normalize(post**learning_parameter)
            normed_lhs = lhs[parent_type] / np.sum(lhs[parent_type]) #norm P(parent_data|parent_type)
            out[parent_type] = np.dot(np.transpose(normed_lhs),parametrized_post)
    
        q = out
        f_q = csv.writer(open('coevo/matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(s_amount,m_amount,lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)),'wb'))
        for i in q:
            f_q.writerow(i)
        return q


def get_utils(typeList,states,messages,lam,alpha,mutual_exclusivity):
    """Either load or compute utility matrix"""
    if os.path.isfile('coevo/matrices/umatrix-s%d-m%d-lam%d-a%d-me%s.csv' %(states,messages,lam,alpha,str(mutual_exclusivity))):
        print '#Loading utilities, ', datetime.datetime.now()
        return np.genfromtxt('coevo/matrices/umatrix-s%d-m%d-lam%d-a%d-me%s.csv' %(states,messages,lam,alpha,str(mutual_exclusivity)),delimiter=',')
    else:
        print '#Computing utilities, ', datetime.datetime.now()
        out = np.zeros([len(typeList), len(typeList)])
        for i in range(len(typeList)):
            for j in range(len(typeList)):
                out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) /3. + \
                         np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))/3. ) / 2

        f_u = csv.writer(open('coevo/matrices/umatrix-s%d-m%d-lam%d-a%d-me%s.csv' %(states,messages,lam,alpha,str(mutual_exclusivity)),'wb'))
        for i in out:
            f_u.writerow(i)
    

        return out

def get_type_bin(type_idx,bins):
    """Helper function that categorizes types into bins according to whether they have the same (permuted) lexica"""
    for b in xrange(len(bins)):
        if type_idx in bins[b]:
            return b
    

def run_dynamics(alpha,lam,k,sample_amount,gens,runs,states,messages,learning_parameter,kind,mutual_exclusivity):
    """Runs experiments"""
    state_freq = np.ones(states) / float(states) #frequency of states s_1,...,s_n 

    print '#Starting, ', datetime.datetime.now()
    
    lexica = get_lexica(states,messages,mutual_exclusivity)
    bins = get_lexica_bins(lexica) #To bin types with identical lexica
    l_prior = get_prior(lexica)
    typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(alpha,lam,lex) for lex in lexica]
    
    likelihoods = np.array([t.sender_matrix for t in typeList])
    
    u = get_utils(typeList,states,messages,lam,alpha,mutual_exclusivity)
    q = get_mutation_matrix(states, messages, state_freq, likelihoods,l_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity)

    
    print '#Beginning multiple runs, ', datetime.datetime.now()
    f = csv.writer(open('coevo/results/%s-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' %(kind,states,messages,lam,alpha,k,sample_amount,learning_parameter,gens,str(mutual_exclusivity)),'wb'))
    f.writerow(['runID','kind']+['t_ini'+str(x) for x in xrange(len(typeList))] +\
               ['lam', 'alpha','k','samples','l','gens', 'm_excl'] + ['t_final'+str(x) for x in xrange(len(typeList))])
    
    if os.path.isfile('coevo/results/00mean-%s-s%d-m%d-g%d-r%d-me%s.csv' %(kind,states,messages,gens,runs,str(mutual_exclusivity))):
        f_mean = csv.writer(open('coevo/results/00mean-%s-s%d-m%d-g%d-r%d-me%s.csv' %(kind,states,messages,gens,runs,str(mutual_exclusivity)), 'a'))
    else: 
        f_mean = csv.writer(open('coevo/results/00mean-%s-s%d-m%d-g%d-r%d-me%s.csv' %(kind,states,messages,gens,runs,str(mutual_exclusivity)), 'wb'))
        f_mean.writerow(['kind','lam','alpha','k','samples','l','gens','runs','m_excl'] + ['t_mean'+str(x) for x in xrange(len(typeList))])
    
    p_sum = np.zeros(len(typeList)) #vector to store mean across runs

    for i in xrange(runs):
        p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
        p_initial = p
        for r in range(gens):
            if kind == 'rmd':
                pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
                pPrime = pPrime / np.sum(pPrime)
                p = np.dot(pPrime, q)
            elif kind == 'm':
                p = np.dot(p,q)
            elif kind == 'r':
                pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
                p = pPrime / np.sum(pPrime)
    
        f.writerow([str(i),kind] + [str(p_initial[x]) for x in xrange(len(typeList))]+\
                   [str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(mutual_exclusivity)] +\
                   [str(p[x]) for x in xrange(len(typeList))])
        p_sum += p
    p_mean = p_sum / runs
    f_mean.writerow([kind,str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(runs),str(mutual_exclusivity)] +\
                        [str(p_mean[x]) for x in xrange(len(typeList))])
    
    inc = np.argmax(p_mean)
    inc_bin = get_type_bin(inc,bins)

    print 
    print '##### Mean results#####'
    print '### Parameters: ###'
    print 'dynamics= %s, alpha = %d, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, generations = %d, runs = %d' % (kind, alpha, lam, k, sample_amount, learning_parameter, gens, runs)
    print '#######################'
    print 
    print 'Incumbent type:', inc, ' with proportion ', p_mean[inc]
    if mutual_exclusivity:
        print 'Target type (t24) proportion: ', p_mean[24]
    else:
        print 'The bin of the incumbent', inc_bin
        print 'Bin #', bins[inc_bin]
    print '#######################'
    print 