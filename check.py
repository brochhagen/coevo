#Load Q
import numpy as np
from lexica import get_lexica,get_prior,get_lexica_bins
import pandas as pd

m_amount,s_amount = 3,3
lam = 20
alpha = 1
k = 1
sample_amount = 250
learning_parameter = 1
mutual_exclusivity=False

print '#Loading mutation matrix, '
q = np.genfromtxt('./matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(s_amount,m_amount,lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)), delimiter=',')

print np.sum(q)


lexica = get_lexica(3,3, False)
#bins = get_lexica_bins(lexica) #To bin types with identical lexica
l_prior = get_prior(lexica)


#dm = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('m',lam,k,learning_parameter))
#drm = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',lam,k,learning_parameter))
#
#restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns
#dm = dm[restrict_to_final]
#drm = drm[restrict_to_final]
#
#dm = dm.iloc[5]
#drm = drm.iloc[5]
#
#
#print dm[0],dm[216]
#print drm[0],drm[216]

from discreteMarkovChain import markovChain
mc = markovChain(q)
mc.computePi('linear')
print mc.pi[:5] #first five values of stationary distribution of q
print l_prior[:5] #first five values of actual prior



##Appendix C
q = np.array([[.56,.22,.22],\
              [.22,.59,.19],\
              [.22,.19,.59]])

mc = markovChain(q)
mc.computePi('linear')
print mc.pi #first five values of stationary distribution of q


q = np.array([[.8,.1,.1],\
              [.2,.7,.1],\
              [.2,.1,.7]])

mc = markovChain(q)
mc.computePi('linear')
print mc.pi #first five values of stationary distribution of q


#####
#We get the same results as before, roughly, if we norm lhs[parent] only. We need to check for Fig. 6 only now.
#We have normed Q if we norm lhs[parent] only
#We get the prior back under any lambda if k = 1 with normed lhs[parent] in 50 gens. Do we get it with 10000 for k=5? (It's not there for 5gens and I think its due to sampling) No, only for lam = 1, which we also do for 50 gens

#After checking Fig. 6 replication:
#1. Write MF (i) no norm(Q) needed, (ii) norm around lhs[parent] but not lhs, (iii) in any case, same results, (iv) we don't get the prior back for l=1 because of sampling in Q. This is more apparent when comparing targets,competitors, and Lalls when lam is high. We do get it if k=1 (already at 50... and getting closer the more gens we run).
#2. Clean code for plots and runs in new folder
#3. Run everything again with 1000, use same Qs but delete results just in case
