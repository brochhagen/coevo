#This convenience script sequentially launches experiments across lambda-gamma parameter configurations
import sys
sys.path.append('coevo/experiments')
from rmd import run_dynamics

lams = [x for x in xrange(1,21)] #lambda values for parameter sweep 
ells = [x for x in xrange(1,16)] #posterior parameters for parameter sweep

for lam in lams:
    for l in ells:
        run_dynamics(1,lam,5,250,50,1000,3,3,l,'rmd',False)
