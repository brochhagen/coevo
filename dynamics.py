##### Main file to run dynamics
from rmd import run_dynamics

lams = [x for x in xrange(1,21)]
ells = [x for x in xrange(1,16)]

for lam in lams:
    for l in ells:
        run_dynamics(1,lam,5,250,50,1000,3,3,l,'rmd',False)
