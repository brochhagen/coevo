###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys

def get_prior_plot(m_excl):
    from lexica import get_prior, get_lexica

    prior = get_prior(get_lexica(3,3,m_excl))
    
    X = np.arange(len(prior))
    if m_excl:
        target = 24
    else:
        targets = [231,236,291,306,326,336]
       
    Y_target = [0 for _ in xrange(len(prior))]
    for target in targets:
        Y_target[target] = prior[target]
        prior[target] = 0
   
    Y_rest = prior
    
    fig, ax = plt.subplots()
    ax.bar(X,Y_target,width=1, color='red')
    ax.bar(X,Y_rest,width=1)
    plt.show()

#get_prior_plot(False)

def get_binned_prior_plot(m_excl):
    from lexica import get_prior, get_lexica, get_lexica_bins
    lexica = get_lexica(3,3,m_excl)
    prior = get_prior(lexica)
    bins = get_lexica_bins(lexica)

    binned_prior = []
    for b in xrange(len(bins)):
        binned_prior.append(np.sum([prior[x] for x in bins[b]]))

    X = np.arange(len(binned_prior))
    
    if m_excl:
        target = 24
    else:
        target = 68
    Y_target = [0 for _ in xrange(target)] + [binned_prior[target]] + [0 for _ in xrange(len(binned_prior)-target-1)]
    
    binned_prior[target] = 0
    Y_rest = binned_prior
    
    fig, ax = plt.subplots()
    ax.bar(X,Y_target,width=1, color='red')
    ax.bar(X,Y_rest,width=1)
    plt.show()

#get_binned_prior_plot(False)



def get_bins_and_mean(group1,group2,group3,m_excl):
    print 'Loading data'
    df_r = pd.read_csv('./results/r-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group1[0],group1[1],group1[2],group1[3],group1[4],group1[5],str(m_excl)))
    df_m = pd.read_csv('./results/m-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group2[0],group2[1],group2[2],group2[3],group2[4],group2[5],str(m_excl)))
    df_rm = pd.read_csv('./results/rmd-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group3[0],group3[1],group3[2],group3[3],group3[4],group3[5],str(m_excl)))
    
    mean_r = [df_r["t_final"+str(x)].mean() for x in xrange(432)]
    mean_m = [df_m["t_final"+str(x)].mean() for x in xrange(432)]
    mean_rm = [df_rm["t_final"+str(x)].mean() for x in xrange(432)]
    
    from lexica import get_lexica, get_lexica_bins
    lexica = get_lexica(3,3,m_excl)
    bins = get_lexica_bins(lexica)
    
    binned_r,binned_m,binned_rm = [], [],[]
    for b in xrange(len(bins)):
        binned_r.append(np.sum([mean_r[x] for x in bins[b]]))
        binned_m.append(np.sum([mean_m[x] for x in bins[b]]))
        binned_rm.append(np.sum([mean_rm[x] for x in bins[b]]))
    
    r_mean_sorted = sorted(mean_r,reverse=True)
    r_bin_sorted = sorted(binned_r,reverse=True)
    r_mean_top = r_mean_sorted[:3]
    r_bin_top = r_bin_sorted[:3]
    
    m_mean_sorted = sorted(mean_m,reverse=True)
    m_bin_sorted = sorted(binned_m,reverse=True)
    m_mean_top = m_mean_sorted[:3]
    m_bin_top = m_bin_sorted[:3]
    
    rm_mean_sorted = sorted(mean_rm,reverse=True)
    rm_bin_sorted = sorted(binned_rm,reverse=True)
    rm_mean_top = rm_mean_sorted[:3]
    rm_bin_top = rm_bin_sorted[:3]
    
    
    idx_mean_r,idx_mean_m,idx_mean_rm = [], [], []
    idx_bin_r,idx_bin_m,idx_bin_rm = [], [], []
    
    
    for i in r_mean_top:
        index = mean_r.index(i)
        idx_mean_r.append(index)
    
    for i in r_bin_top:
        index = binned_r.index(i)
        idx_bin_r.append(index)
    
    for i in m_mean_top:
        index = mean_m.index(i)
        idx_mean_m.append(index)
    
    for i in m_bin_top:
        index = binned_m.index(i)
        idx_bin_m.append(index)
    
    for i in rm_mean_top:
        index = mean_rm.index(i)
        idx_mean_rm.append(index)
    
    for i in rm_bin_top:
        index = binned_rm.index(i)
        idx_bin_rm.append(index)
    
    
    X = np.arange(11)
    empty = [0 for _ in xrange(1)]
    
    Y_r = r_mean_top + empty * 8
    Y_m = empty * 4 + m_mean_top + empty * 4
    Y_rm = empty * 8 + rm_mean_top
    
    
    fig, ax = plt.subplots()
    ax.bar(X,Y_r,width=1, color='green')
    ax.bar(X,Y_m,width=1, color='blue')
    ax.bar(X,Y_rm,width=1, color='red')
    
    r_labels = ['t'+str(x) for x in idx_mean_r]
    m_labels = ['t'+str(x) for x in idx_mean_m]
    rm_labels = ['t'+str(x) for x in idx_mean_rm]
    
    ax.set_xticks(xrange(11))
    ax.set_xticklabels([r_labels[0], r_labels[1],r_labels[2], '', m_labels[0], m_labels[1],m_labels[2], '', \
                        rm_labels[0], rm_labels[1],rm_labels[2]])
    


    plt.show()

    ### Bins ###
    Y_r = r_bin_top + empty * 8
    Y_m = empty * 4 + m_bin_top + empty * 4
    Y_rm = empty * 8 + rm_bin_top
    
    
    fig, ax = plt.subplots()
    ax.bar(X,Y_r,width=1, color='green')
    ax.bar(X,Y_m,width=1, color='blue')
    ax.bar(X,Y_rm,width=1, color='red')
    
    r_labels = ['b'+str(x) for x in idx_bin_r]
    m_labels = ['b'+str(x) for x in idx_bin_m]
    rm_labels = ['b'+str(x) for x in idx_bin_rm]
    
    ax.set_xticks(xrange(11))
    ax.set_xticklabels([r_labels[0], r_labels[1],r_labels[2], '', m_labels[0], m_labels[1],m_labels[2], '',\
                        rm_labels[0], rm_labels[1],rm_labels[2]])
    
    plt.show()


#m_excl = False
#group1,group2,group3 = [30,1,5,200,1,50,1000], [30,1,5,200,1,50,1000],[30,1,5,200,1,50,1000]
#get_bins_and_mean(group1,group2,group3,m_excl)
#
#group1,group2,group3 = [30,1,5,200,10,50,1000], [30,1,5,200,10,50,1000],[30,1,5,200,10,50,1000]
#get_bins_and_mean(group1,group2,group3,m_excl)
#
#group1,group2,group3 = [30,1,15,200,1,50,1000], [30,1,15,200,1,50,1000],[30,1,15,200,1,50,1000]
#get_bins_and_mean(group1,group2,group3,m_excl)
#
#group1,group2,group3 = [30,1,15,200,10,50,1000], [30,1,15,200,10,50,1000],[30,1,15,200,10,50,1000]
#get_bins_and_mean(group1,group2,group3,m_excl)
#

def targets_across_sims(group1,group2,group3,m_excl):
    print 'Loading data'
    df_r = pd.read_csv('./results/r-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group1[0],group1[1],group1[2],group1[3],group1[4],group1[5],str(m_excl)))
    df_m = pd.read_csv('./results/m-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group2[0],group2[1],group2[2],group2[3],group2[4],group2[5],str(m_excl)))
    df_rm = pd.read_csv('./results/rmd-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (group3[0],group3[1],group3[2],group3[3],group3[4],group3[5],str(m_excl)))
    
    targets = [231,236,291,306,326,336]
    targets = ['t_final'+str(x) for x in targets]
    df_r = df_r[targets]
    df_m = df_m[targets]
    df_rm = df_rm[targets]
    
    df_r.iloc[:30].plot.barh(stacked=False) #first 30 independent runs
    plt.show()
    
    df_m.iloc[:30].plot.barh(stacked=False)
    plt.show()
    
    df_rm.iloc[:30].plot.barh(stacked=False) 
    plt.show()
    
#    df_r.iloc[:30].plot(subplots=True) #first 30 independent runs
#    plt.show()
    #
    #df_m.iloc[:30].plot(subplots=True)
    #plt.show()
    ##
    #df_rm.iloc[:30].plot(subplots=True) 
    #plt.show()


#plt.show()

m_excl = False
#group1,group2,group3 = [30,1,15,200,10,50,1000], [30,1,15,200,10,50,1000],[30,1,15,200,10,50,1000]
#targets_across_sims(group1,group2,group3,m_excl)

#group1,group2,group3 = [30,1,5,200,1,50,1000], [30,1,5,200,1,50,1000],[30,1,5,200,1,50,1000]
#targets_across_sims(group1,group2,group3,m_excl)

#group1,group2,group3 = [30,1,20,200,1,50,1000], [30,1,20,200,5,50,1000],[30,1,20,200,5,50,1000]
#targets_across_sims(group1,group2,group3,m_excl)




#################################
def get_independent_runs_heatmap(lam,a,k,sam,l,g,m_excl):
    print 'Loading data'
    df = pd.read_csv('./results/rmd-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (lam,a,k,sam,l,g,m_excl))
    df = df.ix[:,'t_final0':]
    targets = [231,236,291,306,326,336]
    
    for i in targets:
        targetcol = df['t_final'+str(i)]
        df.drop(labels=['t_final'+str(i)], axis=1,inplace=True)
        df.insert(0,'t_final'+str(i), targetcol)
    
    df = df.iloc[:25]
    
    axlabels=["" for _ in xrange(df.shape[1])]
    axlabels[len(axlabels)/2] = 'types'
    ax = sns.heatmap(df, xticklabels=axlabels)#, annot=True) 
    ax.invert_yaxis()
    plt.show()
##################################
lam,a,k,sam,l,g = 30,1,15,200,10,50

#def get_independent_runs_heatmap(lam,a,k,sam,l,g,m_excl):
print 'Loading data'
df = pd.read_csv('./results/rmd-s3-m3-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' % (lam,a,k,sam,l,g,m_excl))
df = df.ix[:,'t_final0':]
#sort by mean
df_sorted = df.reindex_axis(df.mean().sort_values(ascending=False).index, axis=1)
#slide subset of data
df = df_sorted.iloc[:50,:25] #50 independent simulations, top 25

targets = [231,236,291,306,326,336]
targets = [df.columns.get_loc("t_final"+str(x)) for x in targets]

nrows,ncols = df.shape

# Make data for display
mask = np.array(nrows * [ncols * [False]], dtype=bool)
for t_idx in targets:
    mask[:,t_idx] = True
red = np.ma.masked_where(mask, df)#np.repeat(x, 2, axis=1))

mask = np.array(nrows * [ncols * [True]], dtype=bool)
for t_idx in targets:
    mask[:,t_idx] = False
blue = np.ma.masked_where(mask, df)

fig, ax = plt.subplots()
redmesh = ax.pcolormesh(red, cmap='BuGn')
bluemesh = ax.pcolormesh(blue, cmap='Oranges')

# Make things a touch fancier

#       yticks=np.arange(nrows) + 0.5,
#       xticklabels=['Column ' + letter for letter in 'ABCDE'],
#       yticklabels=['Row {}'.format(i+1) for i in range(nrows)])
#

plt.tick_params(axis='x',    which='both', bottom='off', top='off',labelbottom='off') 

fig.subplots_adjust(bottom=0.05, right=0.78, top=0.88)
cbar = fig.colorbar(bluemesh, cax=fig.add_axes([0.81, 0.05, 0.04, 0.83]))#, extend='max', extendfrac=[0,0], extendrect=True)
cbar.ax.text(0.55, 0.1, 'Targets', rotation=90, ha='center', va='center',
             transform=cbar.ax.transAxes, color='gray')
cbar = fig.colorbar(redmesh, cax=fig.add_axes([0.9, 0.05, 0.04, 0.83]))#, extend='max', extendfrac=[0,5],extendrect=True,spacing='uniform')
cbar.ax.text(0.55, 0.1, 'Other types', rotation=90, ha='center', va='center',
             transform=cbar.ax.transAxes, color='gray')

# Make the grouping clearer
#ax.set_xticks(np.arange(0, 2 * ncols, 2), minor=True)
#ax.grid(axis='x', ls='-', color='gray', which='minor')
#ax.grid(axis='y', ls=':', color='gray')

plt.show()
















#for i in targets:
#    targetcol = df['t_final'+str(i)]
#    df.drop(labels=['t_final'+str(i)], axis=1,inplace=True)
#    df.insert(0,'t_final'+str(i), targetcol)


axlabels=["" for _ in xrange(df.shape[1])]
axlabels[len(axlabels)/2] = 'types'
#ax = sns.heatmap(df, xticklabels=axlabels)#, annot=True) 
#ax.invert_yaxis()
#plt.show()






#################################### Deprecrated plot functions for 112 state space ################################################
#def get_prior_plot(m_excl):
#    from lexica import get_prior, get_lexica
#
#    prior = get_prior(get_lexica(3,3,m_excl))
#    
#    X = np.arange(len(prior))
#    if m_excl:
#        target = 24
#    else:
#        target = 68
#    Y_target = [0 for _ in xrange(target)] + [prior[target]] + [0 for _ in xrange(len(prior)-target-1)]
#    
#    prior[target] = 0
#    Y_rest = prior
#    
#    fig, ax = plt.subplots()
#    ax.bar(X,Y_target,width=1, color='green')
#    ax.bar(X,Y_rest,width=1)
#    plt.show()
#
#get_prior_plot(False)
#
#def get_utility_heatmap(s_amount,m_amount,lam,alpha,m_excl):
#    print 'Loading U-matrix'
#    df = pd.read_csv('./matrices/umatrix-s%d-m%d-lam%d-a%d-me%s.csv' %(s_amount,m_amount,lam,alpha,str(m_excl)))
#    
#    axlabels=["" for _ in xrange(df.shape[1])]
#    show = np.arange(0,df.shape[1]+1,10)
#    for i in xrange(len(show)):
#        axlabels[show[i]] = show[i]
#
#    ax = sns.heatmap(df, xticklabels=axlabels, yticklabels=axlabels)#, annot=True) 
#    ax.invert_yaxis()
#
#
##    plt.yticks(rotation=0)
#    from matplotlib.patches import Rectangle
#    if m_excl:
#        ax.add_patch(Rectangle((24,15), 1, 1, fill=False, edgecolor='blue', lw=3))
#    else:
#        ax.add_patch(Rectangle((68,43), 1, 1, fill=False, edgecolor='blue', lw=3))
#
#    plt.show()
#
##get_utility_heatmap(3,3,10,1,False)
#
#
#def get_mutation_heatmap(s_amount,m_amount,lam,alpha,k,samples,l,m_excl):
#    print 'Loading Q-matrix'
#    df = pd.read_csv('./matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(s_amount,m_amount,lam,alpha,k,samples,l,str(m_excl)))
# 
#    axlabels=["" for _ in xrange(df.shape[1])]
#    show = np.arange(0,df.shape[1]+1,10)
#    for i in xrange(len(show)):
#        axlabels[show[i]] = show[i]
#   
#    ax = sns.heatmap(df, yticklabels=axlabels, xticklabels=axlabels, cmap="YlGnBu")#, annot=True) 
##    plt.yticks(rotation=0)
#    from matplotlib.patches import Rectangle
#    if m_excl: 
#        ax.add_patch(Rectangle((24,15), 1, 1, fill=False, edgecolor='red', lw=3))
#    else:
#        ax.add_patch(Rectangle((68,43), 1.25, 1.25, fill=False, edgecolor='red', lw=3))
#
#    ax.invert_yaxis()
#    plt.show()
#
##get_mutation_heatmap(3,3,30,1,5,200,1,False)
##get_mutation_heatmap(3,3,30,1,5,200,10,False)
##get_mutation_heatmap(3,3,30,1,15,200,1,False)
##get_mutation_heatmap(3,3,30,1,15,200,10,False)
##
#
#
#
#def get_some_analysis(group1,group2,group3,m_excl):
#    print 'Loading data'
#    df_r = pd.read_csv('./results/00mean-r-s3-m3-g50-r1000-me%s.csv' % (str(m_excl)))
#    df_m = pd.read_csv('./results/00mean-m-s3-m3-g50-r1000-me%s.csv' % (str(m_excl)))
#    df_rm = pd.read_csv('./results/00mean-rmd-s3-m3-g50-r1000-me%s.csv' %(str(m_excl)))
#    
#    df_r = df_r.loc[df_r['lam'] == group1[0]]
#    df_r = df_r.loc[df_r['alpha'] == group1[1]]
#    df_r = df_r.loc[df_r['k'] == group1[2]]
#    df_r = df_r.loc[df_r['samples'] == group1[3]]
#    df_r = df_r.loc[df_r['l'] == group1[4]]
#    df_r = df_r.loc[df_r['gens'] == group1[5]]
#    df_r = df_r.loc[df_r['runs'] == group1[6]]
#    final_r = df_r.loc[:,'t_mean0':]
#    
#    df_m = df_m.loc[df_m['lam'] == group2[0]]
#    df_m = df_m.loc[df_m['alpha'] == group2[1]]
#    df_m = df_m.loc[df_m['k'] == group2[2]]
#    df_m = df_m.loc[df_m['samples'] == group2[3]]
#    df_m = df_m.loc[df_m['l'] == group2[4]]
#    df_m = df_m.loc[df_m['gens'] == group2[5]]
#    df_m = df_m.loc[df_m['runs'] == group2[6]]
#    final_m = df_m.loc[:,'t_mean0':]
#    
#    df_rm = df_rm.loc[df_rm['lam'] == group3[0]]
#    df_rm = df_rm.loc[df_rm['alpha'] == group3[1]]
#    df_rm = df_rm.loc[df_rm['k'] == group3[2]]
#    df_rm = df_rm.loc[df_rm['samples'] == group3[3]]
#    df_rm = df_rm.loc[df_rm['l'] == group3[4]]
#    df_rm = df_rm.loc[df_rm['gens'] == group3[5]]
#    df_rm = df_rm.loc[df_rm['runs'] == group3[6]]
#    final_rm = df_rm.loc[:,'t_mean0':]
#    
#    r_array = map(list,final_r.values)[0]
#    r_sorted = sorted(r_array,reverse=True)
#    r_top = r_sorted[:3]
#    
#    m_array = map(list,final_m.values)[0]
#    m_sorted = sorted(m_array,reverse=True)
#    m_top = m_sorted[:3]
#    
#    rm_array = map(list,final_rm.values)[0]
#    rm_sorted = sorted(rm_array,reverse=True)
#    rm_top = rm_sorted[:3]
#    
#    
#    idx_r,idx_m,idx_rm = [], [], []
#    
#    for i in r_top:
#        index = r_array.index(i)
#        idx_r.append(index)
#    
#    for i in m_top:
#        index = m_array.index(i)
#        idx_m.append(index)
#    
#    for i in rm_top:
#        index = rm_array.index(i)
#        idx_rm.append(index)
#    if m_excl: target = 24
#    else: target = 68
#    
#    if target not in idx_r:
#        r_top.pop()
#        idx_r.pop()
#        r_top.append(r_array[target])
#        idx_r.append(24)
#    
#    if target not in idx_m:
#        m_top.pop()
#        idx_m.pop()
#        m_top.append(m_array[target])
#        idx_r.append(target)
#    
#    if target not in idx_rm:
#        rm_top.pop()
#        idx_rm.pop()
#        rm_top.append(rm_array[target])
#        idx_r.append(target)
#    
#    
#    X = np.arange(21)
#    empty = [0 for _ in xrange(3)]
#    
#    Y_r = empty + r_top + empty * 5
#    Y_m = empty * 3 + m_top + empty * 3
#    Y_rm = empty * 5 + rm_top + empty
#
#    
#    fig, ax = plt.subplots()
#    ax.bar(X,Y_r,width=1, color='green')
#    ax.bar(X,Y_m,width=1, color='blue')
#    ax.bar(X,Y_rm,width=1, color='red')
#    
#    r_labels = ['t'+str(x) for x in idx_r]
#    m_labels = ['t'+str(x) for x in idx_m]
#    rm_labels = ['t'+str(x) for x in idx_rm]
#
#    ax.set_xticks(xrange(21))
#    ax.set_xticklabels(['', '', '', r_labels[0], r_labels[1],r_labels[2], '', '', '', m_labels[0], m_labels[1],m_labels[2], '', '', '',\
#                        rm_labels[0], rm_labels[1],rm_labels[2], '', '', ''])
#    
#    plt.show()
#
##m_excl = False
##group1,group2,group3 = [30,1,5,200,1,50,1000], [30,1,5,200,1,50,1000],[30,1,5,200,1,50,1000]
##get_some_analysis(group1,group2,group3,m_excl)
##
##group1,group2,group3 = [30,1,5,200,10,50,1000], [30,1,5,200,10,50,1000],[30,1,5,200,10,50,1000]
##get_some_analysis(group1,group2,group3,m_excl)
##
##group1,group2,group3 = [30,1,15,200,1,50,1000], [30,1,15,200,1,50,1000],[30,1,15,200,1,50,1000]
##get_some_analysis(group1,group2,group3,m_excl)
##
##group1,group2,group3 = [30,1,15,200,10,50,1000], [30,1,15,200,10,50,1000],[30,1,15,200,10,50,1000]
##get_some_analysis(group1,group2,group3,m_excl)
