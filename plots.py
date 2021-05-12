###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches #to draw rectangles
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys
import csv

import datetime

def analysis(types,val1,val2,kind):
    from lexica import get_lexica,get_lexica_bins
    from rmd import get_type_bin
    seq_length = 5
    lex = get_lexica(3,3,False)
    bins = get_lexica_bins(lex)

    values_of_inc_in_bins = np.zeros(len(bins))
    numb_of_inc = np.zeros(len(bins))

    df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,val1,seq_length,val2))
    all_types = ['t_final'+str(z) for z in xrange(432)]
    df = df[all_types]
    for i in xrange(len(df)):
        type_of_inc = df.iloc[i].idxmax(axis=1)
        value = df.iloc[i][type_of_inc]
        stripped_type = int(type_of_inc[7:]) #convert name into idx
        bin_of_type = get_type_bin(stripped_type,bins)
        values_of_inc_in_bins[bin_of_type] += value #add value to store in vector
        numb_of_inc[bin_of_type] += 1 #add 1 to counter of times incumbent

    return [values_of_inc_in_bins / len(df), numb_of_inc]


def get_subfigs_replication(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 5
    
    types_to_plot = ['t_final'+str(x) for x in type_list]  
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns

    df1 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('r',list1[0],seq_length,list2[0]))
    df2 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('r',list1[1],seq_length,list2[0]))
    
    df1 = df1[restrict_to_final]
    df1['incumbent'] = df1.max(axis=1)

    df2 = df2[restrict_to_final]
    df2['incumbent'] = df2.max(axis=1)
    
    df1_large = df1[types_to_plot + ['incumbent']]
    df1_large = df1_large.iloc[:num_of_runs]
    Y1_large = df1_large.values
                 
    df2_large = df2[types_to_plot + ['incumbent']]
    df2_large = df2_large.iloc[:num_of_runs]
    Y2_large = df2_large.values

    df1_targets = df1[types_to_plot]
    df1_targets = df1_targets.iloc[:num_of_runs]
    Y1_small = df1_targets.values

    df2_targets = df2[types_to_plot]
    df2_targets = df2_targets.iloc[:num_of_runs]
    Y2_small = df2_targets.values



    colors = ('palegreen','forestgreen','limegreen','darkgreen','green','mediumseagreen','darkred')
    from stackedbar import StackedBarGrapher
    SBG = StackedBarGrapher()

    xlabels = [x for x in xrange(num_of_runs)] #runs
    al = 0.6 #alpha
    ax1_large = plt.subplot(1,3,1)
    SBG.stackedBarPlot(ax1_large,Y1_large,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')


    ax2_large = plt.subplot(1,3,2);
    SBG.stackedBarPlot(ax2_large,Y2_large,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')

    ax1_small = plt.subplot(2,3,3)
    SBG.stackedBarPlot(ax1_small,Y1_small,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')

    ax2_small = plt.subplot(2,3,6)
    SBG.stackedBarPlot(ax2_small,Y2_small,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')

    #Layout
    from matplotlib.ticker import FormatStrFormatter
    ax1_large.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1_large.tick_params(axis='both',which='major',labelsize=17)
    ax1_large.set_xlabel(r'$\lambda$ = '+str(list1[0]),fontsize=25)
    ax1_large.xaxis.set_label_position('top')

    #Start counting simulations with 1 and not 0 on the x-axis and hide other ticks
    xlabels = [item.get_text() for item in ax1_large.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax1_large.set_xticklabels(xlabels)

    p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
    p2 = patches.Rectangle((0, 0), 1, 1, fc="darkred",alpha=al)
    ax1_large.legend([p1, p2], ['prag. L-lack','Other majority'],loc='best',prop={'size':14})

    ax2_large.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2_large.tick_params(axis='both',which='major',labelsize=17)
    ax2_large.set_xlabel(r'$\lambda$ = '+str(list1[1]),fontsize=25)
    ax2_large.xaxis.set_label_position('top')

    xlabels = [item.get_text() for item in ax2_large.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax2_large.set_xticklabels(xlabels)

    ax1_small.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1_small.tick_params(axis='both',which='major',labelsize=17)
    ax1_small.set_xlabel(r'$\lambda$ = '+str(list1[0]),fontsize=25)
    ax1_small.xaxis.set_label_position('top')

    xlabels = [item.get_text() for item in ax1_small.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax1_small.set_xticklabels(xlabels)


    ax2_small.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2_small.tick_params(axis='both',which='major',labelsize=17)
    ax2_small.set_xlabel(r'$\lambda$ = '+str(list1[1]),fontsize=25)
    ax2_small.xaxis.set_label_position('top')

    xlabels = [item.get_text() for item in ax2_small.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax2_small.set_xticklabels(xlabels)


    plt.tight_layout()
    plt.show()

def get_subfigs_mutation(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 5
    
    types_to_plot = ['t_final'+str(x) for x in type_list]  
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns

    df1 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('m',list1[0],seq_length,list2[0]))
    df2 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('m',list1[0],seq_length,list2[1]))
    
    df1 = df1[restrict_to_final]
    df2 = df2[restrict_to_final]
    
    df1_large = df1[types_to_plot]
    df1_large = df1_large.iloc[:num_of_runs]
    Y1_large = df1_large.values
                 
    df2_large = df2[types_to_plot]
    df2_large = df2_large.iloc[:num_of_runs]
    Y2_large = df2_large.values

    colors = ('palegreen','forestgreen','limegreen','darkgreen','green','mediumseagreen','darkred')
    from stackedbar import StackedBarGrapher
    SBG = StackedBarGrapher()

    xlabels = [x for x in xrange(num_of_runs)] #runs
    al = 0.6 #alpha
    ax1_large = plt.subplot(1,3,1)
    SBG.stackedBarPlot(ax1_large,Y1_large,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')

    ax2_large = plt.subplot(1,3,2);
    SBG.stackedBarPlot(ax2_large,Y2_large,colors,gap=.5,al=al,xLabels=xlabels)
    plt.xticks(rotation='horizontal')

    #Third subfigure is the prior
    ax_prior = plt.subplot(1,3,3)

    from lexica import get_prior,get_lexica
    priors = get_prior(get_lexica(3,3,False))
    priors = list(priors)

    targets = [231,236,291,306,326,336]
    targets = targets + [x-216 for x in targets] #adding literal Llack
    
    lbound = [225, 235, 255, 270, 325, 330]
    lbound = lbound + [x-216 for x in lbound]

    lall = [0,216]



    white_space = 17 #to separate from other priors
    Y_target = [0 for _ in xrange(len(priors)+white_space*3)]
    Y_lbound = [0 for _ in xrange(len(priors)+white_space*3)]
    Y_all = [0 for _ in xrange(len(priors)+white_space*3)]

    Y_target[0] = priors[targets[0]]
    Y_target[2] = priors[targets[1]]
    Y_target[4] = priors[targets[2]]
    Y_target[6] = priors[targets[3]]
    Y_target[8] = priors[targets[4]]
    Y_target[10] = priors[targets[5]]

    Y_target[12] = priors[targets[6]]
    Y_target[14] = priors[targets[7]]
    Y_target[16] = priors[targets[8]]
    Y_target[18] = priors[targets[9]]
    Y_target[20] = priors[targets[10]]
    Y_target[22] = priors[targets[11]]
   
    Y_lbound[len(targets)+17+0] = priors[lbound[0]]
    Y_lbound[len(targets)+17+2] = priors[lbound[1]]
    Y_lbound[len(targets)+17+4] = priors[lbound[2]]
    Y_lbound[len(targets)+17+6] = priors[lbound[3]]
    Y_lbound[len(targets)+17+8] = priors[lbound[4]]
    Y_lbound[len(targets)+17+10] = priors[lbound[5]]

    Y_lbound[len(targets)+17+12] = priors[lbound[6]]
    Y_lbound[len(targets)+17+14] = priors[lbound[7]]
    Y_lbound[len(targets)+17+16] = priors[lbound[8]]
    Y_lbound[len(targets)+17+18] = priors[lbound[9]]
    Y_lbound[len(targets)+17+20] = priors[lbound[10]]
    Y_lbound[len(targets)+17+22] = priors[lbound[11]]

    Y_all[len(targets)+17+len(lbound)+17+0] = priors[lall[0]]
    Y_all[len(targets)+17+len(lbound)+17+3] = priors[lall[1]]


    all_types = targets+lbound+lall
    all_types.sort()

    #reorder prior
    for t in all_types:
        priors.insert(0,priors.pop(t))
    for t in xrange(len(all_types)):
        priors[t] = 0

   
    Y_rest = priors[:len(targets)] + [0 for _ in xrange(white_space)] + priors[len(targets):len(targets)+len(lbound)] +\
     [0 for _ in xrange(white_space)] + priors[len(targets)+len(lbound):len(all_types)] + [0 for _ in xrange(white_space)] + priors[len(all_types):]

            
    Y_target.reverse()
    Y_rest.reverse()
    Y_lbound.reverse()
    Y_all.reverse()
   
    ax_prior.grid(False)
    
    X = np.arange(len(priors)+white_space*3)
    ax_prior.barh(X,Y_target,color='green',alpha=al)
    ax_prior.barh(X,Y_lbound,color='crimson',alpha=al)
    ax_prior.barh(X,Y_all,color='dimgrey',edgecolor='dimgrey')
    ax_prior.barh(X,Y_rest,color='royalblue',alpha=al)



    #Layout
    from matplotlib.ticker import FormatStrFormatter
    ax1_large.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1_large.tick_params(axis='both',which='major',labelsize=17)
    ax1_large.set_xlabel('l = '+str(list2[0]),fontsize=25)
    ax1_large.xaxis.set_label_position('top')
    ax1_large.set_ylim(0,np.max(Y1_large)+0.065)

    xlabels = [item.get_text() for item in ax1_large.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax1_large.set_xticklabels(xlabels)


    p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
    p2 = patches.Rectangle((0, 0), 1, 1, fc="darkred",alpha=al)
    ax1_large.legend([p1, p2], ['prag. L-lack','Other majority'],loc='best',prop={'size':17})


    ax2_large.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2_large.tick_params(axis='both',which='major',labelsize=17)
    ax2_large.set_xlabel('l = '+str(list2[1]),fontsize=25)
    ax2_large.xaxis.set_label_position('top')
    ax2_large.set_ylim(0,0.75)

    xlabels = [item.get_text() for item in ax2_large.get_xticklabels()]
    xlabels = [str(int(x) + 1) for x in xlabels]
    ax2_large.set_xticklabels(xlabels)

    ax_prior.set_xlabel('Prior',fontsize=25)
    ax_prior.xaxis.set_label_position('top')
    ax_prior.set_ylabel('Types', fontsize=25)
    ax_prior.tick_params(axis='both', which='major', labelsize=17)
    ax_prior.annotate('L-lack',xy=(.0063236,467),color='green',fontweight='extra bold')
    ax_prior.annotate('L-bound',xy=(0.0028603,438.5),color='crimson',fontweight='extra bold')
    ax_prior.annotate('L-all',xy=(.0063236,429), color='dimgrey',fontweight='extra bold')

    ax_prior.set_ylim(0,len(priors)+white_space*3+3)
    ylabels = [x for x in xrange(len(X))]

    #Hide all y-labels
    for label in ax_prior.get_yticklabels():
        label.set_visible(False)



    plt.tight_layout()
    plt.show()



def get_subfigs_rmd(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 5
    
    x_figs = len(list1)
    y_figs = len(list2)
    fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs)

    types_to_plot = ['t_final'+str(x) for x in type_list] 

    #Finding incumbent for each individual run per parameter configuration and checking whether it's already one of the target types
    #This is pretty roundabout, but explicit and relatively fast
    incumbent_list = [] #list of x-y configurations where incumbent is not target type. 
    for x in xrange(y_figs):
        for y in xrange(x_figs):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',list1[y],seq_length,list2[x]))
            all_types = ['t_final'+str(z) for z in xrange(432)]
            df = df[all_types]
            incumbents = 0
            for run in xrange(num_of_runs):
                ds = df.iloc[run]
                incumbent = ds.idxmax()
                if not incumbent in types_to_plot: 
                    incumbents += 1
            if incumbents == num_of_runs: incumbent_list.append([x,y])
    print '### Incumbent not a target_type in coordinates: ###'
    print incumbent_list

    #Now to the actual plots:
    for x in xrange(y_figs):
        for y in xrange(x_figs):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',list1[y],seq_length,list2[x]))
            restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns
            df = df[restrict_to_final]
        
            types_to_plot = ['t_final'+str(i) for i in type_list] 
            if [x,y] in incumbent_list:
                df['incumbent'] = df.max(axis=1)
                types_to_plot = types_to_plot + ['incumbent']
            df = df[types_to_plot]
            df = df.iloc[:num_of_runs]
            df.plot(ax=axes[x,y],kind='bar', stacked=True, alpha=0.5, color=('palegreen','forestgreen','limegreen','darkgreen','green','mediumseagreen','darkred'), rot=0) #colormap='Greens')

#    #Adding labels and modifying layout 
    learning_labels = [x for x in list2]
    list2.reverse()

    for idx in xrange(len(plt.gcf().axes)):
        ax = plt.gcf().axes[idx]
        ax.tick_params(axis='both', which='major', labelsize=17)

        #Only add top label to subplots that are on the first row
        if idx - x_figs < 0:
            print idx
            x = idx 
            ax.set_xlabel(r'$\lambda$ = '+str(list1[x]),fontsize=20)
            ax.xaxis.set_label_position('top')

        #Only add label to subplots that are on the left-most column
        if idx % x_figs == 0:
            ax.set_ylabel('l = '+str(list2[-1]),fontsize=20)
            list2.pop()
        elif y_figs == 1 and prior == False:
            ax.set_ylabel('Proportion in population',fontsize=17)
            list2.pop()

        #Resize range of values on y to the maximal value in a row:
        if idx in [0,1,2]:
            ax.set_ylim(0,0.065)
        elif idx in [3,4,5]:
            ax.set_ylim(0,0.55)
        elif idx in [6,7,8]:
            ax.set_ylim(0,0.8)
#
        handles, labels = ax.get_legend_handles_labels()
        labels = ['prag. L-lack' for _ in xrange(len(types))] + ['Other majority']
        display = (3,6)
#
        ax.legend([handle for i,handle in enumerate(handles) if i in display], \
                  [label for i,label in enumerate(labels) if i in display], loc='best',prop={'size': 20})
#
#        #Alternatively: Hide legend
        if not idx == display[0]:
            ax.legend().set_visible(False)

        #Start counting simulations with 1 and not 0 on the x-axis and hide other ticks
        if idx in [6,7,8]:
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            xlabels = [str(int(x) + 1) for x in xlabels]
        else:
            xlabels = ["" for _ in ax.get_xticklabels()]
        ax.set_xticklabels(xlabels)

        if not idx in [0,3,6]:
            for label in ax.get_yticklabels()[::1]:
                label.set_visible(False)

    plt.tight_layout()
    plt.show()


def get_heatmap_diff_incumbents(type_list,list1,list2,kind):
    all_types = ['t_final'+str(z) for z in xrange(432)]
    target_types = ['t_final'+str(x) for x in type_list]
    other_types = [x for x in all_types if x not in target_types]
    seq_length = 5

    data = np.zeros((len(list1)+1, len(list2)+1))
    for i in xrange(len(list1)):
        for j in xrange(len(list2)):
            #To avoid errors when adding cols to slices of copies:
            dt = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))
            do = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))

            data[0,j+1] = list2[j]
            data[i+1,0] = list1[i]
            
            dt = dt[target_types] 
            do = do[other_types]
            dt['incumbent_t'] = dt.max(axis=1) #incumbent amongst types in d_t
            do['incumbent_o'] = do.max(axis=1) #incumbent amongst types in d_o
            best_of_targets = dt['incumbent_t'].mean()
            best_of_others = do['incumbent_o'].mean()
            data[i+1,j+1] = best_of_targets - best_of_others
    dPrime = data[1:,1:]

    sns.set(font_scale=2)
    ax = sns.heatmap(dPrime, cmap='YlGnBu', xticklabels=list2, yticklabels=list1, annot_kws={"size": 20})# xticklabels=axlabels)#, annot=True) 
    ax.set_ylabel('rationality parameter ('+r'$\lambda$'+')',fontsize=30)#,fontsize=30)
    ax.set_xlabel('posterior parameter (l)',fontsize=30)#,fontsize=30)

    ax.invert_yaxis()
    plt.show()


def get_3d_diff_incumbents(type_list,list1,list2,kind):
    all_types = ['t_final'+str(z) for z in xrange(432)]
    target_types = ['t_final'+str(x) for x in type_list]
    other_types = [x for x in all_types if x not in target_types]
    seq_length = 5
    if os.path.isfile('differences-data.csv'):
        print '#Loading csv matrix, ', datetime.datetime.now()
        data = np.genfromtxt('differences-data.csv',delimiter=',')
    else:
        print '#Computing csv matrix, ', datetime.datetime.now()

        data = np.zeros((len(list1)+1, len(list2)+1))
        for i in xrange(len(list1)):
            for j in xrange(len(list2)):
                #To avoid errors when adding cols to slices of copies:
                dt = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))
                do = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))

                data[0,j+1] = list2[j]
                data[i+1,0] = list1[i]
                
                dt = dt[target_types] 
                do = do[other_types]
                dt['incumbent_t'] = dt.max(axis=1) #incumbent amongst types in d_t
                do['incumbent_o'] = do.max(axis=1) #incumbent amongst types in d_o
                best_of_targets = dt['incumbent_t'].mean()
                best_of_others = do['incumbent_o'].mean()
                data[i+1,j+1] = best_of_targets - best_of_others
        f_data = csv.writer(open('differences-data.csv','wb'))
        for i in data:
            f_data.writerow(i)

        
    dPrime = data[1:,1:]
    sns.set_style("whitegrid")
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xx,yy= np.meshgrid(range(len(data[0,1:])),range(len(data[1:,0])))
    ax.plot_surface(xx,yy,dPrime,cmap='summer',alpha=0.8) #'PrGn', 'winter', 'autumn'
    sns.set(font_scale=2)
    
    ax.set_xticklabels([x for x in xrange(1,16)])
    ax.set_yticklabels([x for x in xrange(1,21)])
    ax.set_ylabel('Rationality parameter ('+r'$\lambda$'+')',fontsize=17)
    ax.set_xlabel('Posterior parameter (l)',fontsize=17)
    ax.set_zlabel('Difference in proportion',fontsize=17)
    
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 10
    
    ylabels = [1,4,7,10,13,15,18,20]
    xlabels = [1,3,5,7,9,11,13,15]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.tight_layout()
    
    plt.show()

types = [231,236,291,306,326,336]

##Plot 1
#list1 = [1,20] #lambda
#list2 = [5] #posterior parameter
#get_subfigs_replication(types,list1,list2)

##Plot 2
#list1 = [20]
#list2 = [1,15]
#get_subfigs_mutation(types,list1,list2)

##Plot 3
#list1 = [1,5,20]
#list2 = [1,5,15]
#get_subfigs_rmd(types,list1,list2)

##Plot 4
#kind = 'rmd'
#type_list = [231,236,291,306,326,336]
#list1 = [x for x in xrange(1,21)]
#list2 = [x for x in xrange(1,16)]
#get_heatmap_diff_incumbents(type_list,list1,list2,kind)


def get_subfigs_hist_replication(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 5
    
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns

    df1 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('r',list1[0],seq_length,list2[0]))
    df2 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('r',list1[1],seq_length,list2[0]))
    df3 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('r',list1[2],seq_length,list2[0]))
   
    df1 = df1[restrict_to_final]
    df1_majority = df1.max(axis=1).iloc[num_of_runs]

    df2 = df2[restrict_to_final]
    df2_majority = df2.max(axis=1).iloc[num_of_runs]
 
    df3 = df3[restrict_to_final]
    df3_majority = df3.max(axis=1).iloc[num_of_runs]

    competitor_lst = [9,  19,  39,  54, 109, 114] + [225,235,255,270,325,330]
    competitors = ['t_final'+str(z) for z in competitor_lst]
    highest_6_competitors = df1[competitors].iloc[num_of_runs].sort_values(ascending=False)[:6].values
    highest_6_competitors = np.random.permutation(highest_6_competitors)
    
    lall = ['t_final'+str(z) for z in [0,216]]
    highest_lall = df1[lall].iloc[num_of_runs].sort_values(ascending=False)[:1].values

    targets = ['t_final'+str(z) for z in [231,236,291,306,326,336]]
    t_vals = df1[targets].iloc[num_of_runs].values
    
    Y1_large = np.concatenate((t_vals,np.concatenate((highest_6_competitors,highest_lall))))

    highest_6_competitors = df2[competitors].iloc[num_of_runs].sort_values(ascending=False)[:6].values
    highest_6_competitors = np.random.permutation(highest_6_competitors)
    highest_lall = df2[lall].iloc[num_of_runs].sort_values(ascending=False)[:1].values
    t_vals = df2[targets].iloc[num_of_runs].values
    Y2_large = np.concatenate((t_vals,np.concatenate((highest_6_competitors,highest_lall))))

    highest_6_competitors = df3[competitors].iloc[num_of_runs].sort_values(ascending=False)[:6].values
    highest_6_competitors = np.random.permutation(highest_6_competitors)
    highest_lall = df3[lall].iloc[num_of_runs].sort_values(ascending=False)[:1].values
    t_vals = df3[targets].iloc[num_of_runs].values
    Y3_large = np.concatenate((t_vals,np.concatenate((highest_6_competitors,highest_lall))))


    colors = ('green','green','green','green','green','green','darkorange','darkorange','darkorange','darkorange','darkorange', 'darkorange', 'red')


    fig,ax = plt.subplots(nrows=1,ncols=3) 
    X = np.arange(len(Y1_large))
    al = 0.6 #alpha
    ax1_large = plt.subplot(1,3,1)
###old code with horizontal bars:
#    plt.bar(X,Y1_large,color=colors,alpha=al)
#    plt.axhline(y=df1_majority,linestyle='dashed')
#    plt.text(0.05,df1_majority-0.0015,'majority type',color='royalblue',fontweight='extra bold')
#    ax1_large.tick_params(axis='both', which='major', labelsize=17)

#    ax2_large = plt.subplot(1,3,2);
#    plt.bar(X,Y2_large,color=colors,alpha=al)
#    plt.axhline(y=df2_majority,linestyle='dashed')
#    ax2_large.tick_params(axis='both', which='major', labelsize=17)
#
#
#    ax3_large = plt.subplot(1,3,3);
#    plt.bar(X,Y3_large,color=colors,alpha=al)
#    plt.axhline(y=df3_majority,linestyle='dashed')#, color=sns.color_palette()[1],linestyle ='dashed', markevery=10)
#    ax3_large.tick_params(axis='both', which='major', labelsize=17)

###New code with barh
    plt.barh(X,Y1_large,color=colors,alpha=al)
    plt.gca().invert_yaxis()
    plt.axvline(x=df1_majority,linestyle='dashed')
    plt.text(df1_majority-0.004,10,'majority type',color='royalblue',fontweight='extra bold',rotation=90)

    ax2_large = plt.subplot(1,3,2);
    plt.barh(X,Y2_large,color=colors,alpha=al)
    plt.gca().invert_yaxis()
    plt.axvline(x=df2_majority,linestyle='dashed')

    ax3_large = plt.subplot(1,3,3);
    plt.barh(X,Y3_large,color=colors,alpha=al)
    plt.gca().invert_yaxis()
    plt.axvline(x=df3_majority,linestyle='dashed')

    #Layout
    ax1_large.set_xlabel(r'$\lambda$ = '+str(list1[0]),fontsize=25)
    ax1_large.xaxis.set_label_position('top')
    ax2_large.set_xlabel(r'$\lambda$ = '+str(list1[1]),fontsize=25)
    ax2_large.xaxis.set_label_position('top')
    ax3_large.set_xlabel(r'$\lambda$ = '+str(list1[2]),fontsize=25)
    ax3_large.xaxis.set_label_position('top')

    fig.text(0.5, 0.02, 'Proportion', ha='center',fontsize=20)
    fig.text(0.02, 0.5, 'Types', va='center', rotation='vertical',fontsize=20)

###Old version:
#    for label in ax1_large.get_xticklabels():
#        label.set_visible(False)
#    for label in ax2_large.get_xticklabels():
#        label.set_visible(False)
#    for label in ax3_large.get_xticklabels():
#        label.set_visible(False)

#    p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
#    p2 = patches.Rectangle((0, 0), 1, 1, fc="darkorange",alpha=al)
#    p3 = patches.Rectangle((0, 0), 1, 1, fc="red",alpha=al)
#
#    ax1_large.legend([p1, p2, p3], ['Target types','Competitor types','L-all'],loc=(0.0340644,0.69441),prop={'size':14})



    for label in ax1_large.get_yticklabels():
        label.set_visible(False)
    for label in ax2_large.get_yticklabels():
        label.set_visible(False)
    for label in ax3_large.get_yticklabels():
        label.set_visible(False)
    p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
    p2 = patches.Rectangle((0, 0), 1, 1, fc="darkorange",alpha=al)
    p3 = patches.Rectangle((0, 0), 1, 1, fc="red",alpha=al)
    ax2_large.legend([p1, p2, p3], ['Target types','Competitor types','L-all'],loc=(0.2,0.69441),prop={'size':14},frameon=True)

    ax1_large.tick_params(axis='both', which='major', labelsize=15)
    ax2_large.tick_params(axis='both', which='major', labelsize=15)
    ax3_large.tick_params(axis='both', which='major', labelsize=15)


    plt.tight_layout()
    plt.show()

def get_subfigs_hist_mutation(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 99
    
#    types_to_plot = ['t_final'+str(x) for x in type_list]  
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns

    df1 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('m',list1[0],seq_length,list2[0]))
    df2 = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('m',list1[0],seq_length,list2[1]))

    df1 = df1[restrict_to_final]
    df1_majority = df1.max(axis=1).iloc[num_of_runs]

    df2 = df2[restrict_to_final]
    df2_majority = df2.max(axis=1).iloc[num_of_runs]

    competitor_lst = [9,  19,  39,  54, 109, 114] + [225,235,255,270,325,330]
    competitors = ['t_final'+str(z) for z in competitor_lst]
    highest_6_competitors = df1[competitors].iloc[num_of_runs].sort_values(ascending=False)[:6].values
    highest_6_competitors = np.random.permutation(highest_6_competitors)

    print 'highest competitors: ', highest_6_competitors
   
    lall = ['t_final'+str(z) for z in [0,216]]
    highest_lall = df1[lall].iloc[num_of_runs].sort_values(ascending=False)[:1].values

    print 'highest lall: ', highest_lall

    targets = ['t_final'+str(z) for z in [231,236,291,306,326,336]]
    t_vals = df1[targets].iloc[num_of_runs].values

    print 'targets ', t_vals

    
    Y1_large = np.concatenate((t_vals,np.concatenate((highest_6_competitors,highest_lall))))

    highest_6_competitors = df2[competitors].iloc[num_of_runs].sort_values(ascending=False)[:6].values
    highest_6_competitors = np.random.permutation(highest_6_competitors)
    highest_lall = df2[lall].iloc[num_of_runs].sort_values(ascending=False)[:1].values
    t_vals = df2[targets].iloc[num_of_runs].values
    Y2_large = np.concatenate((t_vals,np.concatenate((highest_6_competitors,highest_lall))))

    print 'highest competitors: ', highest_6_competitors
    print 'highest lall: ', highest_lall
    print 'targets ', t_vals



    colors = ('green','green','green','green','green','green','darkorange','darkorange','darkorange','darkorange','darkorange', 'darkorange', 'red')

    fig,ax = plt.subplots(nrows=1,ncols=3)
    X = np.arange(len(Y1_large))
    al = 0.6 #alpha
    ax1_large = plt.subplot(1,3,1)

##Old code with horizontal bars:
#    plt.bar(X,Y1_large,color=colors,alpha=al)
#    plt.axhline(y=df1_majority,linestyle='dashed')
#    plt.text(0.05,df1_majority+0.0005,'majority type',color='royalblue',fontweight='extra bold')
#
#    ax2_large = plt.subplot(1,3,2);
#    plt.bar(X,Y2_large,color=colors,alpha=al)
#    plt.axhline(y=df2_majority,linestyle='dashed')
    plt.barh(X,Y1_large,color=colors,alpha=al)
    plt.gca().invert_yaxis()
    plt.axvline(x=df1_majority,linestyle='dashed')
    plt.text(df1_majority-0.004,10,'majority type',color='royalblue',fontweight='extra bold',rotation=90)

    ax2_large = plt.subplot(1,3,2);
    plt.barh(X,Y2_large,color=colors,alpha=al)
    plt.gca().invert_yaxis()
    plt.axvline(x=df2_majority,linestyle='dashed')


    #Third subfigure is the prior
    ax_prior = plt.subplot(1,3,3)

    from lexica import get_prior,get_lexica
    priors = get_prior(get_lexica(3,3,False))
    priors = list(priors)

    targets = [231,236,291,306,326,336]
    targets = targets + [x-216 for x in targets] #adding literal Llack
    
    lbound = [225, 235, 255, 270, 325, 330]
    lbound = lbound + [x-216 for x in lbound]

    lall = [0,216]


    white_space = 17 #to separate from other priors
    Y_target = [0 for _ in xrange(len(priors)+white_space*3)]
    Y_lbound = [0 for _ in xrange(len(priors)+white_space*3)]
    Y_all = [0 for _ in xrange(len(priors)+white_space*3)]

    Y_target[0] = priors[targets[0]]
    Y_target[2] = priors[targets[1]]
    Y_target[4] = priors[targets[2]]
    Y_target[6] = priors[targets[3]]
    Y_target[8] = priors[targets[4]]
    Y_target[10] = priors[targets[5]]

    Y_target[12] = priors[targets[6]]
    Y_target[14] = priors[targets[7]]
    Y_target[16] = priors[targets[8]]
    Y_target[18] = priors[targets[9]]
    Y_target[20] = priors[targets[10]]
    Y_target[22] = priors[targets[11]]
   

    Y_lbound[len(targets)+17+0] = priors[lbound[0]]
    Y_lbound[len(targets)+17+2] = priors[lbound[1]]
    Y_lbound[len(targets)+17+4] = priors[lbound[2]]
    Y_lbound[len(targets)+17+6] = priors[lbound[3]]
    Y_lbound[len(targets)+17+8] = priors[lbound[4]]
    Y_lbound[len(targets)+17+10] = priors[lbound[5]]

    Y_lbound[len(targets)+17+12] = priors[lbound[6]]
    Y_lbound[len(targets)+17+14] = priors[lbound[7]]
    Y_lbound[len(targets)+17+16] = priors[lbound[8]]
    Y_lbound[len(targets)+17+18] = priors[lbound[9]]
    Y_lbound[len(targets)+17+20] = priors[lbound[10]]
    Y_lbound[len(targets)+17+22] = priors[lbound[11]]

    Y_all[len(targets)+17+len(lbound)+17+0] = priors[lall[0]]
    Y_all[len(targets)+17+len(lbound)+17+3] = priors[lall[1]]


    all_types = targets+lbound+lall
    all_types.sort()

    #reorder prior
    for t in all_types:
        priors.insert(0,priors.pop(t))
    for t in xrange(len(all_types)):
        priors[t] = 0

   
    Y_rest = priors[:len(targets)] + [0 for _ in xrange(white_space)] + priors[len(targets):len(targets)+len(lbound)] +\
     [0 for _ in xrange(white_space)] + priors[len(targets)+len(lbound):len(all_types)] + [0 for _ in xrange(white_space)] + priors[len(all_types):]

            
    Y_target.reverse()
    Y_rest.reverse()
    Y_lbound.reverse()
    Y_all.reverse()
   
    ax_prior.grid(False)
    
    X = np.arange(len(priors)+white_space*3)
    ax_prior.barh(X,Y_target,color='green',alpha=al)
    ax_prior.barh(X,Y_lbound,color='darkorange',alpha=al)
    ax_prior.barh(X,Y_all,color='red',edgecolor='red')
    ax_prior.barh(X,Y_rest,color='royalblue',alpha=al)


    #Layout
    from matplotlib.ticker import FormatStrFormatter
    ax1_large.set_xlabel('l = '+str(list2[0]),fontsize=25)
    ax1_large.xaxis.set_label_position('top')
#    ax1_large.set_ylim(0,np.max(Y1_large)+0.005)

    fig.text(0.02, 0.5, 'Types', va='center', rotation='vertical',fontsize=25)


#    for label in ax1_large.get_xticklabels():
#        label.set_visible(False)
#    for label in ax2_large.get_xticklabels():
#        label.set_visible(False)

    for label in ax1_large.get_yticklabels():
        label.set_visible(False)
    for label in ax2_large.get_yticklabels():
        label.set_visible(False)

    p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
    p2 = patches.Rectangle((0, 0), 1, 1, fc="darkorange",alpha=al)
    p3 = patches.Rectangle((0, 0), 1, 1, fc="red",alpha=al)

    ax2_large.legend([p1, p2, p3], ['Target types','Competitor types','L-all'],loc='best',prop={'size':14},frameon=True)
    ax1_large.tick_params(axis='both', which='major', labelsize=17)


    ax2_large.set_xlabel('l = '+str(list2[1]),fontsize=25)
    ax2_large.xaxis.set_label_position('top')
    ax2_large.tick_params(axis='both', which='major', labelsize=17)



    ax_prior.set_xlabel('Prior',fontsize=25)
    ax_prior.xaxis.set_label_position('top')
#    ax_prior.set_ylabel('Types', fontsize=25)
    ax_prior.tick_params(axis='both', which='major', labelsize=17)
    ax_prior.annotate('L-lack',xy=(.0063236,467),color='green',fontweight='extra bold')
    ax_prior.annotate('L-bound',xy=(0.0028603,438.5),color='darkorange',fontweight='extra bold')
    ax_prior.annotate('L-all',xy=(.0063236,429), color='red',fontweight='extra bold')

    ax_prior.set_ylim(0,len(priors)+white_space*3+3)
    ylabels = [x for x in xrange(len(X))]

    plt.tight_layout()
    plt.show()

def get_subfigs_hist_rmd(type_list,list1,list2):
    seq_length = 5
    num_of_runs = 5 #now it's just the ID of the run to plot
    
    x_figs = len(list1)
    y_figs = len(list2)
    fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs)

    types_to_plot = ['t_final'+str(x) for x in type_list] 
    colors = ('green','green','green','green','green','green','darkorange','darkorange','darkorange','darkorange','darkorange', 'darkorange', 'red')

    majorities = []
    for x in xrange(y_figs):
        for y in xrange(x_figs):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',list1[y],seq_length,list2[x]))
            restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns
            df = df[restrict_to_final]
            majorities.append(df.max(axis=1).iloc[num_of_runs])
        
            types_to_plot = ['t_final'+str(i) for i in type_list] 
            
            df = df[types_to_plot]
            df = df.iloc[num_of_runs]
            df.plot(ax=axes[x,y],kind='bar', alpha=0.5, color=colors, rot=0) 



#    #Adding labels and modifying layout 
    learning_labels = [x for x in list2]
    list2.reverse()
    al = 0.6
    for idx in xrange(len(plt.gcf().axes)):
        ax = plt.gcf().axes[idx]
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.axhline(y=majorities[idx],linestyle='dashed')
        if idx == 0:
            ax.text(0.05,majorities[idx]+0.0015,'majority type',color='royalblue',fontweight='extra bold')

        #Only add top label to subplots that are on the first row
        if idx - x_figs < 0:
            print idx
            x = idx 
            ax.set_xlabel(r'$\lambda$ = '+str(list1[x]),fontsize=20)
            ax.xaxis.set_label_position('top')

        #Only add label to subplots that are on the left-most column
        if idx % x_figs == 0:
            ax.set_ylabel('l = '+str(list2[-1]),fontsize=20)
            list2.pop()
        elif y_figs == 1 and prior == False:
            ax.set_ylabel('Proportion in population',fontsize=17)
            list2.pop()

        #Resize range of values on y to the maximal value in a row:
#        if idx in [0,1,2]:
#            ax.set_ylim(0,0.065)
#        elif idx in [3,4,5]:
#            ax.set_ylim(0,0.55)
#        elif idx in [6,7,8]:
#            ax.set_ylim(0,0.8)
        

        if idx == 0:
            p1 = patches.Rectangle((0, 0), 1, 1, fc="green", alpha=al)
            p2 = patches.Rectangle((0, 0), 1, 1, fc="darkorange",alpha=al)
            p3 = patches.Rectangle((0, 0), 1, 1, fc="red",alpha=al)
            ax.legend([p1, p2, p3], ['Target types','Competitor types','prag. L-all'],loc='best',prop={'size':14})

        for label in ax.get_xticklabels():
            label.set_visible(False)
#        
#        if not idx in [0,3,6]:
#            for label in ax.get_yticklabels():
#                label.set_visible(False)

    plt.tight_layout()
    plt.show()


#Total types = 432
types = [231,236,291,306,326,336] + [9,  19,  39,  54, 109, 114] + [216]
#types = [231,236,291,306,326,336] + [225,235,255,270,325,330] + [216]

#Plot 1
list1 = [1,5,20] #lambda
list2 = [5] #posterior parameter
#get_subfigs_hist_replication(types,list1,list2)


#Plot 2:
list1 = [20]
list2 = [5,15]
#get_subfigs_hist_mutation(types,list1,list2)

#Plot 3:
list1 = [1,10,20]
list2 = [5,10,15]
get_subfigs_hist_rmd(types,list1,list2)


##Plot 4 in 3D:
#kind = 'rmd'
#type_list = [231,236,291,306,326,336]
#list1 = [x for x in xrange(1,21)]
#list2 = [x for x in xrange(1,16)]
#get_3d_diff_incumbents(type_list,list1,list2,kind)

