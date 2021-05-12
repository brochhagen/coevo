import pandas as pd
import numpy as np 
import csv

target_types = [231,236,291,306,326,336]
list1 = [1,5,20] #lambda
list2 = [1,5,15] #l
def get_data_to_csv(target_types,list1,list2):
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the majority type to come from other columns
    targets = ['t_final'+str(z) for z in target_types]
    non_targets = [x for x in restrict_to_final if x not in targets]
    seq_length = 5
 
    f = csv.writer(open('./data-plot4.csv','wb'))
    f.writerow(['runID','lambda','l','proportion largest target','proportion largest nontarget'])
  
    for x in xrange(len(list1)):
        for y in xrange(len(list2)):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',list1[y],seq_length,list2[x]))
            
            df['largest target'] = df[targets].max(axis=1)
            df['largest non target'] = df[non_targets].max(axis=1)
            for idx,row in df.iterrows():
                f.writerow([str(row['runID']),str(row['lam']),str(row['l']),str(row['largest target']),str(row['largest non target'])])


#get_data_to_csv(target_types,list1,list2)

def get_data_to_csv_plot1(targets,competitors,lall,list1):
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the majority type to come from other columns
    targets = ['t_final'+str(z) for z in target_types]
    competitors = ['t_final'+str(z) for z in competitors]
    lall = ['t_final'+str(z) for z in lall]
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)]
    other_types = [x for x in restrict_to_final if x not in targets+competitors+lall]

    seq_length = 5
    kind = 'r'

    f = csv.writer(open('./ordered-data-plot2.csv','wb'))
    f.writerow(['runID','lambda','1st target', '2nd target', '3rd target', '4th target', '5th target', '6th target']+\
               ['1st competitor', '2nd competitor', '3rd competitor', '4th competitor', '5th competitor', '6th competitor'] +\
               ['lit lall', 'prag lall'] + ['other type'+str(z) for z in xrange(432-len(targets)-len(competitors)-len(lall))])
    for i in xrange(len(list1)):
        df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,5))

        for idx,row in df.iterrows():
            ordered_targets = row[targets].values.tolist()
            ordered_targets.sort(reverse=True)
            ordered_targets = [str(x) for x in ordered_targets]


            ordered_competitors = row[competitors].values.tolist()
            ordered_competitors.sort(reverse=True)
            ordered_competitors = [str(x) for x in ordered_competitors]

            ordered_lall = row[lall].values.tolist()
            ordered_lall = [str(x) for x in ordered_lall]

            ordered_other = row[other_types].values.tolist()
            ordered_other.sort(reverse=True)
            ordered_other = [str(x) for x in ordered_other]

            f.writerow([str(row['runID']),str(row['lam'])] +\
                       ordered_targets + ordered_competitors + ordered_lall + ordered_other)
    return df

def get_data_to_csv_plot2(targets,competitors,lall,list1,list2):
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the majority type to come from other columns
    targets = ['t_final'+str(z) for z in target_types]
    competitors = ['t_final'+str(z) for z in competitors]
    lall = ['t_final'+str(z) for z in lall]
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)]
    other_types = [x for x in restrict_to_final if x not in targets+competitors+lall]

    seq_length = 5
    kind = 'm'

    f = csv.writer(open('./ordered-data-plot3.csv','wb'))
    f.writerow(['runID','lambda','l','1st target', '2nd target', '3rd target', '4th target', '5th target', '6th target']+\
               ['1st competitor', '2nd competitor', '3rd competitor', '4th competitor', '5th competitor', '6th competitor'] +\
               ['lit lall', 'prag lall'] + ['other type'+str(z) for z in xrange(432-len(targets)-len(competitors)-len(lall))])
    for i in xrange(len(list1)):
        for j in xrange(len(list2)):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))
    
            for idx,row in df.iterrows():
                ordered_targets = row[targets].values.tolist()
                ordered_targets.sort(reverse=True)
                ordered_targets = [str(x) for x in ordered_targets]
    
    
                ordered_competitors = row[competitors].values.tolist()
                ordered_competitors.sort(reverse=True)
                ordered_competitors = [str(x) for x in ordered_competitors]
    
                ordered_lall = row[lall].values.tolist()
                ordered_lall = [str(x) for x in ordered_lall]
    
                ordered_other = row[other_types].values.tolist()
                ordered_other.sort(reverse=True)
                ordered_other = [str(x) for x in ordered_other]
    
                f.writerow([str(row['runID']),str(row['lam']),str(row['l'])] +\
                           ordered_targets + ordered_competitors + ordered_lall + ordered_other)
    return df

def get_data_to_csv_plot3(targets,competitors,lall,list1,list2):
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the majority type to come from other columns
    targets = ['t_final'+str(z) for z in target_types]
    competitors = ['t_final'+str(z) for z in competitors]
    lall = ['t_final'+str(z) for z in lall]
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)]
    other_types = [x for x in restrict_to_final if x not in targets+competitors+lall]

    seq_length = 5
    kind = 'rmd'

    f = csv.writer(open('./ordered-data-plot4.csv','wb'))
    f.writerow(['runID','lambda','l','1st target', '2nd target', '3rd target', '4th target', '5th target', '6th target']+\
               ['1st competitor', '2nd competitor', '3rd competitor', '4th competitor', '5th competitor','6th competitor'] +\
               ['lit lall', 'prag lall'] + ['other type'+str(z) for z in xrange(432-len(targets)-len(competitors)-len(lall))])
    for i in xrange(len(list1)):
        for j in xrange(len(list2)):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))
    
            for idx,row in df.iterrows():
                ordered_targets = row[targets].values.tolist()
                ordered_targets.sort(reverse=True)
                ordered_targets = [str(x) for x in ordered_targets]
    
    
                ordered_competitors = row[competitors].values.tolist()
                ordered_competitors.sort(reverse=True)
                ordered_competitors = [str(x) for x in ordered_competitors]
    
                ordered_lall = row[lall].values.tolist()
                ordered_lall = [str(x) for x in ordered_lall]
    
                ordered_other = row[other_types].values.tolist()
                ordered_other.sort(reverse=True)
                ordered_other = [str(x) for x in ordered_other]
    
                f.writerow([str(row['runID']),str(row['lam']),str(row['l'])] +\
                           ordered_targets + ordered_competitors + ordered_lall + ordered_other)
    return df


            

targets = [231,236,291,306,326,336]
competitors = [225,235,255,270,325,330] 
lall = [0,216]
list1 = [1,5,20]

get_data_to_csv_plot1(targets,competitors,lall,list1)

list1 = [20]
list2 = [1,15]
get_data_to_csv_plot2(targets,competitors,lall,list1,list2)

list1 = [1,5,20]
list2 = [1,5,15]
get_data_to_csv_plot3(targets,competitors,lall,list1,list2)


