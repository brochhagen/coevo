##This script generates all possible lexica of the types we consider, and their corresponding prior
import numpy as np
from itertools import product,combinations,combinations_with_replacement

def get_lexica(s_amount,m_amount,mutual_exclusivity=True):
    """Generates all possible non-contradictory or tautological lexica for a given number of states and messages"""
    columns = list(product([0.,1.],repeat=s_amount))
    columns.remove((0,0,0)) #remove message false of all states
    columns.remove((1,1,1)) #remove message true of all states
    if mutual_exclusivity:
        matrix = list(combinations(columns,r=m_amount)) #no concept assigned to more than one message
        out = []
        for mrx in matrix:
            lex = np.array([mrx[i] for i in xrange(s_amount)])
            lex = np.transpose(np.array([mrx[i] for i in xrange(s_amount)]))
            out.append(lex)
    else:
        matrix = list(product(columns,repeat=m_amount)) #If we allow for symmetric lexica
        out = []
        for mrx in matrix:
            lex = np.array([mrx[i] for i in xrange(s_amount)])
            lex = np.transpose(np.array([mrx[i] for i in xrange(s_amount)]))
            out.append(lex)
    return out 

def get_lexica_bins(lexica_list):
    """Gathers lexica by type: lexica that are permutations of each other are bundled together. This is occasionally convenient for analyses over lexica types"""
    concepts = [[0,0,1],[0,1,0],[0,1,1],\
                    [1,0,0],[1,0,1],[1,1,0]]
    lexica_concepts = []
    for lex_idx in xrange(len(lexica_list)):
        concept_indices = []
        current_lex = np.transpose(lexica_list[lex_idx])
        for concept_idx in xrange(len(current_lex)):
            concept_indices.append(concepts.index(list(current_lex[concept_idx])))
        lexica_concepts.append(concept_indices)
    
    bin_counter = []
    bins = []
    for lex_idx in xrange(len(lexica_list)):
        sorted_lexica_concepts = lexica_concepts[lex_idx]
        sorted_lexica_concepts.sort()
        if not(sorted_lexica_concepts in bin_counter):
            bin_counter.append(sorted_lexica_concepts)
            bins.append([lex_idx])
        else:
            bins[bin_counter.index(sorted_lexica_concepts)].append(lex_idx)
    ### up to here we get bins for a single linguistic behavior. Now we double that for Literal/Gricean split 
    gricean_bins = []
    for b in bins:
        g_bin = [x+len(lexica_list) for x in b]
        gricean_bins.append(g_bin)
    bins = bins+gricean_bins
    return bins

def get_prior(lexica_list):
    """Learning prior over lexica: preference for simpler lexica"""
    concepts = [[0,0,1],[0,1,0],[0,1,1],\
                    [1,0,0],[1,0,1],[1,1,0]]
    cost = [3,8,4,4,10,5] #cost of each concept in 'concepts'
    cost = np.array([float(max(cost) - c + 1) for c in cost])
    concept_prob = cost / np.sum(cost)
    
    out = []
    for lex_idx in xrange(len(lexica_list)):
        current_lex = np.transpose(lexica_list[lex_idx])
        lex_val = 1 #probability of current lexicon's concepts
        for concept_idx in xrange(len(current_lex)):
            lex_val *= concept_prob[concepts.index(list(current_lex[concept_idx]))]
        out.append(lex_val)
    out = out + out #double for two types of linguistic behavior: literal and Gricean
    return np.array(out) / np.sum(out)
