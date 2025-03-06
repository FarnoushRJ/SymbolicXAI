import time, numpy, torch, mpmath
from itertools import pairwise, product
from tqdm import tqdm
from symbxai.utils import powerset, Query, Query_from_promt
from symbxai.visualization.query_search import setids2logicalANDquery
from random import shuffle

from functools import reduce


import re

def generate_promts(all_input_promts,  modes, more_input_promts=None):
    # assert all([mode in [ 'negation of promts', 'conjuction between promts' ] for mode in modes])

    all_output_promts = []
    
    if 'negation of promts' in modes:
        all_output_promts += ['NOT '+ concept for concept in all_input_promts]
    
    for mode in modes:
        new_promts = []
        match = re.fullmatch('conjuction of order ([0-9]+) between promts', mode )
        if match:
            order = int(match.group(1))
            for multi_index in product(range(len(all_input_promts)), repeat=order):
                if not all( multi_index[i] < multi_index[i+1] for i in range(order-1)): continue
                new_promt = reduce( lambda promt1, promt2: f'{promt1} AND {promt2}', 
                                   [all_input_promts[i] for i in multi_index])
                
                new_promts.append(new_promt)

        all_output_promts += new_promts
    
    if 'implication between promts' in modes:
        new_promts = []
        for promt1 in all_input_promts:
            for promt2 in all_input_promts:
                if promt1 == promt2: continue 
                new_promt = f'( {promt1} IMPLIES {promt2} )'
                new_promts.append(new_promt)
        all_output_promts += new_promts
    
    if 'conjoin different promts with each other' in modes:
        new_promts = []
        for promt1 in all_input_promts:
            for promt2 in more_input_promts:
                new_promt = f'{promt1} AND {promt2}'
                

                new_promts.append(new_promt)
        all_output_promts += new_promts

    # Test for uniqueness of concepts in the promts
    logical_symbols = ['AND', 'OR', 'NOT', 'IMPLIES', '(', ')']
    all_output_promts = [
        promt for promt in all_output_promts
        if len(set([word for word in promt.split() if word not in logical_symbols])) ==
        len([word for word in promt.split() if word not in logical_symbols])
    ]

    if not all_output_promts: 
        raise ValueError(f'The specified modes {modes} do not exist.')
    
    return all_output_promts



def remove_semantic_duplicate_queries(all_promts, all_concepts, verbose=False):
    def filter_semantic_duplicate_queries(list_a, list_b):
        if len(list_a) != len(list_b):
            raise ValueError("Both lists must have the same length")
        
        seen_vectors = {}  # To track already encountered vectors and their indices
        to_remove_indices = []  # Indices to remove from list_a
        
        for i, vector in enumerate(list_b):
            vector_key = tuple(vector)  # Convert vector to tuple for hashing
            if vector_key in seen_vectors:  # Duplicate found
                original_index = seen_vectors[vector_key]
                original_string = list_a[original_index]
                duplicate_string = list_a[i]
                
                if verbose:
                    # Print both strings and indicate which one will be removed
                    print(f"Duplicate vector: {vector}")
                    print(f"Semantically same: '{original_string}' (kept) and '{duplicate_string}' (removed)")
                
                # Mark the current index for removal
                to_remove_indices.append(i)
            else:
                # Save the vector and its index as seen
                seen_vectors[vector_key] = i
        
        # Remove duplicates from list_a based on marked indices
        list_a = [string for i, string in enumerate(list_a) if i not in to_remove_indices]
        
        return list_a

    artificial_concepts2ids = {concept: [i] for i, concept in enumerate(all_concepts)}
    all_queries = [ Query_from_promt(promt=query_promt, concept2ids=artificial_concepts2ids) for query_promt in all_promts]
    artificial_substr_pset = powerset(artificial_concepts2ids.values())
    all_filtervectors = [query.get_filter_vector(artificial_substr_pset) for query in all_queries]

    filtered_promts = filter_semantic_duplicate_queries(all_promts, all_filtervectors)

    return filtered_promts

def approx_query_search(explainer,
                        tokens,
                        maxorder,
                        norm_mode='occlusion',
                        maxvalsnum=1,
                        minvalsnum=0,
                        verbose=True):
    attr = {}
    outvals = {}

    if verbose: start_t = time.time()
    for opt_fct in [max]*maxvalsnum + [min]*minvalsnum:
        all_vals = {f'{order}-order': {} for o in range(1,maxorder +1)}
        fixed_feats = ()
        for order in range(1,maxorder+1):
            if norm_mode == 'significance':
                eta = 2^( order - len(tokens) )
            elif norm_mode == 'occlusion':
                eta = 1
            else:
                raise Exception(f'{norm_mode} is not implemented at the moment')

            for I in range(1, len(tokens) -1):
                if I in fixed_feats: continue
                curr_feats = fixed_feats + (I,)
                # check repetition of previous searches:
                rep_bool = [(set(curr_feats) == set(prev_feats)) for prev_feats in outvals.keys()]
                if any(rep_bool): continue

                all_vals[f'{order}-order'][curr_feats] = explainer.symb_and(list(curr_feats))

            odict = all_vals[f'{order}-order']
            outvals[opt_fct(odict, key=odict.get)] = opt_fct(odict.values())
            fixed_feats = opt_fct(odict, key=odict.get)

    if verbose:
        ftime = time.time()-start_t
        print(f'calculation took {round(ftime, 4)} sec.')
        return outvals, all_vals, ftime
    else:
        return outvals

def exhaustive_query_search(
                        explainer,
                        tokens,
                        maxorder,
                        norm_mode='occlusion',
                        verbose=True):
    all_vals = {}
    outvals = {}
    if verbose: start_t = time.time()

    for order in range(1,maxorder +1):
        # Make the feat-grid:
        grid = [numpy.arange(0, len(tokens))]*(order)
        grid = numpy.meshgrid(*grid)
        grid = [I.flatten() for I in grid]
        all_feats = [feats for feats in zip(*grid) if all(a < b for a, b in pairwise(feats))]
        # print(all_feats)
        if norm_mode == 'significance-2':
            eta = 2^( order - len(tokens) )
        elif norm_mode == 'occlusion':
            eta = 1
        else:
            raise Exception(f'{norm_mode} is not implemented at the moment')

        for curr_feats in tqdm(all_feats):
            all_vals[curr_feats] = eta * explainer.symb_and(list(curr_feats))

    outvals[max(all_vals, key=all_vals.get)] = max(all_vals.values())

    if verbose :
        ftime = time.time() - start_t
        print(f'calculation took {round(ftime, 4)} sec.')
        return outvals, all_vals, ftime
    else:
        return outvals

def weight_query_attr_directly(all_exh_outs, modes):
    all_outs = {mode: {} for mode in modes}
    for mode in modes:
        if mode == 'occlusion':
            alpha = 0.
        elif 'significance' in mode:
            alpha = float(mode.split('-')[1])
        else:
            raise NotImplementedError(f'The weight mode {mode} is not implemented.')

        for key, val in all_exh_outs.items():
            order=len(key)
            all_outs[mode][key] = mpmath.power(2, alpha*(order - 4)) * float(val)
    return all_outs

def weight_query_attr_harsanyi(all_queries,har_div, modes):
    query_attr = {mode: {} for mode in modes}
    for weight_mode in modes:
        for q in tqdm(all_queries, desc=f'Query Attribution of {weight_mode}.'):
            if weight_mode == 'occlusion':
                query_attr[weight_mode][q.hash] = sum([ val for S, val in har_div.items() if q(S) ])

            elif 'shapley':
                attr = 0
                for S, val in har_div.items():
                    weight = sum([int(query(S)) for query in all_queries ])
                    attr += val/weight
                query_attr[weight_mode][q.hash] = attr

            else:
                raise NotImplementedError(f'Weight mode {weight_mode} is not implemented yet.')

    return query_attr

def comp_all_harsanyi_sst(explainer, harsanyi_maxorder=5, do_shuffling =False, neg_tokens=None):

    if neg_tokens is None:
        all_feats = explainer.node_domain
    else:
        all_feats = [ feat for feat in explainer.node_domain if feat not in neg_tokens ]
    power_feats = powerset(all_feats, K=harsanyi_maxorder)
    if do_shuffling: shuffle(power_feats) # only to see better in the tqdm timeline how long it takes.

    har_div = {}

    for S in tqdm(power_feats, desc='Harsanyi Dividends'):
        har_div[tuple(S)] = explainer.harsanyi_div(S)

    return har_div

def calc_weights(weight_mode, hars_div, all_queries):
    if weight_mode == 'occlusion' or 'significance' in weight_mode:
        weight_fct = lambda S : 1
    elif weight_mode == 'shapley':
        def weight_fct(S, padding_val = 0):
            nb_positive_q = sum([int(curr_q(S)) for curr_q in all_queries])
            if nb_positive_q == 0:
                return padding_val
            else:
                return 1/nb_positive_q
    else:
        raise NotImplementedError(f"Weight mode '{weight_mode}' is not implemented.")

    weight_vec =[]
    for S, _ in tqdm(hars_div.items(), desc=f'Query weights for {weight_mode}.'):
        weight_vec.append(weight_fct(S))
    weight_vec = numpy.array(weight_vec)

    return weight_vec

def queryhash2featset(query_hash, tokens):
    # if type(query_hash[0]) == frozenset:
    featsets = frozenset([frozenset([ idx if idx >= 0 else idx + len(tokens) for idx in fset ] ) for fset in query_hash])
    # else:
    #     featsets = frozenset([ idx if idx >= 0 else idx + len(tokens) for idx in query_hash ] )
    return featsets


def setup_queries(  feat_domain,
                    tokens,
                    max_and_order,
                    max_setsize= float('inf'),
                    max_indexdist=1,
                    mode='conj. disj. (neg. disj.) reasonably mixed',
                    neg_tokens=None,
                    repres_style='HTML'):

    def check_pairwise_dist(elem_list, max_indexdist):
        if len(elem_list) == 1:
            return True
        else:
            return all([ abs( elem_list[i+1] - elem_list[i]) <= max_indexdist for i in range(len(elem_list)-1) ])

    assert sorted(feat_domain) == feat_domain, 'has to be sorted'
    # make sure max_setsize is properly parsed.
    if max_setsize is None or max_setsize < 0:
        max_setsize= float('inf')

    # neglect tokens we don't want. usually the [cls] and [sep] token in the transormer models
    if neg_tokens is not None:
        feat_domain = [feat for feat in feat_domain if feat not in neg_tokens]

    all_sets = powerset(feat_domain , K=max_setsize)
    all_sets = [ fset for fset  in all_sets if check_pairwise_dist(fset, max_indexdist) ]
    if mode in ['conj. disj. neg. reasonably mixed', 'conj. disj. (neg. disj.) reasonably mixed' ]:
        all_sets += [ [idx -len(tokens) for idx in fset] for fset in all_sets] # negative indices mean negating a set.

    queries = []
    all_hashs_check = set()
    for order in range(1,max_and_order +1):
        # Make the feat-grid:
        grid = [numpy.arange(0, len(all_sets))]*(order)
        grid = numpy.meshgrid(*grid)
        grid = [I.flatten() for I in grid]
        all_psets_comb_ids = [feats for feats in zip(*grid) if all(a < b for a, b in pairwise(feats))] # all powerset combination identifiers

        for pset_comb_ids in all_psets_comb_ids:
            # specify the query
            curr_sets = frozenset([frozenset(all_sets[ids]) for ids in pset_comb_ids])



            if mode == 'set conjuction':
                str_rep = setids2logicalANDquery(curr_sets, tokens, style=repres_style)
            elif mode == 'conj. disj. reasonably mixed':
                # Check if sets are pairwise disjoint
                if sum(map(len, curr_sets)) != len(set().union(*curr_sets)): continue
                else: ...
                str_rep = setids2logicalANDquery(curr_sets, tokens, style=repres_style)

            elif mode == 'conj. disj. neg. reasonably mixed':

                featsets = queryhash2featset(curr_sets, tokens)
                if sum(map(len, featsets)) != len(set().union(*featsets)): continue
                else: ...
                str_rep = setids2logicalANDquery(curr_sets, tokens, style=repres_style)

            elif mode == 'conj. disj. (neg. disj.) reasonably mixed':
                featsets = queryhash2featset(curr_sets, tokens)
                if sum(map(len, curr_sets)) != len(set().union(*featsets)): continue
                str_rep = setids2logicalANDquery(curr_sets, tokens, mode= 'neg. disj.', style=repres_style)

                # make the frozenset of negative values into multple 1 item frozensets
                temp_sets = frozenset()
                for cset in curr_sets:
                    if all([feat >= 0 for feat in cset]):
                        temp_sets |= {cset}
                    elif all([feat < 0 for feat in cset]):
                        temp_sets |= frozenset([frozenset({feat}) for feat in cset ])
                    else:
                        raise NotImplementedError('the datastructure is wrong')
                curr_sets = temp_sets

            else:
                raise NotImplementedError(f'The mode {mode} is not implemented yet.')

            if curr_sets not in all_hashs_check:
                all_hashs_check |= {curr_sets}

                q = Query(hash_rep=curr_sets, str_rep=str_rep, nb_feats=len(tokens))
                queries.append(q)

    return queries

def m(x, w):
    """Weighted Mean"""
    return numpy.sum(x * w) / numpy.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return numpy.sum(w * (x - m(x, w)) * (y - m(y, w))) / numpy.sum(w)

def corr(x, y, w=None):
    """Weighted Correlation"""
    if type(x) == list or type(y) == list:
        x,y = numpy.array(x), numpy.array(y)
    if w is None:
        w = numpy.ones(x.shape)
    
    divisor = numpy.sqrt(cov(x, x, w) * cov(y, y, w))
    if divisor == 0:
        return 0
    else:
        return cov(x, y, w) / divisor

def calc_corr(arg):
    ''' helping function for the parallelization'''
    q, hars_div, weight_vec = arg
    supp = tuple([S for S in hars_div.keys() if q(S)])

    q_vec, val_vec = map(numpy.array, zip(*[(int(q(S)), val) for S, val in hars_div.items() ]))

    q.attribution = corr(q_vec, val_vec, weight_vec)
    q.set_support(supp)

    return q

def calc_cov(arg):
    ''' helping function for the parallelization'''
    q, hars_div, weight_vec = arg
    supp = tuple([S for S in hars_div.keys() if q(S)])

    q_vec, val_vec = map(numpy.array, zip(*[(int(q(S)), val) for S, val in hars_div.items() ]))

    q.attribution = cov(q_vec, val_vec, weight_vec)
    q.set_support(supp)

    return q

def calc_attr_supp(arg):
    ''' helping function for the parallelization'''
    q, hars_div, weight_vec = arg
    attr = 0
    supp = ()
    for weight, (S, val) in zip(weight_vec,hars_div.items()):
        if q(S):
            attr += weight * val

        #     if weight_mode == 'occlusion':
        #         attr += val
        #     elif weight_mode == 'shapley':
        #         weight = sum([int(query(S)) for query in all_queries ])
        #         attr += val/weight
        #     elif weight_mode == 'occlusion error':
        #         ...
        #     else:
        #         raise NotImplementedError(f'Weight mode {weight_mode} is not implemented yet.')
            supp += (S,)
        #
        # elif not q(S) and weight_mode == 'occlusion error':
        #     attr += val**2

    q.set_support(supp)
    q.attribution = attr
    return q
