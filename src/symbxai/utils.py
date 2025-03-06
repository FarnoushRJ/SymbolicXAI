from itertools import chain, combinations
from functools import reduce
import numpy as np


def powerset(s, K=float('inf'), with_empty=False):
    '''powerset([1,2,3]) --> [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]
    if the sets have not more than K elements in it.'''

    pset = chain.from_iterable( combinations(s, r) for r in range(len(s)+1))
    if not with_empty:
        pset =  [list(elem) for elem in pset if list(elem) != []] # no empty set
    pset = [list(elem) for elem in pset if len(elem)<= K ] # No set bigger that K

    return  pset

class Query_from_hash():
    def __init__(self, 
                 promt=None,
                 concept2idx=None, 
                 bool_fct= None, 
                 str_rep=None, 
                 hash_rep=None, 
                 nb_feats=None
                 ):
       
        self.concept2idx = concept2idx
        self.str_rep  = str_rep
        self.hash = hash_rep
        
        self.nb_feats = nb_feats
        # maybe explicitly giving the subsets for which this query is true? This speeds up the attribution
        self.support = None


        if concept2idx is not None:
            self.all_concepts = list(concept2idx.keys())
            self.nb_feats = len(concept2idx)

        elif nb_feats is not None:
            self.all_concepts = list(range(nb_feats))
        else:
            raise ValueError('either nb_feats or concept2idx must be set')
        
        self.bool_fct = bool_fct
        


    def __call__(self, feat_set):
        if self.concept2idx is not None:
            # check if the set of input concepts corresponds to the concepts we know
            assert all([feature in self.all_concepts for feature in feat_set]), 'One or more concepts in the input set are unknown.'

            # Transform set of concepts into the query format
            feat_set_idx = tuple([self.concept2idx[concept] for concept in feat_set])

           
        else:
           # check if the set of input indices is within the actual range
            assert all([feature in self.all_concepts for feature in feat_set]), 'One or more feature indices in the input set are unknown.'

            # just inherit the feature set
            feat_set_idx = feat_set

        # Copmute query relevance
        return all([ any( [(I in feat_set_idx) if I >= 0 else (I + self.nb_feats not in feat_set_idx) for I in subset ]) for subset in self.hash ])
    
    def query_promt2lamb_fct(self, 
                             promt : str,
                         concept2index_set: dict):
        words = promt.split()
        assert all(word in concept2index_set.keys() or word in ['AND', 'NOT', 'OR', '(', ')', 'IMPLIES'] for word in words), 'Promt is not well formed.'

        # Transform the implication operator
        implication_idxs = [idx for idx,word in enumerate(words) if word == 'IMPLIES']
        for impl_indx in implication_idxs:
            assert words[impl_indx -1] in concept2index_set.keys() and words[impl_indx +1] in concept2index_set.keys(), "We only accept the implication between single concepts, i.e., 'A IMPLIES B' with A and B being known concepts."
            # We want to transform 'A IMPLIES B' into 'NOT A OR B', which is equivalent.
            words[impl_indx] = 'OR'
            words.insert(impl_indx-1, 'NOT')

        # Gor over each word and make a lambda function out of it
        fct_string = ''
        for word in words:
            if word == 'NOT':
                fct_string += 'not '
            elif word == 'AND':
                fct_string += ' and '
            elif word == 'OR':
                fct_string += ' or '
            elif word in ['(', ')']:
                fct_string += word
            elif word in concept2index_set.keys():
                fct_string += f'bool(set({concept2index_set[word]}) & set(feat_set))'
            else:
                ValueError(f"Unexpected word in promt: {word}") 

        return eval('lambda feat_set: '+ fct_string)
    
    def query_promt2hash(self, promt, concept2idx):
        hash = frozenset()
        words = promt.split()
        i = 0
        while i < len(words):
            word = words[i]
            if word == "NOT":
                i += 1
                if i < len(words) and words[i] in concept2idx.keys():
                    hash |= frozenset([frozenset( [concept2idx[words[i]]
                                                    - len(concept2idx) ])])
                    i += 1
                else:
                    raise ValueError(f"Unexpected word after NOT in promt: {words[i]}")
                
            elif word == 'AND':
                i += 1
                pass
            elif word in concept2idx.keys():
                hash |= frozenset([frozenset( [concept2idx[word]])])
                i += 1
            else:
                raise ValueError(f"Unexpected word in promt: {words[i]}") 
        
        return hash
    
class Query_from_promt():
    def __init__(self, 
                 promt,
                 concept2ids, 
                 str_rep=None
                 ):

        
            
        self.concept2ids = concept2ids
        self.promt = promt

        if str_rep is None:
            self.str_rep = promt
        else:
            self.str_rep  = str_rep

        self.bool_fct = self._query_promt2lamb_fct(promt=promt,
                                                  concept2index_set=concept2ids)

        self.filter_vector = None

    def __call__(self, feat_set):
        if type(feat_set[0]) == list:
            # reduce to list of indices
            feat_set = reduce(lambda x,y: x+y, feat_set)
            
        return self.bool_fct(feat_set)
    
    def get_filter_vector(self, subsets):
        if self.filter_vector is None:
            filter_vector = np.array([int(self(feat_set)) for feat_set in subsets])
            self.filter_vector = filter_vector
        
        return self.filter_vector

    def _query_promt2lamb_fct(self, 
                             promt : str,
                             concept2index_set: dict):
        words = promt.split()
        assert all(word in concept2index_set.keys() or word in ['AND', 'NOT', 'OR', '(', ')', 'IMPLIES'] for word in words), f'Promt "{promt}" is not well formed.'

        # Transform the implication operator
        implication_idxs = [idx for idx,word in enumerate(words) if word == 'IMPLIES']
        for impl_indx in implication_idxs:
            assert words[impl_indx -1] in concept2index_set.keys() and words[impl_indx +1] in concept2index_set.keys(), f"We only accept the implication between single concepts, i.e., 'A IMPLIES B' with A and B being known concepts. This is not fulfilled in {promt}"
            # We want to transform 'A IMPLIES B' into 'NOT A OR B', which is equivalent.
            words[impl_indx] = 'OR'
            words.insert(impl_indx-1, 'NOT')

        # Gor over each word and make a lambda function out of it
        fct_string = ''
        for word in words:
            if word == 'NOT':
                fct_string += 'not '
            elif word == 'AND':
                fct_string += ' and '
            elif word == 'OR':
                fct_string += ' or '
            elif word in ['(', ')']:
                fct_string += word
            elif word in concept2index_set.keys():
                fct_string += f'bool(set({concept2index_set[word]}) & set(feat_set))'
            else:
                ValueError(f"Unexpected word in promt: {word}") 

        self.lambda_string = 'lambda feat_set: '+ fct_string
        return eval(self.lambda_string)
    
class Query():
    def __init__(self, bool_fct= None, str_rep=None, hash_rep=None, nb_feats=None):
        self.bool_fct = bool_fct
        self.str_rep  = str_rep
        self.hash = hash_rep
        self.nb_feats = nb_feats
        # maybe explicitly giving the subsets for which this query is true? This speeds up the attribution
        self.support = None


    def __call__(self, feat_set: tuple[int]):
        return all([ any( [(I in feat_set) if I >= 0 else (I + self.nb_feats not in feat_set) for I in subset ]) for subset in self.hash ])

    def get_support(self):
        if self.support is not None:
            return self.support
        else:
            return False

    def set_support(self, sets, node_domain=None, max_order=5 ):
        self.support = sets