import numpy as np
from tqdm import tqdm
from functools import reduce



def get_node_ordering(explainer,
                    attribution_method,
                    auc_task,
                    perturbation_type,
                    verbose=False,
                    node_mapping=None,
                    add_cls=True):

    if node_mapping is None:
        node_mapping = {i:[i] for i in explainer.node_domain}

    # We implement not the exaustive search, but the local best guess search
    def local_best_guess_search(set_attribution_fct, auc_task, node_mapping):
        node_heat = np.zeros(len(node_mapping.keys()))
        growing_node_set = []

        # We want to see the development of the iteration when verbose is True
        if verbose:
            loop_wrapper = lambda iter_range: tqdm(iter_range, desc=f'{attribution_method}-{perturbation_type}-{auc_task}')
        else:
            loop_wrapper = lambda iter_range: iter_range

        for synth_heat in loop_wrapper(range(len(node_mapping.keys()), 0, -1)):
            # Test which node in this iteration is the most promising
            worst_val = -float('inf') if auc_task == 'maximize' else float('inf')

            local_node_heat = [set_attribution_fct( growing_node_set + patches ) if not (set(patches) & set(growing_node_set)) \
                                                        else worst_val \
                                                        for _, patches in node_mapping.items()]
            if auc_task == 'minimize':
                winning_node_id = np.argmin(local_node_heat)
            elif auc_task == 'maximize':
                winning_node_id = np.argmax(local_node_heat)
            else:
                raise NotImplementedError

            node_heat[winning_node_id] = synth_heat # This is a synthetic heat, so it's just the ordering we want to have
            growing_node_set += node_mapping[winning_node_id]

        assert set(growing_node_set) == set( reduce(lambda x,y: x+y, node_mapping.values())), 'we did not catch all patches'

        return node_heat

    # Generate node heat
    if attribution_method == 'random':
        node_heat = np.random.rand(len(node_mapping.keys()))

    elif attribution_method == 'LRP':
        node_heat = np.zeros(len(node_mapping.keys()))
        feat_relevance = explainer.node_relevance()
        for new_patch, patches in node_mapping.items():
            node_heat[new_patch] = sum( feat_relevance[index] for index in patches )


    elif attribution_method == 'SHAP':
        ...
    elif attribution_method == 'PredDiff':
        node_heat = []
        for patches in node_mapping.values():
            curr_rel = explainer.subgraph_relevance(explainer.node_domain) - explainer.subgraph_relevance([ ids for ids in explainer.node_domain if ids not in patches ] )
            node_heat.append(curr_rel)
        node_heat = np.array(node_heat)

    elif attribution_method == 'SymbXAI':
        if perturbation_type == 'removal':
            set_attribution_fct = lambda S: explainer.subgraph_relevance( [ idn for idn in explainer.node_domain if idn not in S ])
        elif perturbation_type == 'generation':
            if add_cls:
                set_attribution_fct = lambda S: explainer.subgraph_relevance([0]+ S)
            else:
                set_attribution_fct = lambda S: explainer.subgraph_relevance(S)
        else:
            raise NotImplementedError

        node_ordering = np.argsort(-local_best_guess_search( set_attribution_fct, auc_task, node_mapping))

    else:
        raise NotImplementedError

    if attribution_method in ['random', 'LRP', 'PredDiff', 'SHAP']: # all first order methods just generate node heat
        if perturbation_type == 'removal' and auc_task == 'minimize':
            node_ordering = np.argsort( -node_heat)
        elif perturbation_type == 'removal' and auc_task == 'maximize':
            node_ordering = np.argsort( node_heat)
        elif perturbation_type == 'generation' and auc_task == 'minimize':
            node_ordering = np.argsort( node_heat)
        elif perturbation_type == 'generation' and auc_task == 'maximize':
            node_ordering = np.argsort( -node_heat)
        else:
            raise NotImplementedError

    return node_ordering
