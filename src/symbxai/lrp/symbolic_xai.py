import torch
from functools import reduce
import numpy as np
from ..model.transformer import ModifiedBertForSequenceClassification
from ..model.vision_transformer import ModifiedViTForImageClassification
import schnetpack as spk
from symbxai.utils import powerset
from torch_geometric.nn import Sequential


class Node:
    """
    Helping class for the explainer functions.
    """

    def __init__(
            self,
            node_rep,
            lamb,
            parent,
            R,
            domain_restrict=None
    ):
        self.node_rep = node_rep
        self.parent = parent
        self.R = R
        self.lamb = lamb
        self.domain_restrict = domain_restrict

    def neighbors(self):
        neighs = list(self.lamb[self.node_rep].nonzero().T[0].numpy())
        if self.domain_restrict is not None:
            neighs = [n for n in neighs if n in self.domain_restrict]
        return neighs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"node representation: {self.node_rep}; parent : {self.parent}"

    def is_root(self):
        return self.parent is None

    def get_walk(self):
        curr_node = self
        walk = [self.node_rep]
        while curr_node.parent is not None:
            curr_node = curr_node.parent
            walk.append(curr_node.node_rep)
        return tuple(walk)


def attribute(qk,pow_s, qks, explainer, eta_mode='shap'):
    out = 0.
    for l in pow_s:
        flat_l = [item for sublist in l for item in sublist]
        harsanyi_div = explainer.harsanyi_div(l)
        if eta_mode == 'shap':
            eta_l = 1/sum([float(q(l)) for q in qks.values()])
        else:
            # This is actually PreDiff
            eta_l = 1.

        out += eta_l * harsanyi_div * float(qk(l))
    return out.item()


class SymbXAI:
    def __init__(
            self,
            layers,
            x,
            num_nodes,
            lamb,
            R_T=None,
            batch_dim=False,
            scal_val=1.,
            start_subgraph_at=None
    ):
        """
        Init function. It basically sets some hyperparameters and saves the activations.
        """
        # Save some parameter.
        self.layers = layers
        self.num_layer = len(layers)
        self.node_domain = list(range(num_nodes))
        self.num_nodes = num_nodes
        self.batch_dim = batch_dim
        self.scal_val = scal_val

        # Some placeholder parameter for later.
        self.tree_nodes_per_act = None
        self.walk_rels_tens = None
        self.walk_rels_list = None
        self.node2idn = None
        self.walk_rel_domain = None
        self.walk_rels_computed = False
        self.start_subgraph_at = start_subgraph_at

        # Set up activations.
        self.xs = [x.data]
        for layer in layers:
            # print(x)
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
            self.xs.append(x.data)


        # Set up for each layer the dependencies.
        if isinstance(lamb, list):
            self.lamb_per_layer = lamb
        else:
            self.lamb_per_layer = [lamb for _ in range(self.num_layer - 1)] + [torch.ones(self.num_nodes).unsqueeze(0)]

        # Initialize the relevance.
        if R_T is None:
            self.R_T = self.xs[-1].data.detach()
        else:
            self.R_T = R_T

    def _relprop_standard(
            self,
            act,
            layer,
            R,
            node_rep
    ):
        """
        Just a vanilla relevance propagation strategy per layer guided along the specified nodes.
        """
        act = act.data.requires_grad_(True)

        # Forward layer guided at the node representation.
        # z = layer(act)[node_rep] if not self.batch_dim else layer(act)[0, node_rep]
        z = layer(act)
        if isinstance(z, tuple):
            z = z[0][node_rep] if not self.batch_dim else z[0][0, node_rep]
        else:
            z = z[node_rep] if not self.batch_dim else z[0, node_rep]

        assert z.shape == R.shape, f'z.shape {z.shape}, R.shape {R.shape}'

        s = R / z
        (z * s.data).sum().backward(retain_graph=True)
        c = torch.nan_to_num(act.grad)
        R = act * c

        return R.data

    def _update_tree_nodes(
            self,
            act_id,
            R,
            node_rep,
            parent,
            domain_restrict=None
    ):
        """
        Update the nodes in the internal dependency tree, by the given hyperparameter.
        """
        lamb = self.lamb_per_layer[act_id - 1]
        self.tree_nodes_per_act[act_id] += [Node(node_rep,
                                                 lamb,
                                                 parent,
                                                 R[node_rep] if not self.batch_dim else R[0, node_rep],
                                                 domain_restrict=domain_restrict)]

    def _setup_walk_relevance_scores(
            self,
            domain_restrict=None,
            verbose=False
    ):
        """
        To set up the relevance scores in the quality of walks. All scores will be saved at self.walk_rels_tens
        and self.walk_rels_list.
        """

        # Create data structure.
        self.tree_nodes_per_act = [[] for _ in range((self.num_layer + 1))]

        # Initialize the last relevance activation.
        self._update_tree_nodes(
            self.num_layer,
            self.R_T,
            0,
            None,
            domain_restrict=domain_restrict
        )

        for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
            # Iterate over the nodes.
            for node in self.tree_nodes_per_act[layer_id + 1]:
                # Compute the relevance.
                R = self._relprop_standard(
                    act,
                    layer,
                    node.R,
                    node.node_rep
                )

                # Distribute the relevance to the neighbors.
                for neigh_rep in node.neighbors():
                    self._update_tree_nodes(
                        layer_id,
                        R,
                        neigh_rep,
                        node,
                        domain_restrict=domain_restrict
                    )

        # save a few parameters.
        if domain_restrict is None:
            self.walk_rel_domain = self.node_domain
        else:
            self.walk_rel_domain = domain_restrict
        self.node2idn = {node: i for i, node in enumerate(self.walk_rel_domain)}
        self.walk_rels_computed = True


    def node_relevance(
            self,
            stop_at=0
    ):

        # Initialize the last relevance.
        curr_node = Node(
            0,
            self.lamb_per_layer[self.num_layer - 1],
            None,
            self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
            domain_restrict=None
        )

        for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:

            # Iterate over the nodes.
            R = self._relprop_standard(
                act,
                layer,
                curr_node.R,
                curr_node.node_rep
            )

            # Create new nodes
            new_node = Node(
                self.node_domain,
                self.lamb_per_layer[layer_id - 1],
                curr_node,
                R[self.node_domain] if not self.batch_dim else R[0, self.node_domain],
                domain_restrict=None
            )

            curr_node = new_node
            if layer_id <= stop_at: break

        node_rel = curr_node.R.sum(-1) * self.scal_val

        return node_rel


    def symb_or(self,
        featset,
        context = None
    ):
        if context is None:
            context = self.node_domain

        return self.subgraph_relevance( context ) - \
                self.subgraph_relevance(
                        list(set(context) - set(featset)))


    def symb_not(self,
        featset,
        context=None):

        if context is None:
            context = self.node_domain

        return self.subgraph_relevance( context ) - \
                self.symb_or(featset, context=context)

    def symb_and(self,
        featset,
        context=None
    ):
        assert len(featset) <=3, 'Sorry, the "and" operator for more than 3 ' \
                            +'elements is not implemented yet!'

        if len(featset) <= 1:
            if type(featset[0]) == list:
                s = featset[0]
            else:
                s = featset

            return self.symb_or(s,context=context)

        elif  len(featset) == 2:
            if type(featset[0]) == list and type(featset[1]) == list:

                s1, s2 = featset[0], featset[1]
                featset = s1 + s2
            else:
                s1, s2 = [featset[0]], [featset[1]]


            return self.symb_or(s1,context=context) \
                    + self.symb_or(s2,context=context)  \
                    - self.symb_or(featset,context=context)
        elif len(featset) == 3:
            if type(featset[0]) == list and \
            type(featset[1]) == list and \
            type(featset[2]) == list :

                s1, s2, s3 = featset[0], featset[1], featset[2]
                featset = s1 + s2 + s3
            else:
                s1, s2, s3 = [featset[0]], [featset[1]], [featset[2]]

            return self.symb_and( [s1, s2], context=context) \
                    + self.symb_and( [s2, s3], context=context) \
                    + self.symb_and( [s1, s3] , context=context) \
                    - self.symb_or(s1, context=context) \
                    - self.symb_or(s2, context=context)  \
                    - self.symb_or(s3, context=context) \
                    + self.symb_or(featset, context=context)

    def harsanyi_div(self,
                    featset,
                    dynamic_prog=False):
        """
        featset: We expect this to be a list of lists with featute indices given.
        """

        out = 0.
        pow_featset = powerset(featset)
        for subset in pow_featset:
            if type(subset[0]) == list:
                # It's a list of lists, please flatten.
                flat_subset = [ idt for set in subset for idt in set ]
            else: # It's just a list of indices
                flat_subset = subset
            subset_val = self.subgraph_relevance(flat_subset)
            out += (-1)**(len(featset) - len(subset)) * subset_val
        return out

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False,
            dynamic_prog=False
            # start_subgraph_at=None
    ):

        if type(subgraph) != list:
            subgraph = list(subgraph)
        assert len(set(subgraph)) == len(subgraph), 'We have dublicates in the subset.'

        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:

            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                if self.start_subgraph_at is not None and layer_id > self.start_subgraph_at:
                    curr_subgraph = self.node_domain
                else:
                    curr_subgraph = subgraph
                # Create new subgraph nodes.
                new_node = Node(curr_subgraph,
                                self.lamb_per_layer[layer_id - 1],
                                curr_subgraph_node,
                                R[curr_subgraph] if not self.batch_dim else R[0, curr_subgraph],
                                domain_restrict=None
                                )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val

    def walk_relevance(self, verbose=False, rel_rep='list'):
        """
        An interface to reach for the relevance scores of all walks.
        """

        if not self.walk_rels_computed:
            if verbose:
                print('setting up walk relevances for the full graph.. this may take a wile.')
            self._setup_walk_relevance_scores()

        # Just return all walk relevances.
        if rel_rep == 'tens':
            # Ask for tensor representation.
            if self.walk_rels_tens is None:  # Not prepared yet.
                self.walk_rels_tens = torch.zeros((len(self.walk_rel_domain),) * len(self.layers))
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()

                    walk_idns = tuple([self.node2idn[idn] for idn in walk])
                    self.walk_rels_tens[walk_idns] = rel * self.scal_val

            return self.walk_rels_tens, self.node2idn
        elif rel_rep == 'list':  # Ask for list representation.
            if self.walk_rels_list is None:  # Not prepared yet.
                self.walk_rels_list = []
                for node in self.tree_nodes_per_act[0]:
                    walk, rel = node.get_walk()[:len(self.layers)], node.R.data.sum()
                    self.walk_rels_list.append((walk, rel * self.scal_val))

            return self.walk_rels_list


# class TransformerSymbXAI(SymbXAI):
#     def __init__(
#             self,
#             sample,
#             target,
#             model,
#             embeddings,
#             scal_val=1.,
#             start_subgraph_at=None
#     ):
#         model.zero_grad()

#         # Prepare the input embeddings.
#         x = embeddings(
#             input_ids=sample['input_ids'],
#             token_type_ids=sample['token_type_ids']
#         )

#         # Make the model explainable.
#         modified_model = ModifiedTinyTransformerForSequenceClassification(
#             model,
#             order='first'
#         )

#         if len(x.shape) >= 3:
#             batch_dim = True
#             num_tokens = x.shape[1]
#         else:
#             batch_dim = False
#             num_tokens = x.shape[0]

#         lamb = torch.ones((num_tokens, num_tokens))
#         lamb_last_layer = torch.zeros((num_tokens, num_tokens))

#         layers = []
#         for layer in modified_model.bert.encoder.layer:
#             layers.append(layer)

#         def output_module(hidden_states):
#             pooled_data = modified_model.bert.pooler(hidden_states)

#             output = (modified_model.classifier(pooled_data) * target).sum().unsqueeze(0).unsqueeze(0)
#             return output

#         layers.append(output_module)

#         lamb_last_layer[0, :] = torch.ones(num_tokens)
#         lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]


#         super().__init__(
#             layers,
#             x.data,
#             num_tokens,
#             lambs,
#             R_T=None,
#             batch_dim=batch_dim,
#             scal_val=scal_val,
#             start_subgraph_at=start_subgraph_at
#         )

#     def subgraph_relevance(
#             self,
#             subgraph,
#             from_walks=False
#     ):
#         if type(subgraph) != list:
#             subgraph = list(subgraph)
#         assert len(set(subgraph)) == len(subgraph), 'We have dublicates in the subset.'
#         # TODO: Change the code for from_walks=True
#         if from_walks:
#             if self.walk_rels_tens is None:
#                 _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

#             # Transform subgraph which is given by a set of node representations,
#             # into a set of node identifications.
#             subgraph_idn = [self.node2idn[idn] for idn in subgraph]

#             # Define the mask for the subgraph.
#             m = torch.zeros((self.walk_rels_tens.shape[0],))
#             for ft in subgraph_idn:
#                 m[ft] = 1
#             ms = [m] * self.num_layer

#             # Extent the masks by an artificial dimension.
#             for dim in range(self.num_layer):
#                 for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
#                     ms[dim] = ms[dim].unsqueeze(unsqu_pos)

#             # Perform tensor-product.
#             m = reduce(lambda x, y: x * y, ms)
#             assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

#             # Just sum the relevance scores where the mask is non-zero.
#             R_subgraph = (self.walk_rels_tens * m).sum()

#             return R_subgraph * self.scal_val
#         else:
#             # Initialize the last relevance.
#             curr_subgraph_node = Node(
#                 0,
#                 self.lamb_per_layer[self.num_layer - 1],
#                 None,
#                 self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
#                 domain_restrict=None
#             )

#             for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
#                 if self.start_subgraph_at is not None and layer_id > self.start_subgraph_at:
#                     curr_subgraph = self.node_domain
#                 else:
#                     curr_subgraph = subgraph
#                 # Iterate over the nodes.
#                 R = self._relprop_standard(act,
#                                            layer,
#                                            curr_subgraph_node.R,
#                                            curr_subgraph_node.node_rep)

#                 if layer_id == 3:
#                     # Create new subgraph nodes.
#                     new_node = Node(0,
#                                     self.lamb_per_layer[layer_id - 1],
#                                     curr_subgraph_node,
#                                     R[0] if not self.batch_dim else R[0, 0],
#                                     domain_restrict=None
#                                     )
#                 else:
#                     # Create new subgraph nodes.
#                     new_node = Node(curr_subgraph,
#                                     self.lamb_per_layer[layer_id - 1],
#                                     curr_subgraph_node,
#                                     R[curr_subgraph] if not self.batch_dim else R[0, curr_subgraph],
#                                     domain_restrict=None
#                                     )

#                 curr_subgraph_node = new_node

#             return curr_subgraph_node.R.sum() * self.scal_val

#     def get_local_best_subgraph(
#             self,
#             alpha: float = 0.0
#     ):
#         subgraph = []
#         all_features = np.arange(self.num_nodes)

#         while len(subgraph) < self.num_nodes:
#             feature_list = list(frozenset(all_features).difference(frozenset(subgraph)))
#             max_score = -float("inf")
#             max_feature = None

#             graph_score = self.subgraph_relevance(subgraph=all_features, from_walks=False)

#             for feature in feature_list:
#                 # s = subgraph + [feature]
#                 s = list(frozenset(all_features).difference(frozenset(subgraph + [feature])))

#                 # mask = torch.full([self.num_nodes], alpha)
#                 # mask[list(s)] = 1.0
#                 # mask = torch.diag(mask)

#                 temp_score = -np.abs(self.subgraph_relevance(subgraph=list(s), from_walks=False) - graph_score)

#                 if temp_score > max_score:
#                     max_score = temp_score
#                     max_feature = feature

#             subgraph += [max_feature]

#         best_subgraph = torch.full((self.num_nodes, ), -1)
#         for i, f in enumerate(subgraph):
#             best_subgraph[f] = self.num_nodes - i

#         return best_subgraph


class BERTSymbXAI(SymbXAI):
    def __init__(
            self,
            sample,
            target,
            model,
            embeddings,
            scal_val=1.,
            use_lrp_layers=True,
            gam=0.15
    ):
        model.zero_grad()

        # Prepare the input embeddings.
        x = embeddings(
            input_ids=sample['input_ids'],
            token_type_ids=sample['token_type_ids']
        )

        # Make the model explainable.
        if use_lrp_layers:
            modified_model = ModifiedBertForSequenceClassification(
                model,
                gam=gam
            )

        else:
            modified_model = model

        if len(x.shape) >= 3:
            batch_dim = True
            num_tokens = x.shape[1]
        else:
            batch_dim = False
            num_tokens = x.shape[0]

        lamb = torch.ones((num_tokens, num_tokens))
        lamb_last_layer = torch.zeros((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.bert.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            pooled_data = modified_model.bert.pooler(hidden_states)
            logits = modified_model.classifier(pooled_data)
            output = (logits * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)

        lamb_last_layer[0, :] = torch.ones(num_tokens)
        lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]

        super().__init__(
            layers,
            x.data,
            num_tokens,
            lambs,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        # TODO: Change the code for from_walks=True
        assert len(set(subgraph)) == len(subgraph), 'We have dublicates in the subset.'
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:
            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                if layer_id == 12:
                    # Create new subgraph nodes.
                    new_node = Node(0,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0],
                                    domain_restrict=None
                                    )
                else:
                    # Create new subgraph nodes.
                    new_node = Node(subgraph,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[subgraph] if not self.batch_dim else R[0, subgraph],
                                    domain_restrict=None
                                    )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val

    # def subgraph_shap(
    #         self,
    #         subgraph
    # ):
    #     out = self.xs[0][subgraph].unsqueeze(0) if not self.batch_dim else self.xs[0][0, subgraph].unsqueeze(0)
    #     for layer in self.layers:
    #         out = layer(out)[0]

    #     return out * self.scal_val

    # def subgraph_shap_masked(
    #         self,
    #         subgraph,
    #         unk_embedding
    # ):
    #     # [CLS] token should be always included.
    #     if 0 not in subgraph:
    #         subgraph += [0]
    #     subgraph_comp = list(set(self.node_domain) - set(subgraph))

    #     if not self.batch_dim:
    #         self.xs[0][subgraph_comp] = unk_embedding
    #     else:
    #         self.xs[0][0, subgraph_comp] = unk_embedding

    #     out = self.xs[0].unsqueeze(0)
    #     for layer in self.layers:
    #         out = layer(out)[0]

    #     return out * self.scal_val

    # def symb_or_shap_masked(
    #         self,
    #         featset,
    #         context=None
    #         ):
    #     if context is None:
    #         context = self.node_domain

    #     return self.subgraph_shap_masked(context) - \
    #            self.subgraph_shap_masked(
    #                list(set(context) - set(featset)))

    # def symb_or_shap(
    #         self,
    #         featset,
    #         context=None
    #         ):
    #     if context is None:
    #         context = self.node_domain

    #     return self.subgraph_shap(context) - \
    #            self.subgraph_shap(
    #                list(set(context) - set(featset)))

######################
#   Mutagenicity    #
#####################
from symbxai.lrp.explain_mutag import get_model_temporary_res, lrp_linear, lrp_gconv
from torch_geometric.utils import to_dense_adj


class MutagenicitySymbXAI(SymbXAI):
    def __init__(self,
            sample,
            model,
            gamma=0.1,
            scal_val=1.,
            target_class= None,
            debug = False ):
        
        self.sample = sample
        self.model = model
        self.gamma = gamma
        self.scal_val = scal_val 
        self.debug = debug
        self.node_domain = list(range(sample.x.shape[0]))

        self.x = self.sample.x
        self.edge_index = self.sample.edge_index
        self.module_temporary_res, self.linear_out = get_model_temporary_res(model, self.x, self.edge_index)

        # init relevance -> pass the read-out step
        if target_class is None:
            # self.target_class = model.forward(self.x, self.edge_index).argmax().item()
            R_init = torch.zeros_like(self.linear_out)
            R_init[:, 0] = -self.linear_out[:, 0]
            R_init[:, 1] = self.linear_out[:, 1]

        else:
            self.target_class = target_class
            R_init = torch.zeros_like(self.linear_out)
            R_init[:, self.target_class] = self.linear_out[:, self.target_class]

        # model.linear 
        self.R_linear = lrp_linear(x = self.module_temporary_res[5]['out2'], 
                        W = model.linear.weight.T, 
                        b = model.linear.bias,
                        R = R_init,
                        rule = 'gamma',
                        gamma = 0.02,
                        debug = self.debug)
        
    def subgraph_relevance(self, subgraph):
        # subgraph_rel(x, edge_index, R_linear, S, , gamma=0.2, debug=False)

        rule_linear='gamma'

        A = to_dense_adj(self.edge_index).squeeze()
        A += torch.eye(A.shape[0]) # add selfloop

        R = self.R_linear

        for step, layer in enumerate(range(len(self.module_temporary_res)-2,0,-2)):
            R_ = torch.zeros_like(R)
            R_[subgraph] = R[subgraph]
            R = R_ 

            R = lrp_linear(x = self.module_temporary_res[layer]['out1'], 
                        W = self.module_temporary_res[layer]['lin2'].weight.T, 
                        b = self.module_temporary_res[layer]['lin2'].bias,
                        R = R, rule=rule_linear, gamma=self.gamma, debug=self.debug)
            R = lrp_linear(x = self.module_temporary_res[layer]['out'], 
                        W = self.module_temporary_res[layer]['lin1'].weight.T, 
                        b = self.module_temporary_res[layer]['lin1'].bias,
                        R = R, rule=rule_linear, gamma=self.gamma, debug=self.debug)
            R = lrp_gconv(x = self.module_temporary_res[layer - 1]['out2'], 
                    A = A, 
                    R = R, 
                    debug=self.debug)

        R_ = torch.zeros_like(R)
        R_[subgraph] = R[subgraph]
        R = R_ 
        R = lrp_linear(x = self.module_temporary_res[0]['out1'], 
                    W = self.module_temporary_res[0]['lin2'].weight.T, 
                    b = self.module_temporary_res[0]['lin2'].bias,
                    R = R, rule=rule_linear, gamma=self.gamma, debug=self.debug)
        R = lrp_linear(x = self.module_temporary_res[0]['out'], 
                    W = self.module_temporary_res[0]['lin1'].weight.T, 
                    b = self.module_temporary_res[0]['lin1'].bias,
                    R = R, rule=rule_linear, gamma=self.gamma, debug=self.debug)
        R = lrp_gconv(x = self.x, 
                    A = A, 
                    R = R, debug=self.debug)
        return R[subgraph].sum() * self.scal_val

   
    
        
######################
# Quantum Chemistry #
#####################
class SchNetSymbXAI(SymbXAI):
    def __init__(
        self,
        sample,
        model,
        target_property,
        xai_mod=True,
        gamma=0.1,
        cutoff=None,
        new_model=True,
        comp_domain=None,
        scal_val=1.
    ):
        model.zero_grad()  # When computing forces, the model still has the gradients.
        _, n_atoms, _, idx_i, idx_j, x, _, f_ij, rcut_ij, node_range, lamb = get_prepro_sample_qc(
            sample, model, new_model=new_model
        )

        for layer in model.representation.interactions:
            layer._set_xai(xai_mod, gamma)
        model.output_modules[0]._set_xai(xai_mod, gamma)

        layers = []
        for inter in model.representation.interactions:
            def layer(h, curr_layer=inter):
                curr_layer.zero_grad()
                return h + curr_layer(h, f_ij, idx_i, idx_j, rcut_ij)
            layers.append(layer)

        def out_layer(h):
            sample['scalar_representation'] = h
            layer = model.output_modules[0]
            layer.zero_grad()
            return layer(sample)[target_property]
        layers += [out_layer]

        super().__init__(
            layers,
            x.data,
            n_atoms,
            lamb,
            R_T=None,
            batch_dim=not new_model,
            scal_val=scal_val
        )


def get_prepro_sample_qc(
    sample,
    model,
    new_model=True,
    add_selfconn=True,
    cutoff=None
):
    if new_model:
        if spk.properties.Rij not in sample:
            model(sample)

        atomic_numbers = sample[spk.properties.Z]
        r_ij = sample[spk.properties.Rij]
        idx_i = sample[spk.properties.idx_i]
        idx_j = sample[spk.properties.idx_j]
        n_atoms = sample[spk.properties.n_atoms]

        x = model.representation.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1).float()
        f_ij = model.representation.radial_basis(d_ij)
        rcut_ij = model.representation.cutoff_fn(d_ij)

        node_range = [i for i in range(n_atoms[0])]
        lamb = torch.zeros(n_atoms[0], n_atoms[0])

        if cutoff is None:
            lamb[idx_i, idx_j] = 1
        else:
            for i, j, d in zip(idx_i, idx_j, d_ij):
                if d <= cutoff:
                    lamb[i, j] = 1

        if add_selfconn:
            lamb += torch.eye(n_atoms[0])

        return (
            atomic_numbers,
            n_atoms,
            r_ij,
            idx_i,
            idx_j,
            x,
            d_ij,
            f_ij,
            rcut_ij,
            node_range,
            lamb
        )
    else:
        atomic_numbers = sample[spk.Properties.Z]
        positions = sample[spk.Properties.R]
        cell = sample[spk.Properties.cell]
        cell_offset = sample[spk.Properties.cell_offset]
        neighbors = sample[spk.Properties.neighbors]
        neighbor_mask = sample[spk.Properties.neighbor_mask]
        atom_mask = sample[spk.Properties.atom_mask]
        n_atoms = torch.tensor(atomic_numbers.shape[1]).unsqueeze(0)

        x = model.representation.embedding(atomic_numbers)
        r_ij = model.representation.distances(
            positions,
            neighbors,
            cell,
            cell_offset,
            neighbor_mask=neighbor_mask
        )
        f_ij = model.representation.distance_expansion(r_ij)
        node_range = [i for i in range(n_atoms[0])]

        hard_cutoff_network = spk.nn.cutoff.HardCutoff(cutoff)
        lamb_raw = hard_cutoff_network(r_ij)[0]

        lamb = torch.zeros(lamb_raw.shape[0], lamb_raw.shape[1] + 1)

        for row_idx, row in enumerate(lamb_raw):
            lamb[row_idx] = torch.cat((row[:row_idx], torch.tensor([0.]), row[row_idx:]))

        if add_selfconn:
            lamb += torch.eye(n_atoms[0])

        return (
            atomic_numbers,
            n_atoms,
            r_ij,
            neighbors,
            neighbor_mask,
            f_ij,
            x,
            node_range,
            lamb
        )

# ----------------
# Vision Models
# ----------------
class ViTSymbolicXAI(SymbXAI):
    def __init__(
            self,
            sample,
            target,
            model,
            embeddings,
            scal_val=1.,
            use_lrp_layers=True,
            start_subgraph_at=None
    ):
        model.zero_grad()

        # Prepare the input embeddings.
        B = sample.shape[0]
        x = embeddings(sample)

        # Make the model explainable.
        if use_lrp_layers:
            modified_model = ModifiedViTForImageClassification(
                model
            )
        else:
            modified_model = model

        if len(x.shape) >= 3:
            batch_dim = True
            num_tokens = x.shape[1]
        else:
            batch_dim = False
            num_tokens = x.shape[0]

        lamb = torch.ones((num_tokens, num_tokens))
        lamb_last_layer = torch.zeros((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.vit.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            hidden_states = modified_model.vit.layernorm(hidden_states)
            logits = modified_model.classifier(hidden_states)[:, 0]
            output = (logits * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)

        lamb_last_layer[0, :] = torch.ones(num_tokens)
        lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]

        super().__init__(
            layers,
            x.data,
            num_tokens,
            lambs,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val,
            start_subgraph_at = start_subgraph_at
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):
        assert len(set(subgraph)) == len(subgraph), 'We have dublicates in the subset.'
        # TODO: Change the code for from_walks=True
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:

            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # fix the subgraph
                if self.start_subgraph_at is not None and layer_id > self.start_subgraph_at:
                    curr_subgraph = self.node_domain
                else:
                    curr_subgraph = subgraph
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                if layer_id == 12:
                    # Create new subgraph nodes.
                    new_node = Node(0,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0],
                                    domain_restrict=None
                                    )
                else:
                    # Create new subgraph nodes.
                    new_node = Node(curr_subgraph,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[curr_subgraph] if not self.batch_dim else R[0, curr_subgraph],
                                    domain_restrict=None
                                    )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val

    
