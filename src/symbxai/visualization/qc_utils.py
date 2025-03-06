from scipy.spatial.distance import cdist
import numpy as np
import networkx as nx

def vis_mol_2d(ax,
            atomic_numbers,
            pos,
           projdim=1,
           with_atom_id=False,
              verbose_out=False,
              with_atom_type=True):

    dists = cdist(pos,pos)
    graph = (dists<1.6).astype(float)
    graph -= np.eye(graph.shape[0])

    #project
    pos= pos[:,[d for d in [0,1,2] if d != projdim]] #{i:p[1:] for i, p in enumerate(pos)}

    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    if with_atom_id:
        names = [f"{atom_names_dict[Z.item()]}$^{i}$" for i, Z in enumerate(atomic_numbers)]
    else:
        names = [atom_names_dict[Z.item()] for i, Z in enumerate(atomic_numbers)]
    G = nx.from_numpy_array(graph)
    # plot atoms
    collection = nx.draw_networkx_nodes(G, pos, node_color="w", node_size=4000, ax=ax)
    collection.set_zorder(2.)
    # plot bonds
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color="w",
        width=5,
        style="dotted",
        node_size=700,
        ax=ax
    )

    # plot atom types
    pos_labels = pos - np.array([0.02, 0.05])
    if with_atom_type:
        nx.draw_networkx_labels(G, pos_labels, {i: name for i, name in enumerate(names)},
                        font_weight='bold', font_size=40, ax=ax)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    if verbose_out:
        return ax, G, pos
    else:
        return ax
