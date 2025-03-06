import matplotlib.pyplot as plt
import networkx as nx
import copy
import dgl
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd
import seaborn as sns

def make_boxplots_with_stripplots(data, 
                 plot_extent=None,
                 figsize=None, 
                 boxcolor=(0.0, 0.5, 0.0), 
                 outlier_quantile=.1, 
                 add_vline=False,
                 xticks=[10,0,-10],
                 plot_format='png'):
    df = pd.DataFrame(data)
    ylabel,xlabel = df.columns 

    # Create the box plot
    fig, ax = plt.subplots(figsize=(10, 3) if figsize is None else figsize)
    light_boxcolor = tuple([val + .4 for val in boxcolor])
    sns.boxplot(x=xlabel,
                y=ylabel, 
                data=df,
                ax=ax, 
                showfliers=False,
                orient='h',
                boxprops=dict(facecolor=light_boxcolor, color=boxcolor, alpha=.2),
                medianprops=dict(color=(.8, 0.5, 0.0), linewidth=3, solid_capstyle='projecting', alpha=.7),
                whiskerprops=dict(color=boxcolor, linewidth=1.5, alpha=.7),
                capprops=dict(color=boxcolor, linewidth=1.5, alpha=.7))

    # Overlay the strip plot
    sns.stripplot(x=xlabel, 
                  y=ylabel, 
                  data=df, 
                  ax=ax, 
                  color='black', 
                  alpha=0.5, 
                  jitter=True,
                  orient='h')


   
    if add_vline:
        ax.vlines(0,ymin=-.5, ymax=len(set(df[ylabel])) -.5, linestyles='--', color='black', alpha=.5)

    # Remove upper, lower, and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(xticks)
    if plot_extent is not None:
        # Save the box plot as a file
        fig.savefig(f"../pics/boxplot_all_rels_{plot_extent}.{plot_format}", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def rescale_score_by_abs(
        score,
        max_score,
        min_score,
        margin=.2
):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score),
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """

    # CASE 1: positive AND negative scores occur --------------------
    if max_score > 0 and min_score < 0:

        if max_score >= abs(min_score):  # deepest color is positive
            if score >= 0:
                return 0.5 + (0.5 - margin/2) * (score / max_score)
            else:
                return 0.5 - (0.5 - margin/2) * (abs(score) / max_score)

        else:  # deepest color is negative
            if score >= 0:
                return 0.5 + (0.5 - margin/2) * (score / abs(min_score))
            else:
                return 0.5 - (0.5 - margin/2) * (score / min_score)

                # CASE 2: ONLY positive scores occur -----------------------------
    elif max_score > 0 and min_score >= 0:
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + (0.5 - margin/2) * (score / max_score)

    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score <= 0 and min_score < 0:
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - (0.5 - margin/2) * (score / min_score)


def getRGB(c_tuple):
    return "#%02x%02x%02x" % (int(c_tuple[0] * 255), int(c_tuple[1] * 255), int(c_tuple[2] * 255))


def backcolor_text(txt, score, colormap=None):
    if colormap is None : colormap = plt.get_cmap('bwr')
    return "<span style=\"font-family:Courier; font-weight:bold; background-color:" + getRGB(colormap(score)) + "\">" + txt + "</span>"


def html_heatmap(words, scores, cmap_name="bwr"):
    """
    Return word-level heatmap in HTML format,
    with words being the list of words (as string),
    scores the corresponding list of word-level relevance values,
    and cmap_name the name of the matplotlib diverging colormap.
    """

    colormap = plt.get_cmap(cmap_name)

    assert len(words) == len(scores)
    max_s = max(scores)
    min_s = min(scores)

    output_text = ""

    for idx, w in enumerate(words):
        score = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + backcolor_text(w, score, colormap) + " "

    return output_text + "\n"


def make_text_string(lsent):
    sentence = ''
    # import pdb; pdb.set_trace()

    for i, token in enumerate(lsent):
        if token not in ['&not;(',',', '.', ';', ':', "n't", "'s", "''", "'d", "'re", "'m'", "s", "'"] \
         and (i != 0) \
         and (sentence[-2:] != "``") \
         and ('##' != token[:2]):
            sentence += ' '
        elif '##' == token[:2]:
            token = token[2:]

        sentence += token

    return sentence


def vis_barh_query(atts, filename=False, stackto=None, xlim=None):
    red_color = (1,0.6,0.6)
    blue_color = (.6,.6,1)
    green_color = (.6, .8, .6)

    fig, ax = plt.subplots(figsize=(3,len(atts)))
    for key, val in list(atts.items())[::-1]:
        ax.barh(key,
                width =val,
                color=red_color if val>0 else blue_color
                , edgecolor='black', linewidth=1.2)
        if stackto is not None:
            ax.barh(key,
                    width =stackto-val,
                    left=val,
                    color=green_color
                    , edgecolor='black', linewidth=1.2)

    ax.vlines(x=0, ymin=-.6, ymax= len(atts) -.4, color='black', ls='solid')
    if xlim:
        ax.set_xlim(xlim)
        # ax.vlines(x=xlim[1], ymin=-.6, ymax= len(atts) -.4, color='black', ls='--')
    for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)

    # plt.xlim([-1,1])
#         plt.xticks([-1,0,1], ['-100\%', '0', '100\%'])
    plt.xticks([])
    if filename:
        plt.savefig(filename, transparent=True, dpi=300, bbox_inches = "tight")
    plt.show()


def vis_tree_heat(tree, node_heat, vocab_words, node_labels=None, save_dir=None, word_dist=50, node_size=2000):
    plt.figure(figsize=[12, 7])

    G = dgl.to_networkx(tree)
    G = G.to_undirected()
    pos = graphviz_layout(G, prog="dot")

    # nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_heat,
                           edgecolors='black',
                           node_size=node_size)

    # labels
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos,
                                labels=node_labels,
                                font_size=16)

    # edges
    nx.draw_networkx_edges(G, pos)

    # words
    mask = tree.ndata['mask']
    leaf_nodes = mask.nonzero().squeeze().numpy()
    pos_ = copy.copy(pos)

    for ln in leaf_nodes: pos_[ln] = (pos_[ln][0], pos_[ln][1] - word_dist)

    input_ids = tree.ndata['x']
    nx.draw_networkx_labels(G, pos_,
                            labels={ln: vocab_words[idw] for ln, idw in zip(leaf_nodes, input_ids[mask == 1])},
                            font_size=16)

    plt.axis('off')
    plt.tight_layout()
    plt.margins(y=.3, x=.0)
    if save_dir is not None:
        plt.savefig(save_dir, transparent=True)
    plt.show()




def make_color(rel, scaling=1.):
    # Scaling.
    rel *= scaling

    # Clipping.
    if rel > 1:
        rel = 1.
    elif rel < -1:
        rel = -1.

    if rel < 0:  # Blue
        color = (1.-abs(rel), 1.-abs(rel), 1.)
    else:  # Red
        color = (1,  1.-abs(rel), 1.-abs(rel))

    return color

def plot_table(table, subsets, tokens, values):
    num_subsets, N = table.shape

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(len(tokens)*2, len(subsets)))

    # Plot histograms
    for j in range(num_subsets):
        value = values[j]
        color = 'red' if value >= 0 else 'blue'
        ax[0].barh(j, value, color=color, edgecolor='black', alpha=.5)

    ax[0].set_xlim(-0.5, num_subsets - 0.5)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Set the bottom and top spines invisible
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    # Set the Y-axis tick label font size
    ax[0].tick_params(axis='y', labelsize=20)

    # draw horizontal line
    ax[0].vlines(0, -0.5, num_subsets - 0.5, colors='black', lw=1)

    # Plot the table
    for i in range(N):
        for j in range(num_subsets):
            color = 'green' if table[j, i] == 1 else 'white'
            rect = plt.Rectangle([i, j],1,  1, facecolor=color, edgecolor='black', alpha=.6)
            ax[1].add_patch(rect)

    # Set the x and y ticks
    ax[1].set_yticks([])
    ax[1].set_xticks([])

    # Set the limits and aspect ratio
    ax[1].set_ylim(0, num_subsets)
    ax[1].set_xlim(0, N)
    ax[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('intermediate_results/fig1_multi_order_subsets.png', transparent=True)
    plt.show()
