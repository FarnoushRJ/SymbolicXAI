from functools import reduce
from symbxai.visualization.utils import make_text_string, getRGB, rescale_score_by_abs
import matplotlib.pyplot as plt
from html2image import Html2Image
from IPython.core.display import display, HTML

def top_percent_keys(outs, pc_top):
    norm_rels = {key: abs(val) for key, val in outs.items()}
    rel_integral = sum(norm_rels.values())
    norm_rels = {key: val/rel_integral for key, val in norm_rels.items()}
    sortedargs, sortedvals = sorted(norm_rels, key=norm_rels.get)[::-1], sorted(norm_rels.values())[::-1]

    top_ids = []
    true_pc = 0
    for ids, val in zip(sortedargs, sortedvals):
        if true_pc + val < pc_top:
            top_ids.append(ids)
            true_pc += val
        else:
            break

    return top_ids

def setids2logicalANDquery(setids, tokens, mode=None,style='HTML'):
    if style == 'HTML':
        not_token = '&not;'
        and_token = " &wedge; "
    elif style == 'Latex':
        not_token = '$\\neg$'
        and_token = " $\\wedge$ "
    else:
        raise NotImplementedError

    # if type(setids[0]) == int:
    #     # query of the form I \wedge J, with I and J being indices
    #     textlist = [tokens[tid] if tid >= 0 else '&not;' + tokens[abs(tid) -1] for tid in setids]
    # elif type(setids[0]) == frozenset:
    if mode == 'neg. disj.': ### Notice that in this mode setids is not a query hash anymore. So this can not be transormed in a query function.
        query_tokenlists = []
        for cset in setids:
            if all([feat<0 for feat in cset]) and len(cset) >1:
                ctokens = [tokens[tid + len(tokens)] for tid in cset]
                if style == 'HTML':
                    ctokens[0] = '&not;(' + ctokens[0]
                elif style == 'Latex':
                    ctokens[0] = '$\\neg$(' + ctokens[0]
                else:
                    raise NotImplementedError
                ctokens[-1] += ')'
            elif all([feat<0 for feat in cset]) and len(cset) ==1:
                ctokens = [tokens[tid + len(tokens)] for tid in cset]
                if style == 'HTML':
                    ctokens[0] = '&not;' + ctokens[0]
                elif style == 'Latex':
                    ctokens[0] = '$\\neg$' + ctokens[0]
                else:
                    raise NotImplementedError


            else:
                ctokens = [tokens[tid] for tid in cset]

            query_tokenlists += [ctokens]
    else:
        # query of the form S wedge T, with S and T being sets
        query_tokenlists = [[tokens[tid] if tid >= 0 else not_token + tokens[tid + len(tokens)] for tid in ids_set] for ids_set in setids]

    textlist = [make_text_string(tokenlist) for tokenlist in query_tokenlists]
    # else:
    #     raise NotImplementedError('Something went wrong.')

    return reduce(lambda x,y: x + and_token + y , textlist)

def plot_quali_table(sentence, tokens, all_queries, vismode, nb_top=5, nb_flop=5, pc_top=10, file_str=None, fontcolor='black'):
    # process the queries into a suitable data format
    all_attributions = {mode:{ q.hash: q.attribution for q in queries} for mode,queries in all_queries.items()}
    all_str_rep = {mode:{ q.hash: q.str_rep for q in queries} for mode,queries in all_queries.items()}

    colormap = plt.get_cmap('bwr')

    html_str = ''
    html_str += f'<h2> {sentence} </h2>'
    html_str += '<table >'

    ## header
    html_str += '<tr>'
    out_color_vals    = {}
    out_keys    = {}
    for mode in all_attributions.keys():
        html_str += f'<th style="text-align: center;">{mode}</th>'
        out_color_vals[mode]    = {}

    html_str+= '</tr>'
    # collect values and queries
    top_flop_by_nb = {}
    for mode, vals in all_attributions.items():
        sort_keys = sorted(vals, key=vals.get)[::-1]

        max_score = float(vals[sort_keys[0]])
        min_score = float(vals[sort_keys[-1]])

        for _, key in enumerate(vals.keys()):
            color_val = rescale_score_by_abs(float(vals[key]), max_score, min_score )
            out_color_vals[mode][key] = color_val

    # display values and queries
    for idx in range(nb_flop + nb_top):
        html_str += '<tr>'
        perc_out_keys = {}
        for mode in all_attributions.keys():
            if idx >= len(all_attributions[mode]): continue # this means the desired table lenght is larger than the numbers of values, so just show all.
            sort_keys = sorted(all_attributions[mode], key=all_attributions[mode].get)[::-1]
            out_keys[mode] = sort_keys[:nb_top] + (sort_keys[-nb_flop:] if nb_flop>0 else [])
            key = out_keys[mode][idx]
            # significance column
            perc_out_keys[mode] = top_percent_keys(all_attributions[mode], pc_top)

            if vismode == 'top-flop percent' and key not in perc_out_keys[mode]:
                opacity = 0.7
                font_weight = 'normal'
                color = getRGB(colormap(out_color_vals[mode][key]))
            else:
                opacity = 1
                font_weight = 'bold'
                color = getRGB(colormap(out_color_vals[mode][key]))

            html_str += f'<td style="font-family:Courier; color: {fontcolor};text-align: center; opacity: {opacity}; font-weight: {font_weight};" bgcolor="{color}">{all_str_rep[mode][key]}</td>'
        html_str += '</tr>'

    html_str +='</table>'
    if file_str is not None:
        htmlimg = Html2Image()
        htmlimg.output_path= 'intermediate_results/query_search_algo'
        htmlimg.screenshot(
            html_str=html_str,
            css_str ='',
            save_as =file_str,
            size    =(800,(nb_flop + nb_top)*50)
                    )

    display(HTML(html_str))
    return out_color_vals, perc_out_keys, out_keys
