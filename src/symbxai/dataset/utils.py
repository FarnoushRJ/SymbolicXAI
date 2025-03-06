from copy import copy
from symbxai.visualization.utils import  make_text_string
from dgl.data import SSTDataset
import datasets, torchvision, dgl, random
import networkx as nx


def load_sst_treebank(sample_range, mode='train', verbose=True):
    # from dgl.data import SSTDataset
    sample_range = list(sample_range)
    dataset = SSTDataset(mode=mode)
    vocab_words = list(dataset.vocab.keys())
    sst_dataset = {mode: {
            'sentence': {},
            'label'   : {}
    }}
    # Preprocess the data into the desired format
    for sid in copy(sample_range):
        tree = dataset[sid]
        input_ids = tree.ndata['x']
        mask = tree.ndata['mask']
        lsent =  [vocab_words[idw] for idw in input_ids[mask == 1]]

        if int(tree.ndata['y'][0].item()) == 2:
            if verbose: print(f'we skip sample {sid}')
            sample_range.remove(sid)
            continue
        target = int(tree.ndata['y'][0]>2)
        sentence =  make_text_string(lsent)

        sst_dataset[mode]['sentence'][sid] = sentence
        sst_dataset[mode]['label'][sid]    = target
    return sst_dataset

def load_imdb_dataset(sample_range):
    dataset = datasets.load_dataset("imdb")['test']
    # Process the dataset
    # We should suffle the datapoints
    sentences, labels = dataset['text'],dataset['label']
    combined = list(zip(sentences,labels))
    random.seed(1)
    random.shuffle(combined)
    sentences, labels = zip(*combined)
    dataset = {'sentence': {idx: sentences[idx] for idx in sample_range} , 'label': {idx: labels[idx] for idx in sample_range} }

    return dataset

def load_fer_dataset(sample_range, processor, data_dir, image_transforms=None):
    from PIL import ImageFile, Image
    import pandas as pd
    # Define the list of file names
    from pathlib import Path

    if image_transforms is None:
        crop_size = 224
        size = int((256 / 224) * crop_size)
        image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=size, interpolation=3),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

    label_to_id = {'sad': 0, 'disgust': 1, 'angry': 2, 'neutral': 3, 'fear': 4, 'surprise': 5, 'happy': 6}
    file_names = []
    labels = []

    # Iterate through all image files in the specified directory
    all_files = sorted((Path(data_dir).glob('*/*.*')))
    assert len(all_files)> 0, f'Sorry, there are no files at {data_dir}.'
    for file in all_files:
        # check number of such files in a directory
        sample_dir = '/'.join(str(file).split('/')[:-1])+'/'

        label = str(file).split('/')[-2]  # Extract the label from the file path
        labels.append(label)  # Add the label to the list
        file_names.append(str(file))  # Add the file path to the list

    # Create a pandas dataframe from the collected file names and labels
    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
    dataset = {'image': {}, 'label': {}}
    for i in sample_range:
        path = df.iloc[[i]]['image'][i]
        image = image_transforms(Image.open(path).convert("RGB"))
        dataset['image'][i] = image
        label = label_to_id[df.iloc[[i]]['label'][i]]
        dataset['label'][i] = label
    return dataset

def test_contr_conj(tree, vocab_words, verbose=False):
    input_ids = tree.ndata['x'] # word id of the node
    labels = tree.ndata['y'] #  label of the node
    mask = tree.ndata['mask'] # 1 if the node is a leaf, otherwise 0
    adj= tree.adj()


    lsent = [vocab_words[idw] for idw in input_ids[mask == 1]]
    sentence = make_text_string(lsent)
    sent_label = labels[0]

    if 'but' in lsent:
        # find the index of 'but' in the tree
        but_treeid = (input_ids == 70).nonzero() # word index of 'but' is 70
        but_sentid = lsent.index('but')
        if len(but_treeid) != 1: return False # skip the sample if there are multiple 'but's
        but_treeid = but_treeid.item()
        assert vocab_words[input_ids[but_treeid]] == 'but'

        # Find the subsentences
        # Find parent node of 'but'
        G = dgl.to_networkx(tree)
        but_parent = list(G.successors(but_treeid))
        assert len(but_parent) == 1, f'sorry parents of "but" are {but_parent}'
        but_parent = but_parent[0]

        # find the subsentence of the parent node of but;
        Xnids = nx.ancestors(G, but_parent)
        Xnids = [nid for sid, nid in enumerate(mask.nonzero().squeeze().numpy()) if nid in Xnids and nid != but_treeid]
        Ynids = [nid for sid, nid in enumerate(mask.nonzero().squeeze().numpy()) if nid not in Xnids and nid != but_treeid]


        # exclusion criterium 1: "X but Y" doesn't reconstructs the whole sentence
        if len(Xnids + [but_treeid] + Ynids) != len(mask.nonzero().squeeze().numpy()):
            return False

        # exclusion criterium 2: "X" and "Y" are not contrastive
        # find node for subsentence X
        but_parent_leafs = list(G.reverse(copy=True).neighbors(but_parent))
        assert len(but_parent_leafs) == 2 and but_treeid in but_parent_leafs
        X_treeid = but_parent_leafs[0] if but_parent_leafs[0] !=but_treeid else but_parent_leafs[1]

        # find node for subentence Y
        # travel one node higher in the tree
        Xb_parent = list(G.successors(but_parent))
        if len(Xb_parent) != 1: return False
        Xb_parent = Xb_parent[0]
        Xb_parent_leafs = list(G.reverse(copy=True).neighbors(Xb_parent))
        assert len(Xb_parent_leafs) == 2 and but_parent in Xb_parent_leafs
        Y_treeid = Xb_parent_leafs[0] if Xb_parent_leafs[0] != but_parent else Xb_parent_leafs[1]

        ## if the words that correspond to the nodes for X and Y are empty, just skip
        Y_words = [vocab_words[input_ids[nid]] for nid in nx.ancestors(G, Y_treeid) if nid in mask.nonzero().squeeze().numpy() ]
        X_words = [vocab_words[input_ids[nid]] for nid in nx.ancestors(G, X_treeid) if nid in mask.nonzero().squeeze().numpy() ]
        if len(X_words) == 0 or len(Y_words) == 0:
            return False

        ## if the structure of the text is not properly separated into the structure X but Y
        if len(X_words) + len(Y_words) < len(lsent) -2:
            # note: we substract 2 because 'but' is always missing, '.' if often missing etc.
            return False

        ## if the sentiments of X and Y are the same, just skip
        if int(labels[X_treeid]) > 2 and int(labels[Y_treeid]) < 2:
            pass
        elif int(labels[X_treeid]) < 2 and int(labels[Y_treeid]) > 2:
            pass
        elif (int(labels[X_treeid]) == 2) != (int(labels[Y_treeid]) == 2):
            pass
        else:
            return False

        ## if the sentiment of the full sentence and of Y are not the same, just skip
        if int(labels[Y_treeid]) >=2 and int(labels[0].item()) >= 2:
            pass
        elif int(labels[Y_treeid]) <=2 and int(labels[0].item()) <= 2:
            pass
        else:
            return False

        if verbose:
            print('-------')
            print('sid:', sid)
            print('full sentence:', sentence)
            print('ground truth target', labels[0].item())
            print('\n')
            print('subset X is:\n', make_text_string([vocab_words[input_ids[nid]] for nid in  Xnids ]))
            print('subsentence of X node is:\n', make_text_string([vocab_words[input_ids[nid]] for nid in nx.ancestors(G, X_treeid) if nid in mask.nonzero().squeeze().numpy() ] ) )
            print('sentiment of X is:', labels[X_treeid] )
            print('\n')
            print('subset of Y is:\n', make_text_string([vocab_words[input_ids[nid]] for nid in  Ynids ]))
            print('subsentence of Y node is:\n', make_text_string([vocab_words[input_ids[nid]] for nid in nx.ancestors(G, Y_treeid) if nid in mask.nonzero().squeeze().numpy() ] ) )
            print('sentiment of Y is:', labels[Y_treeid] )
            print('\n')

        # all good
        return Xnids, Ynids, but_treeid, mask, lsent, sentence, sent_label, labels[X_treeid], labels[Y_treeid]

    return False

def process_treeid2tokenid(indices, mask, tokens, lsent, verbose=False):
    assert len(mask.nonzero()) == len(lsent), f'len(mask.nonzero()) = {len(mask.nonzero())} != {len(lsent)} = len(indices) '
    cls_id = 0
    sep_id = len(tokens)-1

    wordInd2token = {}
    curr_tid = 1 # start with the first token
    for word_tid in range(len(lsent)):
        if tokens[curr_tid] == lsent[word_tid]:
            wordInd2token[word_tid] = [curr_tid]
            curr_tid += 1
            continue
        else:
            word_string = tokens[curr_tid].replace('##', '')
            tids = [curr_tid]
            while word_string != lsent[word_tid]:
                curr_tid += 1
                word_string += tokens[curr_tid].replace('##', '')
                tids.append(curr_tid)
                if curr_tid == sep_id :
                    raise RuntimeError(f'Not possible to parse {tokens} into {lsent}')

            wordInd2token[word_tid] = tids
            curr_tid += 1

    if verbose: print(wordInd2token)

    token_indices = []
    for ind in indices:
        word_ids = list(mask.nonzero().squeeze().numpy())
        if ind in word_ids:
            token_indices += wordInd2token[word_ids.index(ind)]

    return list(map(lambda x : int(x), token_indices))
