import torch
import numpy as np
import os
import json
# from symbxai.model.gcn import GCN
from symbxai.model.gin import GIN


def load_best_model(dataset_name, models_repository='../saved_models'):
    best_acc = 0
    for file in os.listdir(f'{models_repository}/'):
        if file[:len(dataset_name)] == dataset_name:
            filename = file.split('.')[0]
            model_info = json.load(open(f'{models_repository}/{filename}.json', 'r'))
            if model_info['test_acc'] > best_acc:
                best_acc = model_info['test_acc']
                best_model_filename = filename
    model_info = json.load(open(f'{models_repository}/{best_model_filename}.json', 'r'))
    hidden_dim = model_info['hidden_dim']
    gcn_layers = model_info['gcn_layers']
    mlp_layers = model_info['mlp_layers']
    architecture = model_info['architecture']
    input_features = model_info['input_features']
    num_classes = model_info['num_classes']
    bias = model_info['bias'] if 'bias' in model_info else True
    if architecture == 'GIN':
        model = GIN(hidden_dim=hidden_dim, input_dim=input_features, gcn_layers=gcn_layers, mlp_layers=mlp_layers, nbclasses=num_classes, node_level=False, directed=False, regression=False, bias=bias)
    elif architecture == 'GCN':
        model = GCN(hidden_dim=hidden_dim, input_dim=input_features, gcn_layers=gcn_layers, nbclasses=num_classes, node_level=False, directed=False, regression=False)
    else:
        raise NotImplementedError(f'Architecture {architecture} not implemented')
    model.load_state_dict(torch.load(f'{models_repository}/{best_model_filename}.pth'))
    print(f'Loaded model {best_model_filename}.pth with test accuracy {best_acc:.4f}')
    return model


def get_masked_patch_ids(image, mask, patch_size):
    """
    Returns unique IDs of patches that include parts of the segmentation mask.

    Args:
        image (np.array): Original image array.
        mask (np.array): Binary mask array with the same height and width as the image.
        patch_size (tuple): Size of each patch (height, width).
    
    Returns:
        List[int]: List of unique integer IDs for patches that contain the mask.
    """
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    # Calculate the total number of patches along each dimension
    num_patches_y = (img_height + patch_height - 1) // patch_height
    num_patches_x = (img_width + patch_width - 1) // patch_width
    
    # Initialize list to store patch IDs containing the mask
    masked_patch_ids = []
    
    # Loop over the image in steps of the patch size
    for i in range(0, img_height, patch_height):
        for j in range(0, img_width, patch_width):
            # Calculate the current patch ID
            patch_id = (i // patch_height) * num_patches_x + (j // patch_width)
            
            # Extract the current patch from the mask
            mask_patch = mask[i:i + patch_height, j:j + patch_width]
            
            # Check if there are any mask pixels in this patch
            if np.any(mask_patch):
                # Store the patch ID
                masked_patch_ids.append(patch_id)
    
    return masked_patch_ids