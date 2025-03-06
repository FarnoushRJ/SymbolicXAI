import torch

def get_lrp_hyperparameters(rule, **kwargs):
    if rule == 'gamma':
        gamma = kwargs['gamma']
        epsilon = kwawrgs['epsilon'] if 'epsilon' in kwargs else 1e-6
        alpha = 0
        beta = 0
    elif rule == 'epsilon':
        gamma = 0
        epsilon = kwawrgs['epsilon']
        alpha = 0
        beta = 0
    elif rule == 'ab':
        gamma = 0
        epsilon = kwawrgs['epsilon'] if 'epsilon' in kwargs else 1e-6
        alpha = kwargs['alpha']
        beta = kwargs['beta']
    elif rule == 'none':
        gamma = 0
        epsilon = 0
        alpha = 0
        beta = 0
    else:
        raise NotImplementedError(f'rule {rule} is not implemented')
    return gamma, epsilon, alpha, beta

@torch.no_grad()
def lrp_linear(x, W, b, R, rule='none', debug=False, **kwargs):
    # x: (#node, dim_t)
    # W: (dim_t, dim_{t+1})
    # b: (dim_{t+1})
    # R: (#node, dim_{t+1})
    gamma, epsilon, alpha, beta = get_lrp_hyperparameters(rule, **kwargs)

    if rule in ['gamma', 'epsilon', 'none']:
        numerator = x.unsqueeze(-1) * (W + gamma * W.clamp(0)).unsqueeze(0) # (#node, dim_t, dim_t+1)
        denominator = numerator.sum(axis=1) + (b + gamma * b.clamp(0)) + epsilon if b is not None else numerator.sum(axis=1) + epsilon
        R_linear = (numerator / denominator.unsqueeze(1) * R.unsqueeze(1)).sum(axis=-1) 
        R_b = (b + gamma * b.clamp(0)).unsqueeze(0) / denominator * R if b is not None else torch.tensor(0)
    elif rule in ['ab']:
        numerator = x.unsqueeze(-1) * W.unsqueeze(0) # (#node, dim_t, dim_t+1)
        numerator_pos = numerator.clamp(0)
        numerator_neg = numerator.clamp(max=0)
        denominator_pos = numerator_pos.sum(axis=1) + b.clamp(0) + epsilon if b is not None else numerator_pos.sum(axis=1) + epsilon
        denominator_neg = numerator_neg.sum(axis=1) + b.clamp(max=0) - epsilon if b is not None else numerator_neg.sum(axis=1) - epsilon
        R_linear = ((alpha * numerator_pos / denominator_pos.unsqueeze(1) + \
                     beta  * numerator_neg / denominator_neg.unsqueeze(1)) * R.unsqueeze(1)).sum(axis=-1)
    if debug:
        print(R_linear.sum() + R_b.sum() - R.sum())
    return R_linear
    
@torch.no_grad()
def lrp_gconv(x, A, R, rule='none', debug=False, **kwargs):
    # x: (#node, dim_t)
    # A: (#node, #node)
    # R: (#node, dim_{t+1})
    numerator = x.unsqueeze(1) * A.unsqueeze(-1) # (#node_t, #node_t+1, dim_t)
    # T = torch.nan_to_num(numerator / (numerator.sum(axis=0).unsqueeze(0)), 0.) # (#node_t, #node_t+1, dim_t)
    # T_lastdim = T[:, :, -1] # (#node_t, #node_t+1)
    # print(T_lastdim.sum(axis=0))
    # print(R[:, -1])
    # print((A@x)[:, -1])
    if rule == 'none':
        R_out = (torch.nan_to_num(numerator / (numerator.sum(axis=0).unsqueeze(0)), 0.) * R.unsqueeze(0)).sum(axis=1)
    elif rule == 'epsilon':
        epsilon = kwargs['epsilon']
        R_out = (torch.nan_to_num(numerator / (numerator.sum(axis=0).unsqueeze(0) + epsilon), 0.) * R.unsqueeze(0)).sum(axis=1)
    elif rule == 'ab':
        epsilon = kwawrgs['epsilon'] if 'epsilon' in kwargs else 1e-6
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        numerator_pos = numerator.clamp(0)
        numerator_neg = numerator.clamp(max=0)
        denominator_pos = numerator_pos.sum(axis=0) + epsilon
        denominator_neg = numerator_neg.sum(axis=0) - epsilon
        R_out = ((torch.nan_to_num(alpha * numerator_pos / denominator_pos, 0.) + \
                  torch.nan_to_num(beta  * numerator_neg / denominator_neg, 0.)) * R.unsqueeze(0)).sum(axis=1)

    if debug:
        # print(R_out.sum(axis=0))
        # print(R.sum(axis=0))
        print(R_out.sum() - R.sum())
    return R_out

from torch_geometric.utils import to_dense_adj
def get_model_temporary_res(model, x, edge_index):
    A = to_dense_adj(edge_index).squeeze()
    A += torch.eye(A.shape[0]) # add selfloop
    relu = list(model.gin[0].nn.modules())[1]
    module_temporary_res = [{} for _ in range(len(model.gin))]
    # model.gin[0]: GINConv(nn=MLP(7, 32, 32))
    module_temporary_res[0]['out'] = A @ x

    module_temporary_res[0]['lin1'] = list(model.gin[0].nn.modules())[2][0]
    module_temporary_res[0]['lin2'] = list(model.gin[0].nn.modules())[2][1]

    module_temporary_res[0]['out1'] = relu(module_temporary_res[0]['out'] @ module_temporary_res[0]['lin1'].weight.T + module_temporary_res[0]['lin1'].bias) if model.bias else relu(module_temporary_res[0]['out'] @ module_temporary_res[0]['lin1'].weight.T)
    module_temporary_res[0]['out2'] = module_temporary_res[0]['out1'] @ module_temporary_res[0]['lin2'].weight.T + module_temporary_res[0]['lin2'].bias if model.bias else module_temporary_res[0]['out1'] @ module_temporary_res[0]['lin2'].weight.T

    # model.gin[1]: ReLU(inplace=True)
    module_temporary_res[1]['out2'] = relu(module_temporary_res[0]['out2'])

    for i in range(1, len(model.gin) // 2):
        # model.gin[2]: GINConv(nn=MLP(32, 32, 32))
        module_temporary_res[2 * i]['out']  = A @ module_temporary_res[2 * i - 1]['out2']

        module_temporary_res[2 * i]['lin1'] = list(model.gin[2 * i].nn.modules())[2][0]
        module_temporary_res[2 * i]['lin2'] = list(model.gin[2 * i].nn.modules())[2][1]

        module_temporary_res[2 * i]['out1'] = relu(module_temporary_res[2 * i]['out'] @ module_temporary_res[2 * i]['lin1'].weight.T + module_temporary_res[2 * i]['lin1'].bias) if model.bias else relu(module_temporary_res[2 * i]['out'] @ module_temporary_res[2 * i]['lin1'].weight.T)
        module_temporary_res[2 * i]['out2'] = module_temporary_res[2 * i]['out1'] @ module_temporary_res[2 * i]['lin2'].weight.T + module_temporary_res[2 * i]['lin2'].bias if model.bias else module_temporary_res[2 * i]['out1'] @ module_temporary_res[2 * i]['lin2'].weight.T

        # model.gin[3]: ReLU(inplace=True)
        module_temporary_res[2 * i + 1]['out2'] = relu(module_temporary_res[2 * i]['out2'])

    # model.linear
    linear_out = module_temporary_res[len(model.gin) - 1]['out2'] @ model.linear.weight.T + model.linear.bias if model.bias else module_temporary_res[len(model.gin) - 1]['out2'] @ model.linear.weight.T
    # Read-out function
    read_out = linear_out.sum(axis=0)

    assert read_out.isclose(model.forward(x, edge_index), atol=1e-04).all(),  f'read_out = {read_out.detach()} != {model.forward(x, edge_index).detach()} = model.forward(x, edge_index), which is a problem.'
    return module_temporary_res, linear_out

def get_model_temporary_res_old(model, x, edge_index):
    A = to_dense_adj(edge_index).squeeze()
    A += torch.eye(A.shape[0]) # add selfloop
    relu = list(model.gin.module_0.nn.modules())[1]
    module_temporary_res = [{} for _ in range(len(list(model.gin.modules())))]

    # model.gin.module_0: GINConv(nn=MLP(7, 32, 32))
    module_temporary_res[0]['out'] = A @ x

    module_temporary_res[0]['lin1'] = list(model.gin.module_0.nn.modules())[2][0]
    module_temporary_res[0]['lin2'] = list(model.gin.module_0.nn.modules())[2][1]

    module_temporary_res[0]['out1'] = relu(module_temporary_res[0]['out'] @ module_temporary_res[0]['lin1'].weight.T + module_temporary_res[0]['lin1'].bias) if model.bias else relu(module_temporary_res[0]['out'] @ module_temporary_res[0]['lin1'].weight.T)
    module_temporary_res[0]['out2'] = module_temporary_res[0]['out1'] @ module_temporary_res[0]['lin2'].weight.T + module_temporary_res[0]['lin2'].bias if model.bias else module_temporary_res[0]['out1'] @ module_temporary_res[0]['lin2'].weight.T

    # model.gin.module_1: ReLU(inplace=True)
    module_temporary_res[1]['out2'] = relu(module_temporary_res[0]['out2'])

    # model.gin.module_2: GINConv(nn=MLP(32, 32, 32))
    module_temporary_res[2]['out']  = A @ module_temporary_res[1]['out2']

    module_temporary_res[2]['lin1'] = list(model.gin.module_2.nn.modules())[2][0]
    module_temporary_res[2]['lin2'] = list(model.gin.module_2.nn.modules())[2][1]

    module_temporary_res[2]['out1'] = module_temporary_res[2]['out'] @ module_temporary_res[2]['lin1'].weight.T + module_temporary_res[2]['lin1'].bias if model.bias else module_temporary_res[2]['out'] @ module_temporary_res[2]['lin1'].weight.T
    module_temporary_res[2]['out1'] = relu(module_temporary_res[2]['out1'])
    module_temporary_res[2]['out2'] = module_temporary_res[2]['out1'] @ module_temporary_res[2]['lin2'].weight.T + module_temporary_res[2]['lin2'].bias if model.bias else module_temporary_res[2]['out1'] @ module_temporary_res[2]['lin2'].weight.T

    # model.gin.module_3: ReLU(inplace=True)
    module_temporary_res[3]['out2'] = relu(module_temporary_res[2]['out2'])

    # model.gin.module_4: GINConv(nn=MLP(32, 32, 32))
    module_temporary_res[4]['out'] = A @ module_temporary_res[3]['out2']

    module_temporary_res[4]['lin1'] = list(model.gin.module_4.nn.modules())[2][0]
    module_temporary_res[4]['lin2'] = list(model.gin.module_4.nn.modules())[2][1]

    module_temporary_res[4]['out1'] = module_temporary_res[4]['out'] @ module_temporary_res[4]['lin1'].weight.T + module_temporary_res[4]['lin1'].bias if model.bias else module_temporary_res[4]['out'] @ module_temporary_res[4]['lin1'].weight.T
    module_temporary_res[4]['out1'] = relu(module_temporary_res[4]['out1'])
    module_temporary_res[4]['out2'] = module_temporary_res[4]['out1'] @ module_temporary_res[4]['lin2'].weight.T + module_temporary_res[4]['lin2'].bias if model.bias else module_temporary_res[4]['out1'] @ module_temporary_res[4]['lin2'].weight.T

    # model.gin.module_5: ReLU(inplace=True)
    module_temporary_res[5]['out2'] = relu(module_temporary_res[4]['out2'])

    # model.linear
    linear_out = module_temporary_res[5]['out2'] @ model.linear.weight.T + model.linear.bias if model.bias else module_temporary_res[5]['out2'] @ model.linear.weight.T

    # Read-out function
    read_out = linear_out.sum(axis=0)

    assert read_out.isclose(model.forward(x, edge_index), atol=1e-04).all(), f'read_out = {read_out.detach()} != {model.forward(x, edge_index).detach()} = model.forward(x, edge_index), which is a problem.'
    return module_temporary_res, linear_out