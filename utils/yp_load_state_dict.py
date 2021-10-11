import torch

def load_weights_to_model(model, weight_path):
    from collections import OrderedDict
    pretrain_dict = torch.load(weight_path, map_location='cpu')
    try:
        state_dict = pretrain_dict['state_dict']
    except:
        state_dict = pretrain_dict

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():  # avoid dataparallel trained on gpus
        name = k.replace('module.','')
        new_state_dict[name] = v

    new_state_dict2 = OrderedDict()
    for k, v in new_state_dict.items():  # for IRR-PWC and PWCNet
        name = k.replace('_model.','')
        new_state_dict2[name] = v

    model.load_state_dict(new_state_dict2, strict=False)
    return model