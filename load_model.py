import torch
from utils import get_model
import torch.backends.cudnn as cudnn
import pdb

def prepare_model(model_path, use_cuda, arch, strip_parallel=True):
    def strip_data_parallel(s):
        if s.startswith('module'):
            return s[len('module.'):]
        else:
            return s
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(arch, num_classes=num_classes,
                      normalize_input=normalize_input)
    state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    pdb.set_trace()
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if not all([k.startswith('module') for k in state_dict]):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
    else:
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    if strip_parallel:
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model


prepare_model(model_path='rst_adv/checkpoint-epoch200.pt',
              use_cuda=True,
              arch='wrn-28-10')