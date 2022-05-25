import nni
import os
import sys; sys.path.append('/gaozhangyang/experiments/AlphaDesign')
from ex1_AlphaDesign.parser import create_parser
import logging
import random
import numpy as np
import torch

def SetSeed(seed,det=True):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = create_parser().parse_args()

    config = args.__dict__
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    if not args.joint_data: 
        if args.limit_length == 1:
            prefix = 'SL'
        else:
            prefix = 'SF'
    else:
        if args.limit_length == 1:
            prefix = 'JL'
        else:
            prefix = 'JF'
    if args.search:
        config['ex_name'] = prefix+'_'+ args.method+'_'+args.data_name#+'_{}'.format(args.epoch_e)
    print(args)

    SetSeed(args.seed)
    
    from ex1_AlphaDesign.engine_ADesign import Exp
    exp = Exp(args)
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(args)
    
    print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.model.load_state_dict(torch.load(os.path.join(exp.path, 'checkpoint.pth')))

    with torch.no_grad():
        num_correct = exp.evaluate(exp.test_loader, 'test')
    print("Final | perplexity: {0:.7f}\n".format(num_correct))
    logging.info("Final | perplexity: {0:.7f}\n".format(num_correct))
