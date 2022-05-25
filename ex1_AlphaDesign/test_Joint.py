import nni
import os
import sys; sys.path.append('/gaozhangyang/Protein_Design')
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
    # config['ex_name'] = 'conf_seq_dec'
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    config['method'] = "AlphaDesign"
    config['joint_data'] = 1
    config['batch_size'] = 32
    config['limit_length'] = 0

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
        # config['ex_name'] = prefix+'_'+ args.method+'_'+args.data_name+'_{}'.format(args.epoch_e)

    print(args)

    SetSeed(args.seed)
    
    from ex1_AlphaDesign.engine_ADesign import Exp
    exp = Exp(args)
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    # exp.train(args)
    
    print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.model.load_state_dict(torch.load(os.path.join(exp.path, 'checkpoint.pth')))

    data_name_list = ['UP000000437_7955_DANRE_v2',
            'UP000000559_237561_CANAL_v2',
            'UP000000589_10090_MOUSE_v2',
            'UP000000625_83333_ECOLI_v2',
            'UP000000803_7227_DROME_v2',
            'UP000000805_243232_METJA_v2',
            'UP000001450_36329_PLAF7_v2',
            'UP000001584_83332_MYCTU_v2',
            'UP000001940_6239_CAEEL_v2',
            'UP000002195_44689_DICDI_v2',
            'UP000002296_353153_TRYCC_v2',
            'UP000002311_559292_YEAST_v2',
            'UP000002485_284812_SCHPO_v2',
            'UP000002494_10116_RAT_v2',
            'UP000005640_9606_HUMAN_v2',
            'UP000006548_3702_ARATH_v2',
            'UP000007305_4577_MAIZE_v2',
            'UP000008153_5671_LEIIN_v2',
            'UP000008816_93061_STAA8_v2',
            'UP000008827_3847_SOYBN_v2',
            'UP000059680_39947_ORYSJ_v2']

    from alphfold_data import AlphaFold
    for data_name in data_name_list:
        preprocess_path = os.path.join(args.preprocess_path, data_name)
        test_set = AlphaFold(preprocess_path, mode = 'test', limit_length=args.limit_length, joint_data=0) 

        with torch.no_grad():
            recovery = exp.sample_all(test_set)

        print("Final | {}: \tnum is {}\trecovery {}\n".format(data_name, len(test_set), recovery))
        logging.info("Final | {}: \tnum is{}\trecovery {}\n".format(data_name, len(test_set), recovery))
