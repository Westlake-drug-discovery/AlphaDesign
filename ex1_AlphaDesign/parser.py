import argparse


# 需要修改的参数
def create_parser():
    """Creates a parser with all the variables that can be edited by the user.

    Returns:
        parser: a parser for the command line
    """
    parser = argparse.ArgumentParser()
    # Set-up parameters
    # /usr/data/gzy/SemiRetro/exp1_USPTO50k/part1_center_identification/results
    parser.add_argument('--res_dir',default='/gaozhangyang/experiments/AlphaDesign/ex1_AlphaDesign/results',type=str)
    parser.add_argument('--ex_name', default='debug', type=str) 
    parser.add_argument('--gpu', default=7, type=int)
    parser.add_argument('--search',default=1,type=int)
    parser.add_argument('--method',default='AlphaDesign', choices=['NIPS19', 'GVP', 'GCA', 'AlphaDesign','SGNN'])

    # dataset parameters
    parser.add_argument('--preprocess_path',default="/gaozhangyang/experiments/Protein_Design/dataset/preprocessed")
    parser.add_argument('--data_name', default='UP000000437_7955_DANRE_v2') 
    parser.add_argument('--batch_size',default=16,type=int,help='Batch size')
    parser.add_argument('--limit_length', default=1, type=int)
    parser.add_argument('--joint_data', default=0, type=int)
    
    # Training parameters
    parser.add_argument('--epoch_s', default=0, type=int, help='start epoch')
    parser.add_argument('--epoch_e', default=101, type=int, help='end epoch') 
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--patience', default=20,type=int)
    parser.add_argument('--lr',default=0.01,type=float,help='Learning rate') 
    
    # NIPS19
    parser.add_argument('--hidden', type=int, default=128, help='number of hidden dimensions')
    parser.add_argument('--k_neighbors', type=int, default=30, help='Neighborhood size for k-NN')
    parser.add_argument('--vocab_size', type=int, default=20, help='Alphabet size')
    parser.add_argument('--features', type=str, default='full', help='Protein graph features')
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing rate')
    
    # AlphaDesign
    parser.add_argument('--max-nodes', default=3000, type=int)
    parser.add_argument('--top-k', type=int, default=30)
    return parser


    