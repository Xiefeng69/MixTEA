import os
import torch
import argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from src.utils import set_random_seed, save_args
from src.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parameters for model training 
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="en_fr_15k", help="the alignment task name", choices=["d_w_15k", "d_y_15k", "en_fr_15k", "en_de_15k"])
    parser.add_argument("--fold", type=int, default=1, help="the fold cross-validation")
    parser.add_argument("--setting", type=str, default="semi", choices=["unsup", "semi"])
    parser.add_argument("--epoch", type=int, default=300, help="epoch to run")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--k", type=float, default=5, help="hit@k")
    parser.add_argument("--patience", type=int, default=5, help="patience default 5")
    parser.add_argument("--val", action="store_true", default=True, help="need validation?")
    parser.add_argument("--val_start", type=int, default=10, help="when to start validation")
    parser.add_argument("--val_iter", type=int, default=10, help="If val, whats the validation step?")
    parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
    parser.add_argument("--il_start", type=int, default=20, help="If Il, when to start?")
    parser.add_argument("--il_iter", type=int, default=20, help="If IL, what's the update step?")
    parser.add_argument("--il_method", type=str, default="sim_th", choices=["sim_th", "bi_sim"])
    parser.add_argument("--eval_metric", type=str, default="inner", choices=["inner", "l2"])
    parser.add_argument("--record", action="store_true", default=False)

    # parameters for model architecture
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay coefficient")
    parser.add_argument("--ent_dim", type=int, default=256)
    parser.add_argument("--rel_dim", type=int, default=128)
    parser.add_argument("--t_dropout", type=float, default=0.2, help="dropout rate for teacher model")
    parser.add_argument("--s_dropout", type=float, default=0.2, help="dropout rate for student model")
    parser.add_argument("--layer", type=int, default=2, help="layer number of GNN-based encoder")
    parser.add_argument("--head", type=int, default=1, help="number of multi-head attention")
    parser.add_argument('--ema_decay', type=float, default=0.9, help='ema_decay')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=20, help='consistency_rampup until 20 epoch')
    parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")
    parser.add_argument("--neg_margin", type=float, default=2, help="negative margin for loss computation")
    parser.add_argument("--neg_samples_size", type=int, default=5, help="number of negative samples")
    parser.add_argument("--neg_iter", type=int, default=10, help="re-calculate epoch of negative samples")
    parser.add_argument('--truncated_epsilon', type=float, default=0.9, help='the epsilon of truncated negative sampling')
    
    args = parser.parse_args()
    save_args(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ncurrent device is \033[92m{device}\033[0m \n")
    set_random_seed(args.seed)

    if args.record:
        args.val_start = 1
        args.val_iter = 1

    train(args=args, device=device)