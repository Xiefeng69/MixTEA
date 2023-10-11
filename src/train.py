import time
import torch
import numpy as np
import copy
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import *
from src.utils import *
from src.model import KGEncoder

def train(args, device):

    print('[loading KG data...]\n')
    kgdata = KGData(args.task, args.fold, setting=args.setting)
    train_pair, valid_pair, test_pair = kgdata.load_pair_data()
    adj_matrix, r_index, r_val, adj_features, rel_features, rel_features_top, rdict, rel_in, rel_out = kgdata.load_matrix_data()
    unlabeled_pair = np.concatenate((valid_pair, test_pair), axis=0)
    unlabeled_pair = copy.deepcopy(unlabeled_pair)
    unlabeled_s = [e1 for e1, e2 in unlabeled_pair]
    unlabeled_t = [e2 for e1, e2 in unlabeled_pair]
    np.random.shuffle(unlabeled_s)
    np.random.shuffle(unlabeled_t)

    # adj_matrix = np.stack(adj_matrix.nonzero(), axis = 1)
    # rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    # ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

    ent_num = kgdata.ent_num
    rel_num = kgdata.rel_num
    triple_num = kgdata.triple_num

    adj_matrix = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_matrix)).to(device)
    # rel_features = sparse_mx_to_torch_sparse_tensor(rel_features).to(device)
    rel_features_top = sparse_mx_to_torch_sparse_tensor(rel_features_top).to(device)
    rel_in = sparse_mx_to_torch_sparse_tensor(rel_in).to(device)
    rel_out = sparse_mx_to_torch_sparse_tensor(rel_out).to(device)

    print('[model initializing...]\n')

    model = KGEncoder(args, ent_num=ent_num, adj_matrix=adj_matrix, rel_features=(rel_in, rel_out), device=device, name="student")
    _model = KGEncoder(args, ent_num=ent_num, adj_matrix=adj_matrix, rel_features=(rel_in, rel_out), device=device, name="teacher")
    model = model.to(device=device)
    _model = _model.to(device=device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_dataset = Dataset(np.array(train_pair))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    unlabeled_dataset = Dataset(np.array(unlabeled_pair))
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=len(unlabeled_pair), shuffle=False)
    # criterion = torch.nn.MarginRankingLoss(margin=args.neg_margin)

    print("--------------------INFO--------------------\n")
    print(f'- current task: \033[93m{args.task}\033[0m\n')
    print(f'- #entity: \033[93m{ent_num}\033[0m\n')
    print(f'- #relation: \033[93m{rel_num}\033[0m\n')
    print(f'- #triple: \033[93m{triple_num}\033[0m\n')
    print(f'- #labeled number: \033[93m{len(train_pair)+len(valid_pair)+len(test_pair)}\033[0m\n')
    print(f'- #batch size: \033[93m{args.batch_size}\033[0m\n')
    print(f'- #total params: \033[93m{pytorch_total_params}\033[0m\n')
    print("--------------------------------------------\n")

    #STEP: begin training
    try:
        t_total_start = time.time()
        best_score = 0.0
        hit1_st = 1.0
        hit1_ts = 0.0
        model_path = f"save/{args.task}"

        '''begin model training...'''
        for e in range(1, args.epoch):
            model.train()
            _model.train()
            global bad_count
            align_total_loss = 0.0
            pseudo_total_loss = 0.0
            t_start = time.time()
            adjust_learning_rate(optimizer, e, args.lr)

            if e % args.neg_iter == 1:
                neg_sample_list = list()

            # supervised alignment learning with labeled data
            for idx, data in enumerate(train_loader):
                model.train()
                kg1_align, kg2_align = data
                kg1_align, kg2_align = np.array(kg1_align), np.array(kg2_align)
                vec = model()
                # negtive sampling
                if e % args.neg_iter == 1:
                    neg_left, neg_right = kgdata.negative_sampling(
                        bsize=len(kg1_align), 
                        kg1_align=kg1_align, 
                        kg2_align=kg2_align, 
                        neg_samples_size=args.neg_samples_size, 
                        target_left=unlabeled_s, 
                        target_right=unlabeled_t, 
                        vec=vec.detach().cpu().numpy(), 
                        e=args.truncated_epsilon
                    )
                    neg_sample_list.append([neg_left, neg_right])
                else:
                    neg_left, neg_right = neg_sample_list[idx]
                
                align_loss = model.alignment_loss(
                    vec, 
                    kg1_align, 
                    kg2_align, 
                    neg_left, 
                    neg_right, 
                    neg_samples_size=args.neg_samples_size, 
                    neg_margin=args.neg_margin, 
                    dist=args.dist
                )
                align_total_loss += align_loss
            
            # pseudo mapping learning with unlabeled data
            for _, data in enumerate(unlabeled_loader):
                torch.cuda.empty_cache()
                kg1_ids, kg2_ids = data
                kg1_ids, kg2_ids = np.array(kg1_ids), np.array(kg2_ids)
                vec = model()
                with torch.no_grad():
                    _vec = _model()
                pseudo_loss = model.pseudo_ce_loss(ent_embedding1=vec[kg1_ids], ent_embedding2=vec[kg2_ids], ent_embedding3=_vec[kg1_ids], ent_embedding4=_vec[kg2_ids], hit1_st=hit1_st, hit1_ts=hit1_ts)
                r = args.consistency * sigmoid_rampup(e, args.consistency_rampup)
                pseudo_loss = pseudo_loss * r
                pseudo_total_loss += pseudo_loss
            
            # the final objective
            if not args.il:
                loss = align_total_loss + pseudo_total_loss
            else:
                loss = align_total_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update teacher model
            _model.update(model, epoch=e)
            
            print(f"epoch: {e}, align_loss: {round(align_total_loss.item(), 2)}, pseudo_loss: {round(pseudo_total_loss.item(), 2)} time: {round((time.time()-t_start), 2)}\n")

            if e >= args.val_start and e % args.val_iter == 0 and args.val:
                with torch.no_grad():
                    model.eval()
                    vec = model()
                    vec = vec.detach().cpu().numpy()
                    if args.record:
                        Lvec = np.array([vec[e] for e in test_pair[:,0]])
                        Rvec = np.array([vec[e] for e in test_pair[:,1]])
                        del vec;
                        hit1, hitk, mrr, mr = eval_entity_alignment_faiss(Lvec, Rvec, test_num=len(test_pair), k=args.k, eval_metric=args.eval_metric)
                        print(f"[test: epoch: {e}, hit@1: {round(hit1, 3)}, hit@{args.k}: {round(hitk, 3)}, mrr is {round(mrr, 3)}]\n")
                        with open(f'result/{args.task}_test.csv', 'a', encoding='utf-8') as file:
                            file.write('\n')
                            file.write(f"{e}, {round(hit1, 3)}, {round(hitk, 3)}, {round(mrr, 3)}, {round(mr, 3)}")
                    else:
                        Lvec = np.array([vec[e] for e in valid_pair[:,0]])
                        Rvec = np.array([vec[e] for e in valid_pair[:,1]])
                        del vec;
                        hit1_st, hitk, mrr, mr = eval_entity_alignment_faiss(Lvec, Rvec, test_num=len(valid_pair), k=args.k, eval_metric=args.eval_metric)
                        print(f"[validation: epoch: {e}, hit@1: {round(hit1_st, 3)}, hit@{args.k}: {round(hitk, 3)}, mrr is {round(mrr, 3)}]\n")
                        hit1_ts, _, _, _ = eval_entity_alignment_faiss(Rvec, Lvec, test_num=len(valid_pair), k=args.k, eval_metric=args.eval_metric)
                        with open(model_path, "wb") as f:
                            torch.save(model.state_dict(), f)
                        if hit1_st > best_score:
                            bad_count = 0
                            best_score = hit1_st
                        else:
                            bad_count = bad_count + 1
                        if bad_count == args.patience:
                            break;

            '''iterative learning...'''
            if e >= args.il_start and (e-args.il_start) % args.il_iter == 0 and args.il:
                with torch.no_grad():
                    model.eval()
                    vec = model()
                    vec = vec.detach().cpu().numpy()
                    # Lvec = np.array([vec[e] for e in test_pair[:,0]])
                    # Rvec = np.array([vec[e] for e in test_pair[:,1]])
                    # Lvec = Lvec / (np.linalg.norm(Lvec,axis=-1,keepdims=True) + 1e-5)
                    # Rvec = Rvec / (np.linalg.norm(Rvec,axis=-1,keepdims=True) + 1e-5)
                    new_alignment_pair = find_new_alignment(train_pair, unlabeled_s, unlabeled_t, vec, strategy=args.il_method)
                    if new_alignment_pair is not None:
                        new_train_pair = np.concatenate((train_pair, new_alignment_pair), axis=0)
                        train_dataset = Dataset(new_train_pair)
                        batchsize = len(new_train_pair)
                        train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False)

    except KeyboardInterrupt:
        print('-' * 40)
        print(f'Exiting from training early, epoch {e}\n')

    print("[training end]")
    total_time = int(time.time()-t_total_start)
    with torch.no_grad():
        with open(model_path, "rb") as f:
            model.load_state_dict(torch.load(f))
        model.eval()
        vec = model()
        vec = vec.detach().cpu().numpy()
        Lvec = np.array([vec[e] for e in test_pair[:,0]])
        Rvec = np.array([vec[e] for e in test_pair[:,1]])
        del vec;
        hit1, hitk, mrr, mr = eval_entity_alignment_faiss(Lvec, Rvec, test_num=len(test_pair), k=args.k, eval_metric=args.eval_metric)
        hit1, hitk, mrr, mr = round(hit1, 3), round(hitk, 3), round(mrr, 3), round(mr, 3)
    print(f'total time consume: {datetime.timedelta(seconds=total_time)}')
    print(f'+ Hit@1: \033[94m{hit1}\033[0m')
    print(f'+ Hit@k: \033[94m{hitk}\033[0m')
    print(f'+ MRR: \033[94m{mrr}\033[0m')
    print(f'+ MR: \033[94m{mr}\033[0m')

    print("---------------save result-----------------\n")
    with open(f'result/{args.task}.csv', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write(f"{args.fold}, {round(hit1, 3)}, {round(hitk, 3)}, {round(mrr, 3)}, {round(mr, 3)}, {args.neg_margin}, {args.layer}, {args.ema_decay}")
    print("-------------------------------------------\n")