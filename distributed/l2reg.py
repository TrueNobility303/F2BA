import argparse
import copy
import hypergrad as hg # hypergrad package
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized

from torchvision import datasets

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


################################################################################
#
#  Bilevel Optimization Toy Example
#
#  min_{x,w} f(x, w)
#  s.t. x = argmin_x g(x, w)
#
#  here: f(x, w) is on valset
#        g(x, w) is on trainset
#
#  f_x = df/dx
#  f_w = df/dw
#  g_x = dg/dx
#  g_w = dg/dw
#
################################################################################


# centralized and decentralized methods
METHODS = [
    'F2BA',
    'AID',
    'ITD',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="news20")
    parser.add_argument('--generate_data', action='store_true',
            default=False, help='whether to create data')
    parser.add_argument('--pretrain', action='store_true',
                                      default=False, help='whether to create data and pretrain on valset')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--iterations', type=int, default=10, help='T')
    parser.add_argument('--K', type=int, default=10, help='k')
    parser.add_argument('--data_path', default='./data', help='where to save data')
    parser.add_argument('--model_path', default='./save_l2reg', help='where to save model')
    parser.add_argument('--log', default='./log', help='where to save log')

    parser.add_argument('--x_lr', type=float, default=10)
    parser.add_argument('--xhat_lr', type=float, default=10)
    parser.add_argument('--w_lr', type=float, default=1000)
    parser.add_argument('--w_momentum', type=float, default=0.0)
    parser.add_argument('--x_momentum', type=float, default=0.0)

    parser.add_argument('--v_lr', type=float, default=0.1, help='stepsize in matrxi inverse')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='F2BA', choices=METHODS)
    parser.add_argument('--lmbd', type=float, default=10.0)
    parser.add_argument('--size', type=int, default=4, help='number of agents')
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args

def init_process(rank, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=args.size)
    fn(rank, args)

def get_data(args):

    # def from_sparse(x):
    #     x = x.tocoo()
    #     values = x.data
    #     indices = np.vstack((x.row, x.col))
    #     i = torch.LongTensor(indices)
    #     v = torch.FloatTensor(values)
    #     shape = x.shape
    #     return torch.FloatTensor(i, v, torch.Size(shape))

    val_size = 0.5
    train_x, train_y = fetch_20newsgroups_vectorized(subset='train',
                                                     return_X_y=True,
                                                     data_home=args.data_path,
                                                     download_if_missing=True)

    test_x, test_y = fetch_20newsgroups_vectorized(subset='test',
                                                   return_X_y=True,
                                                   data_home=args.data_path,
                                                   download_if_missing=True)

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=val_size)
    test_x, teval_x, test_y, teval_y = train_test_split(test_x, test_y, stratify=test_y, test_size=0.5)

    train_x, val_x, test_x, teval_x = map(torch.LongTensor, [train_x, val_x, test_x, teval_x])
    train_y, val_y, test_y, teval_y = map(torch.LongTensor, [train_y, val_y, test_y, teval_y])

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)

### original f, g, and gradients

def f(x, w, dataset):
    data_x, data_y = dataset
    y = data_x.mm(x)
    loss = F.cross_entropy(y, data_y, reduction='mean')
    return loss

def g(x, w, dataset):
    data_x, data_y = dataset
    y = data_x.mm(x)
    loss = F.cross_entropy(y, data_y, reduction='mean')
    reg_loss = 0.5 * (x.pow(2) * w.view(-1, 1).exp()).mean() # l2 reg loss
    return loss + reg_loss

def g_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_w(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, w,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_x_xhat_w(x, xhat, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset) - g(xhat.detach(), w, dataset)
    grad = torch.autograd.grad(loss, [x, w],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return loss, grad[0], grad[1]

def g_x_xhat_w_bo(x, xhat, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset) - g(xhat, w, dataset)
    grad = torch.autograd.grad(loss, [x, xhat, w],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad[0], grad[1], grad[2]

def f_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = f(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

### Define evaluation metric

def evaluate(x, w, testset):
    with torch.no_grad():
        test_x, test_y = testset  
        y = test_x.mm(x)
        loss = F.cross_entropy(y, test_y).detach().item()
        acc = (y.argmax(-1).eq(test_y).sum() / test_y.shape[0]).detach().cpu().item()
    return loss, acc


def baseline(args, x, w, trainset, valset, testset, tevalset): # no regularization
    opt = torch.optim.SGD([x], lr=args.x_lr, momentum=args.x_momentum)
    n = trainset[0].shape[0]

    best_teval_loss = np.inf
    best_config = None
    for epoch in range(args.epochs):
        opt.zero_grad()
        x.grad = f_x(x, None, trainset).data
        opt.step()
        test_loss, test_acc = evaluate(x, None, testset)
        teval_loss, teval_acc = evaluate(x, None, tevalset)
        if teval_loss < best_teval_loss:
            best_teval_loss = teval_loss
            best_config = (test_loss, test_acc, x.data.clone())
        #print(f"[baseline] epoch {epoch:5d} test loss {test_loss:10.4f} test acc {test_acc:10.4f}")
    print(f"[baseline] best test loss {best_config[0]} best test acc {best_config[1]}")
    return best_config

def F2BA(rank, args):

    trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
    device = torch.device(args.device)
    trainset = (trainset[0].float().to(device).to_dense(), trainset[1].to(device).to_dense())
    valset   = (valset[0].float().to(device).to_dense(), valset[1].to(device).to_dense())
    testset  = (testset[0].float().to(device).to_dense(), testset[1].to(device).to_dense())
    tevalset = (tevalset[0].float().to(device).to_dense(), tevalset[1].to(device).to_dense())

    n_feats  = trainset[0].shape[-1]
    num_classes = trainset[1].unique().shape[-1]

    n = trainset[0].shape[0]
    p = n // args.size
    trainset = (trainset[0][rank*p:(rank+1)*p,:], trainset[1][rank*p:(rank+1)*p])
    valset   = (valset[0][rank*p:(rank+1)*p,:], valset[1][rank*p:(rank+1)*p])
   
    #x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
    x = torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
    x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
    x.data.copy_(torch.load("./save_l2reg/pretrained.pt").to(device))
    w = torch.zeros(n_feats, requires_grad=True, device=device)

    pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
    loss = pretrained_stats["pretrain_test_loss"]
    acc  = pretrained_stats["pretrain_test_acc"]
    
    if rank == 0:
        print(f"[info] pretrained without regularization achieved loss {loss:.2f} acc {acc:.2f}")
    
    xhat = copy.deepcopy(x)

    total_time = 0.0
    time_lst = []
    comm_lst = []
    loss_lst = []

    outer_opt = torch.optim.SGD([w], lr=args.w_lr, momentum=args.w_momentum)
    inner_opt = torch.optim.SGD([
        {'params': [x], 'lr': args.x_lr},
        {'params': [xhat], 'lr': args.xhat_lr}], momentum=args.x_momentum)

    for epoch in range(args.epochs):

        xhat.data = x.data.clone()
        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            gx = g_x(xhat, w, trainset)
            fx = f_x(x, w, valset)
            xhat.grad =  args.lmbd * gx
            x.grad = fx  + args.lmbd * gx 

            dist.all_reduce(xhat.grad.data, op=dist.ReduceOp.SUM)
            dist.all_reduce(x.grad.data, op=dist.ReduceOp.SUM)
            xhat.grad.data /= args.size
            x.grad.data /= args.size
            inner_opt.step()

        _, gx, gw_minus_gw_k = g_x_xhat_w(x, xhat, w, trainset)
        outer_opt.zero_grad()
        w.grad =  args.lmbd * gw_minus_gw_k 
        dist.all_reduce(w.grad.data, op=dist.ReduceOp.SUM)
        w.grad.data /= args.size
        outer_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        if rank == 0:
            test_loss, test_acc = evaluate(x, w, testset)
            teval_loss, teval_acc = evaluate(x, w, tevalset)
            comm_lst.append((epoch+1) * args.iterations )
            loss_lst.append(teval_loss)
            time_lst.append(total_time)
            print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | teval loss {teval_loss:6.4f}")

    if rank == 0:
        filename = f"./{args.log}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_xhatlr{args.xhat_lr},wlr{args.w_lr}_sd{args.seed}"
        with open(filename,'w') as f:
            print(comm_lst,loss_lst,time_lst, sep='\n', file=f)

def AID(rank, args):
    trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
    device = torch.device(args.device)
    trainset = (trainset[0].float().to(device).to_dense(), trainset[1].to(device).to_dense())
    valset   = (valset[0].float().to(device).to_dense(), valset[1].to(device).to_dense())
    testset  = (testset[0].float().to(device).to_dense(), testset[1].to(device).to_dense())
    tevalset = (tevalset[0].float().to(device).to_dense(), tevalset[1].to(device).to_dense())

    n_feats  = trainset[0].shape[-1]
    num_classes = trainset[1].unique().shape[-1]

    n = trainset[0].shape[0]
    p = n // args.size
    trainset = (trainset[0][rank*p:(rank+1)*p,:], trainset[1][rank*p:(rank+1)*p])
    valset   = (valset[0][rank*p:(rank+1)*p,:], valset[1][rank*p:(rank+1)*p])
   
    #x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
    x = torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
    x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
    x.data.copy_(torch.load("./save_l2reg/pretrained.pt").to(device))
    w = torch.zeros(n_feats, requires_grad=True, device=device)

    pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
    loss = pretrained_stats["pretrain_test_loss"]
    acc  = pretrained_stats["pretrain_test_acc"]
    
    if rank == 0:
        print(f"[info] pretrained without regularization achieved loss {loss:.2f} acc {acc:.2f}")

    total_time = 0.0
    time_lst = []
    comm_lst = []
    loss_lst = []

    v = torch.zeros_like(x).reshape([-1])

    outer_opt = torch.optim.SGD([w], lr=args.w_lr)
    inner_opt = torch.optim.SGD([x], lr=args.x_lr)
    inverse_opt = torch.optim.SGD([v], lr=args.v_lr)
    for epoch in range(args.epochs):

        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            x.grad = g_x(x, w, trainset)
            dist.all_reduce(x.grad.data, op=dist.ReduceOp.SUM)
            x.grad.data /= args.size
            inner_opt.step()
        
        # take the matrix inverse by solving the quadratic function

        b = f_x(x,w, valset).reshape([-1])
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        b /= args.size

        for it in range(args.iterations):
            inverse_opt.zero_grad()
            gx = g_x(x, w, trainset,create_graph=True,retain_graph=True).reshape([-1])
            v.grad =  torch.autograd.grad(torch.matmul(gx, v.detach()), x)[0].reshape([-1])
            dist.all_reduce(v.grad.data, op=dist.ReduceOp.SUM)
            v.grad.data /= args.size
            v.grad.data -= b
            inverse_opt.step()            

        outer_opt.zero_grad()
        gx = g_x(x, w, trainset,create_graph=True).reshape([-1])
        w.grad =  -torch.autograd.grad(torch.matmul(gx, v.detach()), w, retain_graph=True)[0]
        dist.all_reduce(w.grad.data, op=dist.ReduceOp.SUM)
        w.grad.data /= args.size
        outer_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        if rank == 0:
            test_loss, test_acc = evaluate(x, w, testset)
            teval_loss, teval_acc = evaluate(x, w, tevalset)
            comm_lst.append((epoch+1) * args.iterations )
            loss_lst.append(teval_loss)
            time_lst.append(total_time)
            print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | teval loss {teval_loss:6.4f}")

    if rank == 0:
        filename = f"./{args.log}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_vlr{args.v_lr}_wlr{args.w_lr}_sd{args.seed}"
        with open(filename,'w') as f:
            print(comm_lst,loss_lst,time_lst, sep='\n', file=f)

def ITD(rank, args):
    trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
    device = torch.device(args.device)
    trainset = (trainset[0].float().to(device).to_dense(), trainset[1].to(device).to_dense())
    valset   = (valset[0].float().to(device).to_dense(), valset[1].to(device).to_dense())
    testset  = (testset[0].float().to(device).to_dense(), testset[1].to(device).to_dense())
    tevalset = (tevalset[0].float().to(device).to_dense(), tevalset[1].to(device).to_dense())

    n_feats  = trainset[0].shape[-1]
    num_classes = trainset[1].unique().shape[-1]

    n = trainset[0].shape[0]
    p = n // args.size
    trainset = (trainset[0][rank*p:(rank+1)*p,:], trainset[1][rank*p:(rank+1)*p])
    valset   = (valset[0][rank*p:(rank+1)*p,:], valset[1][rank*p:(rank+1)*p])
   
    #x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
    x = torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
    x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
    x.data.copy_(torch.load("./save_l2reg/pretrained.pt").to(device))
    w = torch.zeros(n_feats, requires_grad=True, device=device)

    pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
    loss = pretrained_stats["pretrain_test_loss"]
    acc  = pretrained_stats["pretrain_test_acc"]
    
    if rank == 0:
        print(f"[info] pretrained without regularization achieved loss {loss:.2f} acc {acc:.2f}")

    total_time = 0.0
    time_lst = []
    comm_lst = []
    loss_lst = []

    v = torch.zeros_like(x).reshape([-1])

    outer_opt = torch.optim.SGD([w], lr=args.w_lr)
    inner_opt = torch.optim.SGD([x], lr=args.x_lr)
    inverse_opt = torch.optim.SGD([v], lr=1)
    for epoch in range(args.epochs):

        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            x.grad = g_x(x, w, trainset)
            dist.all_reduce(x.grad.data, op=dist.ReduceOp.SUM)
            x.grad.data /= args.size
            inner_opt.step()

            if it > args.iterations // 2:
                inverse_opt.zero_grad()
                gx = g_x(x, w, trainset,create_graph=True,retain_graph=True).reshape([-1])
                v.grad =  torch.autograd.grad(torch.matmul(gx, v.detach()), x)[0].reshape([-1])
                dist.all_reduce(v.grad.data, op=dist.ReduceOp.SUM)
                v.grad.data /= args.size
                inverse_opt.step()       
                 
        outer_opt.zero_grad()
        p = args.v_lr * args.iterations * v
        gx = g_x(x, w, trainset,create_graph=True).reshape([-1])
        w.grad =  -torch.autograd.grad(torch.matmul(gx, v.detach()), w, retain_graph=True)[0]
        dist.all_reduce(w.grad.data, op=dist.ReduceOp.SUM)
        w.grad.data /= args.size
        outer_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        if rank == 0:
            test_loss, test_acc = evaluate(x, w, testset)
            teval_loss, teval_acc = evaluate(x, w, tevalset)
            comm_lst.append((epoch+1) * args.iterations )
            loss_lst.append(teval_loss)
            time_lst.append(total_time)
            print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | teval loss {teval_loss:6.4f}")

    if rank == 0:
        filename = f"./{args.log}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_vlr{args.v_lr}_wlr{args.w_lr}_sd{args.seed}"
        with open(filename,'w') as f:
            print(comm_lst,loss_lst,time_lst, sep='\n', file=f)

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.generate_data:
        trainset, valset, testset, tevalset = get_data(args)
        torch.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pt"))
        print(f"[info] successfully generated data to {args.data_path}/l2reg.pt")

    elif args.pretrain:
        trainset, valset, testset, tevalset = torch.load(os.path.join(args.data_path, "l2reg.pt"))
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = args.device
        trainset = (trainset[0].float().to(device), trainset[1].to(device))
        valset   = (valset[0].float().to(device), valset[1].to(device))
        testset  = (testset[0].float().to(device), testset[1].to(device))
        tevalset = (tevalset[0].float().to(device), tevalset[1].to(device))

        # pretrain a model (training without regularization)
        n_feats  = trainset[0].shape[-1]
        num_classes = trainset[1].unique().shape[-1]

        x = torch.randn((n_feats, num_classes), requires_grad=True, device=device)
        x.data = nn.init.kaiming_normal_(x.data.t(), mode='fan_out').t()
        w = torch.zeros(n_feats, requires_grad=True, device=device)

        best_loss, best_acc, x_data = eval("baseline")(args=args,
                                                       x=x,
                                                       w=w,
                                                       trainset=trainset,
                                                       valset=valset,
                                                       testset=testset,
                                                       tevalset=tevalset)
        torch.save(x_data.cpu().data.clone(), f"./save_l2reg/pretrained.pt")

        loss, acc = evaluate(x_data, w, testset)
        torch.save({
            "pretrain_test_loss": loss,
            "pretrain_test_acc": acc,
            }, os.path.join(f"./save_l2reg/pretrained.stats"))
        print(f"[info] Training without regularization results in loss {loss:.2f} acc {acc:.2f}")

    else:
        processes = []
        mp.set_start_method("spawn")
        for rank in range(args.size):

            if args.alg == 'F2BA':
                p = mp.Process(target=init_process, args=(rank, args, F2BA))
            elif args.alg == 'AID':
                p = mp.Process(target=init_process, args=(rank, args, AID))
            elif args.alg == 'ITD':
                p = mp.Process(target=init_process, args=(rank, args, ITD))

            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        # stats = eval(args.alg)(args=args,
        #                        x=x,
        #                        w=w,
        #                        trainset=trainset,
        #                        valset=valset,
        #                        testset=testset,
        #                        tevalset=tevalset,
        #                        size=args.size)

        # if args.alg == "BOME":
        #     save_path = f"./{args.model_path}/{args.alg}u1{args.u1}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        # elif args.alg == 'BVFSM':
        #     save_path = f"./{args.model_path}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"
        # else:
        #     save_path = f"./{args.model_path}/{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_sd{args.seed}"
        # torch.save(stats, save_path)