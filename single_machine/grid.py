import os
import argparse
import torch 
import matplotlib.pyplot as plt

os.system("mkdir -p ./trainlogs/")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion", "l2reg"])
args = parser.parse_args()

model_path  = 'save_data_cleaning'
w_lr_list = [10.0,  100.0,  1000.0]
lmbd_list = [10.0,  100.0,  1000.0]
iterations_list = [5]
seed = 42

dataset = args.dataset
epoch = 5000

if args.dataset in ["mnist", "fashion"]:    
    x_lr = 0.01
    xhat_lr = 0.01

    alg = 'F2BA'
    F2BA_minloss  = 1e6
    F2BA_time = []
    F2BA_loss = []
    for w_lr in w_lr_list:
        for lmbd in lmbd_list:
            for iterations in iterations_list:
                os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_lmbd{lmbd}_{seed}.log")
                loss = save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
                stats = torch.load(save_path)  
                if stats[3][-1] < F2BA_minloss:
                    F2BA_minloss =  stats[3][-1]
                    F2BA_time = [x[0]  for x in stats]
                    F2BA_loss = [x[3] for x in stats]
    
    alg = "ITD"
    ITD_minloss = 1e6
    for w_lr in w_lr_list:
        for iterations in iterations_list:
            os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")
            save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
            stats = torch.load(save_path)  
            if stats[3][-1] < ITD_minloss:
                ITD_minloss =  stats[3][-1]
                ITD_time = [x[0]  for x in stats]
                ITD_loss = [x[3] for x in stats]
                

    alg = "AID_CG"
    AID_minloss = 1e6
    for w_lr in w_lr_list:
        for iterations in iterations_list:
            os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")
            save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
            stats = torch.load(save_path)  
            if stats[3][-1] < ITD_minloss:
                AID_minloss =  stats[3][-1]
                AID_time = [x[0]  for x in stats]
                AID_loss = [x[3] for x in stats]
    
    
    # alg = "AID_FP"
    # for w_lr in w_lr_list:
    #     os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")
    
    # alg = "reverse"
    # for w_lr in w_lr_list:
    #     os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

else:
    x_lr  = 100.0
    xhat_lr = 100.0 

    alg = 'F2BA'
    for w_lr in w_lr_list:
        os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

    alg = "ITD"
    for w_lr in w_lr_list:
        os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

    alg = "AID_CG"
    for w_lr in w_lr_list:
        os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

    alg = "AID_FP"
    for w_lr in w_lr_list:
        os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")
    
    alg = "reverse"
    for w_lr in w_lr_list:
        os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{x_lr}_{seed}.log")

plt.figure()
plt.rc('xtick', labelsize=15)    
plt.rc('ytick', labelsize=15)    
plt.plot(AID_time, AID_loss, ':r', linewidth=3, label = 'AID')
plt.plot(ITD_time, ITD_loss, '--m', linewidth=3, label = 'ITD')
plt.plot(F2BA_time, F2BA_loss, '-k', linewidth=3, label = r'F${}^2$BA')
plt.tick_params(labelsize=20)
plt.xlim(0,F2BA_time[-1])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('time(s)',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid()
plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig(f"./{dataset}.png")
plt.savefig(f"./{dataset}.eps",format='eps')