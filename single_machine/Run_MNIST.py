import torch
import matplotlib.pyplot as plt
import os
import argparse
import math
import numpy as np

os.system("mkdir -p ./trainlogs/")

dataset = 'mnist'
seed = 1
epoch = 50000
iterations = 10

model_path  = 'save_data_cleaning'

w_lr = 100.0
x_lr =0.01
xhat_lr = 0.01
lmbd = 10.0
alg = "F2BA"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)        
F2BA_time = np.array([x[0]  for x in stats])
F2BA_loss = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.1
xhat_lr = 0.1
alg = "ITD"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
ITD_time = np.array([x[0]  for x in stats])
ITD_loss = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.001
xhat_lr = 0.001
alg = "AID"
os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
AID_time = np.array([x[0]  for x in stats])
AID_loss = np.array([x[3] for x in stats])

min_loss = min( min(np.amin(AID_loss), np.amin(ITD_loss)),np.amin(F2BA_loss  ))
AID_loss = AID_loss - min_loss
ITD_loss = ITD_loss - min_loss
F2BA_loss= F2BA_loss - min_loss

plt.rc('font', size=20)
plt.figure()
plt.rc('xtick', labelsize=15)    
plt.rc('ytick', labelsize=15)    
plt.plot(AID_time, AID_loss, ':r', linewidth=3, label = 'AID')
plt.plot(ITD_time, ITD_loss, '--m', linewidth=3, label = 'ITD')
plt.plot(F2BA_time, F2BA_loss, '-k', linewidth=3, label = r'F${}^2$BA')
plt.tick_params(labelsize=20)
plt.xlim(0,40)
plt.ylim(1e-4,1)
#plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel('time(s)',fontsize=20)
plt.ylabel('gap',fontsize=20)
plt.yscale('log')
plt.grid()
plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig(f"./{dataset}.png")
plt.savefig(f"./{dataset}.eps",format='eps')

