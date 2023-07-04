import matplotlib.pyplot as plt 
import numpy as np
import torch 

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
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)        
F2BA_time_MNIST = np.array([x[0]  for x in stats])
F2BA_loss_MNIST = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.1
xhat_lr = 0.1
alg = "ITD"
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
ITD_time_MNIST = np.array([x[0]  for x in stats])
ITD_loss_MNIST = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.001
xhat_lr = 0.001
alg = "AID_CG"
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
AID_time_MNIST = np.array([x[0]  for x in stats])
AID_loss_MNIST = np.array([x[3] for x in stats])

min_loss_MNIST = min( min(np.amin(AID_loss_MNIST), np.amin(ITD_loss_MNIST)),np.amin(F2BA_loss_MNIST  ))
AID_loss_MNIST = AID_loss_MNIST - min_loss_MNIST
ITD_loss_MNIST = ITD_loss_MNIST - min_loss_MNIST
F2BA_loss_MNIST= F2BA_loss_MNIST - min_loss_MNIST

dataset = 'fashion'
seed = 1
epoch = 50000
iterations = 10

model_path  = 'save_data_cleaning'

w_lr = 100.0
x_lr =0.01
xhat_lr = 0.01
lmbd = 10.0
alg = "F2BA"
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --lmbd {lmbd} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr}  --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_lmbd{lmbd}_sd{seed}"
stats = torch.load(save_path)        
F2BA_time_Fashion = np.array([x[0]  for x in stats])
F2BA_loss_Fashion = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.1
xhat_lr = 0.1
alg = "ITD"
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
ITD_time_Fashion = np.array([x[0]  for x in stats])
ITD_loss_Fashion = np.array([x[3] for x in stats])

w_lr = 100.0
x_lr =0.001
xhat_lr = 0.001
alg = "AID_CG"
#os.system(f"python data_cleaning.py --dataset {dataset} --alg {alg} --epochs {epoch} --seed {seed} --iterations {iterations} --x_lr {x_lr} --w_lr {w_lr} --xhat_lr {xhat_lr} > trainlogs/{dataset}_{alg}_{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_{seed}.log")
save_path = f"./{model_path}/{dataset}_{alg}_k{iterations}_xlr{x_lr}_wlr{w_lr}_xhatlr{xhat_lr}_sd{seed}"
stats = torch.load(save_path)        
AID_time_Fashion = np.array([x[0]  for x in stats])
AID_loss_Fashion = np.array([x[3] for x in stats])

min_loss_Fashion = min( min(np.amin(AID_loss_Fashion), np.amin(ITD_loss_Fashion)),np.amin(F2BA_loss_Fashion  ))
AID_loss_Fashion = AID_loss_Fashion - min_loss_Fashion
ITD_loss_Fashion = ITD_loss_Fashion - min_loss_Fashion
F2BA_loss_Fashion= F2BA_loss_Fashion - min_loss_Fashion

font_size = 25
plt.rc('font' , size = font_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.plot(AID_time_MNIST, AID_loss_MNIST, ':r', linewidth=3, label = 'AID')
ax1.plot(ITD_time_MNIST, ITD_loss_MNIST, '--m', linewidth=3, label = 'ITD')
ax1.plot(F2BA_time_MNIST, F2BA_loss_MNIST, '-k', linewidth=3, label = r'F${}^2$BA (Ours)')
ax1.set_xlim(0,40)
ax1.set_ylim(1e-4,1)
ax1.set_yscale('log')
ax2.tick_params(axis='both', which='major', labelsize=font_size)
ax1.set_xlabel('time(s)',fontsize=font_size)
ax1.set_ylabel('gap',fontsize=font_size)
ax1.grid()

ax2.plot(AID_time_Fashion, AID_loss_Fashion, ':r', linewidth=3, label = 'AID')
ax2.plot(ITD_time_Fashion, ITD_loss_Fashion, '--m', linewidth=3, label = 'ITD')
ax2.plot(F2BA_time_Fashion, F2BA_loss_Fashion, '-k', linewidth=3, label = r'F${}^2$BA (Ours)')
ax2.set_xlim(0,60)
ax2.set_ylim(1e-4,1)
ax2.set_yscale('log')
ax2.tick_params(axis='both', which='major', labelsize=font_size)
ax2.set_xlabel('time(s)',fontsize=font_size)
ax2.set_ylabel('gap',fontsize=font_size)
ax2.grid()

# Add a unified legend at the right-hand side of the plots
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=25,
            frameon=False)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.6, bottom=0.3, top=0.9)
#plt.tight_layout()
# Show the plot
plt.show()
plt.savefig('data_cleaning.png',  dpi=500)
plt.savefig('data_cleaning.eps', format='eps')