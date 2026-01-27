import matplotlib.pyplot as plt 
import torch 
import matplotlib as mpl

def path_to_lst(path):
    stats = torch.load(path)
    total_time = [s[0] for s in stats]
    test_loss = [s[1] for s in stats]
    test_acc = [s[2] for s in stats]

    return total_time, test_loss, test_acc

# Baseline stochastic Hessian-vector-product-based method stocBiO
path_stocBiO = './save_l2reg/stocBiO_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'

# Enhanced with variance reduction but requires mean-square smoothness assumptions
path_MRBO = './save_l2reg/MRBO_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'
path_VRBO = './save_l2reg/VRBO_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'

# Baseline stochastic fully first-order method F2SA
path_F2SA = './save_l2reg/F2SA_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'

# Enhanced with central difference and normalized gradient descent under higher-order smoothness assumptions
path_F2SA_2 = './save_l2reg/F2SA_p_p2_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'

path_F2SA_3 = './save_l2reg/F2SA_p_p3_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'
path_F2SA_4 = './save_l2reg/F2SA_p_p4_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'
path_F2SA_5 = './save_l2reg/F2SA_p_p5_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'
path_F2SA_8 = './save_l2reg/F2SA_p_p8_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'
path_F2SA_10 = './save_l2reg/F2SA_p_p10_k10_xlr100_xm0.0_wlr0.1_wm0.0_sd1_lmbd10'

total_time_F2SA, test_loss_F2SA, test_acc_F2SA = path_to_lst(path_F2SA)
total_time_F2SA_2, test_loss_F2SA_2, test_acc_F2SA_2 = path_to_lst(path_F2SA_2)
total_time_F2SA_3, test_loss_F2SA_3, test_acc_F2SA_3 = path_to_lst(path_F2SA_3)
total_time_F2SA_5, test_loss_F2SA_5, test_acc_F2SA_5 = path_to_lst(path_F2SA_5)
total_time_F2SA_8, test_loss_F2SA_8, test_acc_F2SA_8 = path_to_lst(path_F2SA_8)     
total_time_F2SA_10, test_loss_F2SA_10, test_acc_F2SA_10 = path_to_lst(path_F2SA_10)

total_time_stocBiO, test_loss_stocBiO, test_acc_stocBiO = path_to_lst(path_stocBiO)
total_time_MRBO, test_loss_MRBO, test_acc_MRBO = path_to_lst(path_MRBO)
total_time_VRBO, test_loss_VRBO, test_acc_VRBO = path_to_lst(path_VRBO)

# print(f"Total Time \nF2SA: {total_time_F2SA[-1]} \nF2SA-2: {total_time_F2SA_2[-1]} \nstocBiO: {total_time_stocBiO[-1]}")
# print(f"MRBO: {total_time_MRBO[-1]} \nVRBO: {total_time_VRBO[-1]}")

# Make subplot spacing larger and outer margins tighter
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05
mpl.rcParams['figure.subplot.left'] = 0.06
mpl.rcParams['figure.subplot.right'] = 0.98
mpl.rcParams['figure.subplot.top'] = 0.92
mpl.rcParams['figure.subplot.bottom'] = 0.18

# Wrap subplots_adjust so later calls can't force tiny wspace or large margins
__sa_real = plt.subplots_adjust
def __sa_normalized(*args, **kwargs):
    # prefer wider spacing between the two subplots
    if 'wspace' not in kwargs or kwargs['wspace'] < 0.1:
        kwargs['wspace'] = 0.1
    # tighten outer margins
    if 'left' not in kwargs or kwargs['left'] > 0.08:
        kwargs['left'] = 0.06
    if 'right' not in kwargs or kwargs['right'] < 0.98:
        kwargs['right'] = 0.98
    return __sa_real(*args, **kwargs)
plt.subplots_adjust = __sa_normalized
# Make horizontal spacing between subplots tighter
mpl.rcParams['figure.subplot.wspace'] = 0.15

pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
loss = pretrained_stats["pretrain_test_loss"]
acc  = pretrained_stats["pretrain_test_acc"]
font_size = 32
plt.rc('font' , size = font_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

ax1.plot(test_loss_F2SA, linewidth=3, label=r'F${}^2$SA')
ax1.plot(test_loss_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax1.plot(test_loss_F2SA_3, linewidth=3, label=r'F${}^2$SA-3')
ax1.plot(test_loss_F2SA_5, linewidth=3, label=r'F${}^2$SA-5')
ax1.plot(test_loss_F2SA_8, linewidth=3, label=r'F${}^2$SA-8')
ax1.plot(test_loss_F2SA_10, linewidth=3, label=r'F${}^2$SA-10')
ax1.plot(test_loss_VRBO, linewidth=3, label='VRBO')
ax1.plot(test_loss_MRBO, linewidth=3, label='MRBO')
ax1.plot(test_loss_stocBiO, linewidth=3, label='stocBiO')
ax1.axhline(y=loss, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax1.set_xlabel('#Iterations')
ax1.set_ylabel('Test Loss')

ax1.grid()
ax2.plot(test_acc_F2SA, linewidth=3, label=r'F${}^2$SA')
ax2.plot(test_acc_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax2.plot(test_acc_F2SA_3, linewidth=3, label=r'F${}^2$SA-3')
ax2.plot(test_acc_F2SA_5, linewidth=3, label=r'F${}^2$SA-5')
ax2.plot(test_acc_F2SA_8, linewidth=3, label=r'F${}^2$SA-8')
ax2.plot(test_acc_F2SA_10, linewidth=3, label=r'F${}^2$SA-10')  
ax2.plot(test_acc_VRBO, linewidth=3, label='VRBO')
ax2.plot(test_acc_MRBO, linewidth=3, label='MRBO')
ax2.plot(test_acc_stocBiO, linewidth=3, label='stocBiO')
ax2.axhline(y=acc, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax2.set_xlabel('#Iterations')
ax2.set_ylabel('Test Accuracy')
ax2.grid()

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=font_size, frameon=False)

plt.subplots_adjust(wspace=0.35, bottom=0.3, top=0.9)
plt.savefig('./save_l2reg/l2reg_stoc.png')
plt.savefig('./save_l2reg/l2reg_stoc.pdf', format='pdf')

# another figure for non-parallel complexity

x_MRBO = [40*i for i in range(len(test_loss_MRBO))]
x_VRBO = [40*i for i in range(len(test_loss_VRBO))]
x_stocBiO = [40*i for i in range(len(test_loss_stocBiO))]
x_F2SA = [20*i for i in range(len(test_loss_F2SA))]
x_F2SA_2 = [20*i for i in range(len(test_loss_F2SA_2))]
x_F2SA_3 = [40*i for i in range(len(test_loss_F2SA_3))]
x_F2SA_5 = [60*i for i in range(len(test_loss_F2SA_5))]
x_F2SA_8 = [80*i for i in range(len(test_loss_F2SA_8))]
x_F2SA_10 = [100*i for i in range(len(test_loss_F2SA_10))]

plt.rc('font' , size = font_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
ax1.plot(x_F2SA, test_loss_F2SA, linewidth=3, label=r'F${}^2$SA')
ax1.plot(x_F2SA_2, test_loss_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax1.plot(x_F2SA_3[:2000//4], test_loss_F2SA_3[:2000//4], linewidth=3, label=r'F${}^2$SA-3')
ax1.plot(x_F2SA_5[:2000//6], test_loss_F2SA_5[:2000//6], linewidth=3, label=r'F${}^2$SA-5')
ax1.plot(x_F2SA_8[:2000//8], test_loss_F2SA_8[:2000//8], linewidth=3, label=r'F${}^2$SA-8')
ax1.plot(x_F2SA_10[:2000//10], test_loss_F2SA_10[:2000//10], linewidth=3, label=r'F${}^2$SA-10')
ax1.plot(x_VRBO[:2000//4], test_loss_VRBO[:2000//4], linewidth=3, label='VRBO')
ax1.plot(x_MRBO[:2000//4], test_loss_MRBO[:2000//4], linewidth=3, label='MRBO')
ax1.plot(x_stocBiO[:2000//4], test_loss_stocBiO[:2000//4], linewidth=3, label='stocBiO')
ax1.axhline(y=loss, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax1.set_xlabel('#Gradients Evaluations')
ax1.set_ylabel('Test Loss')

ax1.grid()
ax2.plot(x_F2SA, test_acc_F2SA, linewidth=3, label=r'F${}^2$SA')
ax2.plot(x_F2SA_2, test_acc_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax2.plot(x_F2SA_3[:2000//4], test_acc_F2SA_3[:2000//4], linewidth=3, label=r'F${}^2$SA-3')
ax2.plot(x_F2SA_5[:2000//6], test_acc_F2SA_5[:2000//6], linewidth=3, label=r'F${}^2$SA-5')
ax2.plot(x_F2SA_8[:2000//8], test_acc_F2SA_8[:2000//8], linewidth=3, label=r'F${}^2$SA-8')
ax2.plot(x_F2SA_10[:2000//10], test_acc_F2SA_10[:2000//10], linewidth=3, label=r'F${}^2$SA-10')  
ax2.plot(x_VRBO[:2000//4], test_acc_VRBO[:2000//4], linewidth=3, label='VRBO') 
ax2.plot(x_MRBO[:2000//4], test_acc_MRBO[:2000//4], linewidth=3, label='MRBO')
ax2.plot(x_stocBiO[:2000//4], test_acc_stocBiO[:2000//4], linewidth=3, label='stocBiO')
ax2.axhline(y=acc, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax2.set_xlabel('#Gradient Evaluations')
ax2.set_ylabel('Test Accuracy')
ax2.grid()

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=font_size, frameon=False)    
plt.subplots_adjust(wspace=0.35, bottom=0.3, top=0.9)
plt.savefig('./save_l2reg/l2reg_stoc_grad.png')
plt.savefig('./save_l2reg/l2reg_stoc_grad.pdf', format='pdf')
