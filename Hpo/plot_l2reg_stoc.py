import matplotlib.pyplot as plt 
import torch 

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
path_F2SA_2 = './save_l2reg/F2SA_2_k10_xlr100_xm0.0_wlr1.0_wm0.0_sd1_lmbd10'

total_time_F2SA, test_loss_F2SA, test_acc_F2SA = path_to_lst(path_F2SA)
total_time_F2SA_2, test_loss_F2SA_2, test_acc_F2SA_2 = path_to_lst(path_F2SA_2)
total_time_stocBiO, test_loss_stocBiO, test_acc_stocBiO = path_to_lst(path_stocBiO)
total_time_MRBO, test_loss_MRBO, test_acc_MRBO = path_to_lst(path_MRBO)
total_time_VRBO, test_loss_VRBO, test_acc_VRBO = path_to_lst(path_VRBO)

print(f"Total Time \nF2SA: {total_time_F2SA[-1]} \nF2SA-2: {total_time_F2SA_2[-1]} \nstocBiO: {total_time_stocBiO[-1]}")
print(f"MRBO: {total_time_MRBO[-1]} \nVRBO: {total_time_VRBO[-1]}")

pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
loss = pretrained_stats["pretrain_test_loss"]
acc  = pretrained_stats["pretrain_test_acc"]

font_size = 24
plt.rc('font' , size = font_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.plot(test_loss_F2SA, linewidth=3, label=r'F${}^2$SA')
ax1.plot(test_loss_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax1.plot(test_loss_VRBO, linewidth=3, label='VRBO')
ax1.plot(test_loss_stocBiO, linewidth=3, label='stocBiO')
ax1.plot(test_loss_MRBO, linewidth=3, label='MRBO')
ax1.axhline(y=loss, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax1.set_xlabel('#Total Iterations')
ax1.set_ylabel('Test Loss')
ax1.grid()

ax2.plot(test_acc_F2SA, linewidth=3, label=r'F${}^2$SA')
ax2.plot(test_acc_F2SA_2, linewidth=3, label=r'F${}^2$SA-2')
ax2.plot(test_acc_VRBO, linewidth=3, label='VRBO')
ax2.plot(test_acc_stocBiO, linewidth=3, label='stocBiO')
ax2.plot(test_acc_MRBO, linewidth=3, label='MRBO')
ax2.axhline(y=acc, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax2.set_xlabel('#Total Iterations')
ax2.set_ylabel('Test Accuracy')
ax2.grid()

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=font_size, frameon=False)

plt.subplots_adjust(wspace=0.6, bottom=0.3, top=0.9)
plt.savefig('./save_l2reg/l2reg_stoc.png')
plt.savefig('./save_l2reg/l2reg_stoc.pdf', format='pdf')