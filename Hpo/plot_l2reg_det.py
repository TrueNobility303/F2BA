import matplotlib.pyplot as plt 
import torch 

def path_to_lst(path):
    stats = torch.load(path)
    total_time = [s[0] for s in stats]
    test_loss = [s[1] for s in stats]
    test_acc = [s[2] for s in stats]
    return total_time, test_loss, test_acc
                                                                              
# Baseline Hessian-vector-product-based method AID
path_AID = './save_l2reg/AID_FP_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'

# Baseline fully first-order methods
path_F2BA = './save_l2reg/F2BA_k10_xlr100_xm0.0_wlr1000_wm0.0_sd1_lmbd10'
path_AccF2BA = './save_l2reg/AccF2BA_k10_xlr100_xm0.0_wlr1000_wm0.9_sd1_lmbd10'

# F2BA and AccF2BA Enhanced with lower-level AGD
path_F2BA_ = './save_l2reg/F2BA_k10_xlr100_xm0.5_wlr1000_wm0.0_sd1_lmbd10'
path_AccF2BA_= './save_l2reg/AccF2BA_k10_xlr100_xm0.5_wlr1000_wm0.9_sd1_lmbd10'

total_time_AID, test_loss_AID, test_acc_AID = path_to_lst(path_AID)
total_time_F2BA, test_loss_F2BA, test_acc_F2BA = path_to_lst(path_F2BA)
total_time_AccF2BA, test_loss_AccF2BA, test_acc_AccF2BA = path_to_lst(path_AccF2BA)
total_time_F2BA_, test_loss_F2BA_, test_acc_F2BA_ = path_to_lst(path_F2BA_)
total_time_AccF2BA_, test_loss_AccF2BA_, test_acc_AccF2BA_ = path_to_lst(path_AccF2BA_)

print(f"Total Time \nAID: {total_time_AID[-1]} \nF2BA: {total_time_F2BA[-1]} \nAccF2BA: {total_time_AccF2BA[-1]}")
print(f"F2BA+: {total_time_F2BA_[-1]} \nAccF2BA+: {total_time_AccF2BA_[-1]}")

pretrained_stats = torch.load("./save_l2reg/pretrained.stats")
loss = pretrained_stats["pretrain_test_loss"]
acc  = pretrained_stats["pretrain_test_acc"]

font_size = 24
plt.rc('font' , size = font_size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.plot(test_loss_F2BA_, linewidth=3, label=r'F${}^2$BA$^+$')
ax1.plot(test_loss_AccF2BA_, linewidth=3, label=r'AccF${}^2$BA$^+$')
ax1.plot(test_loss_F2BA, linewidth=3, label=r'F${}^2$BA')
ax1.plot(test_loss_AccF2BA, linewidth=3, label=r'AccF${}^2$BA')
ax1.plot(test_loss_AID, linewidth=3, label='AID')
ax1.axhline(y=loss, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax1.set_xlabel('#Total Iterations')
ax1.set_ylabel('Test Loss')
ax1.grid()

ax2.plot(test_acc_F2BA_, linewidth=3, label=r'F${}^2$BA$^+$')
ax2.plot(test_acc_AccF2BA_, linewidth=3, label=r'AccF${}^2$BA$^+$')
ax2.plot(test_acc_F2BA, linewidth=3, label=r'F${}^2$BA')
ax2.plot(test_acc_AccF2BA, linewidth=3, label=r'AccF${}^2$BA')
ax2.plot(test_acc_AID, linewidth=3, label='AID')
ax2.axhline(y=acc, color='#ADD8E6', linewidth=3, linestyle='--', label='w/o Reg')
ax2.set_xlabel('#Total Iterations')
ax2.set_ylabel('Test Accuracy')
ax2.grid()

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=font_size, frameon=False)

plt.subplots_adjust(wspace=0.6, bottom=0.3, top=0.9)
plt.savefig('./save_l2reg/l2reg_det.png')
plt.savefig('./save_l2reg/l2reg_det.pdf', format='pdf')