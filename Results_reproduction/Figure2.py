import numpy as np
import matplotlib.pyplot as plt

result_dirs = ['results_GEN_weak_regularization', 'results_GEN_strong_regularization']
fold = 2
epoch_1 = 97 # 97_used
epoch_2 = 150  # 150_used

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
ax1_2 = ax1.twinx()
ax3_2 = ax3.twinx()
for result_dir in result_dirs:
    bac_kp_p = np.load(result_dir + f'/balanced_accuracy_{fold}.npy')
    tr_losses = np.load(result_dir + f'/trainLoss_{fold}.npy') # train loss
    loss_kp_p = np.load(result_dir + f'/testLoss_{fold}.npy') # test loss
    acc_kp_p = np.load(result_dir + f'/testAcc_{fold}.npy')
    # train_acc_kp_p = np.load(result_dir + f'/trainAcc_{fold}.npy')

    if result_dir == 'results_GEN_weak_regularization':

        color = 'tab:red'
        ax1.plot(tr_losses, color=color)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax1.set_ylabel('Train Loss', color=color, fontsize=15)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create secondary y-axis for test loss on ax1

        color = 'tab:blue'
        ax1_2.plot(loss_kp_p, color=color)
        ax1_2.set_ylabel('Test Loss', color=color, fontsize=15)
        ax1_2.tick_params(axis='y', labelcolor=color)
        ax1.set_title('(A)', fontsize=15)

        ax2.plot(range(len(acc_kp_p)), acc_kp_p, color='m')
        ax2.plot(range(len(bac_kp_p)), bac_kp_p, color='b')
        ax2.legend(['Test accuracy', 'Test balanced accuracy'], loc='lower right', prop={'size': 14})
        ax2.set_xlabel('Epoch', fontsize=15)
        ax2.set_ylabel('Predictive accuracy', fontsize=15)
        ax2.set_xticks(np.arange(0, len(tr_losses)+1, 50))
        ax2.set_yticks(np.arange(0.3, 0.9, 0.1))
        ax2.set_title('(B)', fontsize=15)
    else:

        color = 'tab:red'
        # color = 'm'
        ax3.plot(tr_losses, color=color)
        ax3.set_xlabel('Epoch', fontsize=15)
        ax3.set_ylabel('Train Loss', color=color, fontsize=15)
        ax3.tick_params(axis='y', labelcolor=color)

        # Create secondary y-axis for test loss on ax2

        color = 'tab:blue'
        # color = 'blue'
        ax3_2.plot(loss_kp_p, color=color)
        ax3_2.set_ylabel('Test Loss', color=color, fontsize=15)
        ax3_2.tick_params(axis='y', labelcolor=color)
        ax3.set_title('(C)', fontsize=15)

        ax4.plot(range(len(acc_kp_p)), acc_kp_p, color='m')
        ax4.plot(range(len(bac_kp_p)), bac_kp_p, color='b')
        ax4.legend(['Test accuracy', 'Test balanced accuracy'], loc='lower right', prop={'size': 14})
        ax4.set_xlabel('Epoch', fontsize=15)
        ax4.set_ylabel('Predictive accuracy', fontsize=15)
        ax4.set_xticks(np.arange(0, 201, 50))
        ax4.set_yticks(np.arange(0.3, 0.9, 0.1))
        ax4.set_title('(D)', fontsize=15)

tick_size = 13.5
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
ax1_2.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)
ax3.tick_params(axis='both', which='major', labelsize=tick_size)
ax3_2.tick_params(axis='both', which='major', labelsize=tick_size)
ax4.tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.savefig('Figure 2.png')
plt.show()
