import os
from Physics import physics_loss
import torch
import matplotlib.pyplot as plt
from NN import Net
from Data import t_test, f_train, u_train, i_train, data_train, f_test, u_test, i_test, data_test

# Hice un cambio para ver que pasaba
# _______________________________________________________________________________
model_path_1 = 'NN_model_2.pth'
# model_path_2 = 'PINN_model_1.pth'

# _______________________________________________________________________________
window_size = 4
input_dim = 2 * window_size
out_dim = 1

net1 = Net(input_dim, out_dim, loss2=None, n_units=100, epochs=5000, net_weight=1, physic_weight=0, lr=1e-4)

net2 = Net(input_dim, out_dim, loss2=physics_loss, n_units=100, epochs=5000, net_weight=1, physic_weight=0.05, lr=1e-4)

if os.path.exists(model_path_1):
    net1.load_state_dict(torch.load(model_path_1))
    net1.eval()
else:
    losses1 = net1.fit(u_train, i_train, data_train[:, 0], window_size)  # data train
    torch.save(net1.state_dict(), model_path_1)

#if os.path.exists(model_path_2):
#    net2.load_state_dict(torch.load(model_path_2))
#    net2.eval()
#else:
#    losses2 = net2.fit(f_train, u_train, i_train, data_train[:, 0], window_size)  # data train
#    torch.save(net2.state_dict(), model_path_2)

# _______________________________________________________________________________
prediction_1 = net1.predict(u_test, i_test, window_size)
# prediction_2 = net2.predict(f_te u_test, i_test, window_size)

plt.figure()
plt.plot(t_test, data_test[:, 0], alpha=1, label='data_displacement')
plt.plot(t_test[:(len(t_test)-window_size+1)], prediction_1[:, 0], alpha=0.7, label='NN')
plt.legend()
plt.ylabel('Dis (mm)')
plt.xlabel('Time (s)')

#plt.figure()
#plt.plot(t_test, data_test[:, 0], alpha=0.8, label='data_displacement')
#plt.plot(t_test[:(len(t_test)-window_size+1)], prediction_2[:, 0], alpha=0.7, label='PINN')
#plt.legend()
#plt.ylabel('Dis (mm)')
#plt.xlabel('Time (s)')
plt.show()
