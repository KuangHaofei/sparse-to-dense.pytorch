import matplotlib.pyplot as plt
import numpy as np

#10 100, 300, 500
mse = [0.188, 0.149, 0.147, 0.133]
rmse = [0.398, 0.342, 0.351, 0.335]
absrel = [0.092, 0.057, 0.062, 0.058]
mae = [0.188, 0.129, 0.136, 0.128]
delta1 = [0.916, 0.957, 0.951, 0.955]

# plt.title('Training loss')
# plt.plot(mse, '-o')
# plt.plot(rmse, '-o')
# plt.plot(absrel, '-o')
# plt.plot(mae, '-o')
# # plt.plot(delta1, '-o')
#
# plt.xlabel('Samples')
# # plt.xticks([100, 300, 500])
# plt.ylabel('loss')
#
# plt.show()

loss = np.array(mse)

# file = open('loss.txt', 'w')
#
# file.write(str(mse))
# file.close()

np.savetxt('loss.txt', loss)