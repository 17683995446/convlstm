import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import cv2

list_0 = np.zeros((224,224,16),dtype=float)



fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 224, 1)
Y = np.arange(0, 224, 1)
X, Y = np.meshgrid(X, Y)
img = cv2.resize(cv2.imread('F:\DeepLearning\sa_convLSTM-master\outputs\\train_val_loss_curve_epoch_0.png'), (224, 224))
blueimg = img[:, :, 0]  # 需要哪个通道的三维图，选择哪个通道即可。


for i in range(224):
    for j in range(224):
        for k in range(16):
            list_0[i][j][k]=blueimg[i][j]/16
print(list_0[100][100])


surf = ax.plot_surface(X, Y, blueimg, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 255)  # z轴的取值范围
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()