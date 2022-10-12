import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="dark")
number = np.random.randint(1000000, size=1)
rs = np.random.RandomState(number)
print(number)
sns.set(font_scale = 0)
# Set up the matplotlib figure
# f, axes = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
# for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):
# Create a cubehelix colormap to use with kdeplot
cmap = sns.cubehelix_palette(as_cmap=True)

# Generate and plot a random bivariate dataset
data = np.random.multivariate_normal(mean=np.asarray([3, 0]), cov=np.eye(2), size=50)
data_no = np.random.multivariate_normal(mean=np.asarray([0,3]), cov=np.eye(2), size=100)
# data_no =np.random.normal(6.2, 3.5, size=(1000,2))
x_no = data_no[:,0]
y_no = data_no[:,1]
# breakpoint()
x = np.concatenate([data[:, 0], x_no],-1)
y = np.concatenate([data[:, 1], y_no],-1)


estimated_meanx = np.mean(x)
estimated_meany = np.mean(y)
estimated_covariance = np.cov(np.concatenate([data, data_no], 0).T)
fig, ax = plt.subplots()
data_new = np.random.multivariate_normal(mean=np.asarray([estimated_meanx, estimated_meany]), cov=estimated_covariance, size=1000)

# cbar = fig.colorbar(cax, ticks=[-1, 0])
# cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
# cbar.ax.set_yticklabels(['low ID score', 'high ID score'])




x1, y1 = rs.normal(size=(2, 50))
sns.kdeplot(
    x=x1, y=y1,
shade=True,
    cmap=cmap,
cbar=True,
)

# ax.set_axis_off()

sns.scatterplot(x=x1,y=y1, s=3,color=".5")
plt.xticks([])
plt.yticks([])
plt.xlim([-3, 6])
plt.ylim([-3, 6])
# ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
# f.subplots_adjust(0, 0, 1, 1, .08, .08)
plt.savefig('fig2.pdf')

breakpoint()
sns.kdeplot(
    x=x, y=y,
shade=True,
    cmap=cmap,
cbar=True,
# cbar_kws=cbar
)

sns.scatterplot(x=x,y=y, s=3,color=".5")
plt.xticks([])
plt.yticks([])
plt.xlim([-3, 6])
plt.ylim([-3, 6])
# ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
# f.subplots_adjust(0, 0, 1, 1, .08, .08)
plt.savefig('fig1.pdf')

plt.clf()
# cbar = fig.colorbar(cax, ticks=[-1, 0])
# cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar
# cbar.ax.set_yticklabels(['low ID score', 'high ID score'])
sns.kdeplot(
    x=data_new[:,0], y=data_new[:,1],
shade=True,
    cmap=cmap,
cbar=True,
)

# ax.set_axis_off()

sns.scatterplot(x=x,y=y, s=3,color=".5")
plt.xticks([])
plt.yticks([])
plt.xlim([-3, 6])
plt.ylim([-3, 6])
# ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
# f.subplots_adjust(0, 0, 1, 1, .08, .08)
plt.savefig('fig2.pdf')