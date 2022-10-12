"""
Plotting/sampling points on a sphere.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from vMF import *

__all__ = ['Sphere', 'vMFSample', 'Coord3D']


class Coord3D():
    """
    Store 3D coordinates.
    """

    def __init__(self, x, y, z):
        """
        Store 3D coordinates.
        """
        self.x = x
        self.y = y
        self.z = z


class Sphere():
    """
    Plotting/sampling points on a sphere.
    """

    def __init__(self):
        """
        Plotting points on a sphere.
        """
        self.sphere = []
        self.samples = []
        self.distribution = ['uniform', 'vMF']

    def _make_sphere(self, radius):
        """
        Get mesh for a sphere of a given radius.
        """
        pi = np.pi
        cos = np.cos
        sin = np.sin
        radius = radius
        phi, theta = np.mgrid[0.0:pi:200j, 0.0:2.0 * pi:200j]
        x = radius * sin(phi) * cos(theta)
        y = radius * sin(phi) * sin(theta)
        z = radius * cos(phi)
        # breakpoint()
        self.sphere = Coord3D(x, y, z)
        return self.sphere

    def _draw_sphere(self, ax, radius):
        """
        Draw a sphere on a given axis.
        """
        sphere = self._make_sphere(radius)  # subtract a little so points show on sphere
        ax.plot_surface(sphere.x, sphere.y, sphere.z,
                        rstride=1, cstride=1,
                        color=sns.xkcd_rgb["light grey"],
                        alpha=0.5,
                        linewidth=0)
        return sphere




    def plot(self, radius=1, data=None):
        """
        Plot a sphere with samples superposed, if supplied.

        :param radius: radius of the base sphere
        :param data: list of sample objects
        """

        # sns.set_style('dark')
        sns.set(context="paper", style="white")
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')

        # plot the sphere
        grid_points = self._draw_sphere(ax, radius)

        if data == None:
            data = self.samples

        is_list = True
        try:
            N = len(data)
        except:
            N = 1
            is_list = False

        palette = sns.color_palette("GnBu_d", N)
        # colors = [palette[i] for i in reversed(range(N))]
        colors = [np.array([0., 1., 0.], dtype=np.float32), np.array([1., 0., 0.], dtype=np.float32),
                  np.array([0., 0., 1.], dtype=np.float32)]
        # plot data, if supplied
        if is_list:
            data_check = data[0]
        else:
            data_check = data

        if type(data_check) is vMFSample:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='$class = $' + str(d.label), color=colors[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='$class = $' + str(data.label), color=colors[i])


        elif type(data_check) is Coord3D:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='uniform samples', color=palette[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='uniform samples', color=palette[i])


        else:
            print('Error: data type not recognised')

        ax.set_axis_off()
        # ax.legend(bbox_to_anchor=[0.65, 0.75])
        plt.savefig('toy.jpg', dpi=250)
        return grid_points

    def _draw_sphere_add(self, ax, radius, fig, fcolors):
        """
        Draw a sphere on a given axis.
        """
        sphere = self._make_sphere(radius - 0.01)  # subtract a little so points show on sphere
        surf = ax.plot_surface(sphere.x, sphere.y, sphere.z,
                               rstride=1, cstride=1,
                               cmap = cm.coolwarm,
                               facecolors=fcolors,
                               alpha=1.0,
                               linewidth=0)
        # ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

        # rotate the axes and update
        # for angle in range(0, 360):
        # ax.view_init(30, 0, vertical_axis='y')
        ax.view_init(30, 240, vertical_axis='y')
            # plt.draw()
            # plt.pause(.001)

        fig.colorbar(surf, shrink=0.5, aspect=5)

    def plot_add(self, radius=1, data=None, fcolors=None):
        """
        Plot a sphere with samples superposed, if supplied.

        :param radius: radius of the base sphere
        :param data: list of sample objects
        """

        # sns.set_style('dark')
        sns.set(context="paper", style="white")
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')
        # breakpoint()
        # plot the sphere
        self._draw_sphere_add(ax, radius, fig, fcolors)

        if data == None:
            data = self.samples

        is_list = True
        try:
            N = len(data)
        except:
            N = 1
            is_list = False

        palette = sns.color_palette("GnBu_d", N)
        # colors = [palette[i] for i in reversed(range(N))]
        colors = [np.array([0., 1., 0.], dtype=np.float32), np.array([1., 0., 0.], dtype=np.float32),
                  np.array([0., 0., 1.], dtype=np.float32)]
        # plot data, if supplied
        if is_list:
            data_check = data[0]
        else:
            data_check = data

        if type(data_check) is vMFSample:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='$class = $' + str(d.label), color=colors[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='$class = $' + str(data.label), color=colors[i])


        elif type(data_check) is Coord3D:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='uniform samples', color=palette[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='uniform samples', color=palette[i])


        else:
            print('Error: data type not recognised')

        ax.set_axis_off()
        ax.legend(bbox_to_anchor=[0.95, 0.75])
        plt.savefig('toy_new_after_unbalanced_binary_vmf.jpg', dpi=250)

    def _draw_sphere_outliers(self, ax, radius):
        """
        Draw a sphere on a given axis.
        """
        sphere = self._make_sphere(radius - 0.01)  # subtract a little so points show on sphere
        surf = ax.plot_surface(sphere.x, sphere.y, sphere.z,
                        rstride=1, cstride=1,
                        color=sns.xkcd_rgb["light grey"],
                        alpha=0.5,
                        linewidth=0)

        # ax.view_init(30, 0, vertical_axis='y')
        ax.view_init(15, 0, vertical_axis='y')



    def plot_outliers(self, radius=1, data=None, outliers=None):
        """
        Plot a sphere with samples superposed, if supplied.

        :param radius: radius of the base sphere
        :param data: list of sample objects
        """

        # sns.set_style('dark')
        sns.set(context="paper", style="white")
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')
        # breakpoint()
        # plot the sphere
        self._draw_sphere_outliers(ax, radius)

        if data == None:
            data = self.samples

        is_list = True
        try:
            N = len(data)
        except:
            N = 1
            is_list = False

        palette = sns.color_palette("GnBu_d", N)
        # colors = [palette[i] for i in reversed(range(N))]
        colors = [np.array([0., 1., 0.], dtype=np.float32), np.array([1., 0., 0.], dtype=np.float32),
                  np.array([0., 0., 1.], dtype=np.float32)]
        # plot data, if supplied
        if is_list:
            data_check = data[0]
        else:
            data_check = data

        if type(data_check) is vMFSample:
            i = 0
            # breakpoint()
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='$class = $' + str(d.label), color=colors[i])
                    i += 1
                ax.scatter(outliers[:,0]*1.1, outliers[:,1]*1.1, outliers[:,2]*1.1, s=50, alpha=1.0,
                           label='outliers', color='k')
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='$class = $' + str(data.label), color=colors[i])


        elif type(data_check) is Coord3D:
            i = 0
            if is_list:
                for d in data:
                    ax.scatter(d.x, d.y, d.z, s=50, alpha=0.7,
                               label='uniform samples', color=palette[i])
                    i += 1
            else:
                ax.scatter(data.x, data.y, data.z, s=50, alpha=0.7,
                           label='uniform samples', color=palette[i])


        else:
            print('Error: data type not recognised')

        ax.set_axis_off()
        ax.legend(bbox_to_anchor=[0.65, 0.75])
        plt.savefig('toy_outlier11.jpg', dpi=250)

    def sample(self, n_samples, radius=1, distribution="uniform", mu=None, kappa=None, label=None):
        """
        Sample points on a spherical surface.
        """

        # uniform
        if distribution == self.distribution[0]:
            u = np.random.uniform(0, 1, n_samples)
            v = np.random.uniform(0, 1, n_samples)

            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)

            # convert to cartesian
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi)
            self.samples = Coord3D(x, y, z)

        # vMF
        elif distribution == self.distribution[1]:
            # try:
            s = sample_vMF(mu, kappa, n_samples)
            # except:
            #     print('Error: mu and kappa must be defined when sampling from vMF')
            #     return
            self.samples = vMFSample(s, kappa, label)

        else:
            print('Error: sampling distribution not recognised (try \'uniform\' or \'vMF\')')
        # breakpoint()
        return self.samples


class vMFSample():
    """
    Store 3D coordinates from the sample_vMF function
    and kappa value.
    """

    def __init__(self, sample, kappa, label):
        """
        Store 3D coordinates from the sample_vMF function
        and kappa value.
        """
        tp = np.transpose(sample)
        self.x = tp[0]
        self.y = tp[1]
        self.z = tp[2]
        self.label = label

        self.kappa = kappa
        self.sample = sample


from torch import nn
import torch.nn.functional as F
import torch


class Logistic(torch.nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)  # One in and one out
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred


class LogisticNoPara(torch.nn.Module):
    def __init__(self):
        super(LogisticNoPara, self).__init__()
        # self.linear = torch.nn.Linear(1, 1, bias= False) # One in and one out
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid((x))
        return y_pred


class MLP(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self, input_dim, class_num):  # 初始化网络结构
        super(MLP, self).__init__()  # 多继承需用到super函数
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 5)
        self.fc3 = nn.Linear(5, class_num)

    def forward(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

    def forward_inter(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        # x = self.fc3(x)  # output(10)
        return x

class MLP_add(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self, input_dim, class_num):  # 初始化网络结构
        super(MLP_add, self).__init__()  # 多继承需用到super函数
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 16)
        self.reg = nn.Linear(16, 2)
        self.fc3 = nn.Linear(16, class_num)

    def forward(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        reg = self.reg(x)
        x = self.fc3(x)  # output(10)
        return x, reg

    def forward_inter(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        # x = self.fc3(x)  # output(10)
        return x


class MLP_5(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self, input_dim, class_num):  # 初始化网络结构
        super(MLP_5, self).__init__()  # 多继承需用到super函数
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 500)
        self.fc4 = nn.Linear(500, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, class_num)

    def forward(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


# from torchvision import transforms, utils

class Linear(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self, input_dim, class_num):  # 初始化网络结构
        super(Linear, self).__init__()  # 多继承需用到super函数
        self.fc1 = nn.Linear(input_dim * 2, class_num)

    def forward(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = self.fc1(x)  # output(10)
        return x


class Linear_2d(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self, input_dim, class_num):  # 初始化网络结构
        super(Linear_2d, self).__init__()  # 多继承需用到super函数
        self.fc1 = nn.Linear(5, class_num)

    def forward(self, x):  # 正向传播过程
        #         x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = self.fc1(x)  # output(10)
        return x