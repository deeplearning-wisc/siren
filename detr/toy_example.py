import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
from matplotlib import cm
import matplotlib.pyplot as plt
import sklearn
from sklearn import covariance
import faiss
from sphere import Sphere, MLP
from vMF import density
import argparse
import os
from utils import maha_train_estimation
# generate the training data.

parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--name', default=1., type=str)
args = parser.parse_args()
num_samples = 1000
kappa = [100]
if not os.path.exists('./toy_train.npz'):
    # generate train data.
    samples = []
    for k in kappa:
        samples.append(Sphere().sample(num_samples, distribution = 'vMF',
                                       mu = np.asarray([0, 0, 1]), kappa = k, label=0))
        samples.append(Sphere().sample(num_samples, distribution='vMF',
                                       mu=np.asarray([np.sqrt(3) / 2, 0, -1 / 2]), kappa=k, label=1))
        samples.append(Sphere().sample(num_samples, distribution='vMF',
                                       mu=np.asarray([-np.sqrt(3) / 2, 0, -1 / 2]), kappa=k, label=2))
    grid_points = Sphere().plot(data = samples)
    # breakpoint()
    for index in range(3):
        if index == 0:
            data_train = np.concatenate((samples[index].x.reshape(-1, 1), samples[index].y.reshape(-1,1)), 1)
            data_train = np.concatenate((data_train, samples[index].z.reshape(-1, 1)), 1)
            labels_train = np.ones(num_samples) * index
        else:
            temp_train = np.concatenate((samples[index].x.reshape(-1, 1), samples[index].y.reshape(-1, 1)), 1)
            temp_train = np.concatenate((temp_train, samples[index].z.reshape(-1, 1)), 1)
            data_train = np.concatenate((data_train, temp_train), 0)
            labels_train = np.concatenate((labels_train, np.ones(num_samples) * index), -1)
    np.savez('./toy_train.npz', data=data_train,label=labels_train,
             grid_points1=grid_points, samples=samples)
    # generate test data.
    num_samples = 100
    # sample from vMF for a range of kappa
    samples = []
    for k in kappa:
        samples.append(Sphere().sample(num_samples, distribution='vMF',
                                       mu=np.asarray([0, 0, 1]), kappa=k, label=0))
        samples.append(Sphere().sample(num_samples, distribution='vMF',
                                       mu=np.asarray([np.sqrt(3) / 2, 0, -1 / 2]), kappa=k, label=1))
        samples.append(Sphere().sample(num_samples, distribution='vMF',
                                       mu=np.asarray([-np.sqrt(3) / 2, 0, -1 / 2]), kappa=k, label=2))

    for index in range(3):
        if index == 0:
            data_test = np.concatenate((samples[index].x.reshape(-1, 1), samples[index].y.reshape(-1, 1)), 1)
            data_test = np.concatenate((data_test, samples[index].z.reshape(-1, 1)), 1)
            labels_test = np.ones(num_samples) * index
        else:
            temp_test = np.concatenate((samples[index].x.reshape(-1, 1), samples[index].y.reshape(-1, 1)), 1)
            temp_test = np.concatenate((temp_test, samples[index].z.reshape(-1, 1)), 1)
            data_test = np.concatenate((data_test, temp_test), 0)
            labels_test = np.concatenate((labels_test, np.ones(num_samples) * index), -1)
    np.savez('./toy_test.npz', data=data_test, label=labels_test)
else:
    data_train = np.load('./toy_train.npz',allow_pickle=True)['data']
    labels_train = np.load('./toy_train.npz',allow_pickle=True)['label']
    # breakpoint()
    samples = np.load('./toy_train.npz',allow_pickle=True)['samples']
    grid_points = np.load('./toy_train.npz',allow_pickle=True)['grid_points1']
    data_test = np.load('./toy_test.npz',allow_pickle=True)['data']
    labels_test = np.load('./toy_test.npz',allow_pickle=True)['label']

model_vanilla = MLP(3, 3).cuda()
optimizer = torch.optim.Adam(model_vanilla.parameters(),lr=0.001,weight_decay=0.0005)
epochs = 400
criter = torch.nn.CrossEntropyLoss()
data_train = torch.from_numpy(data_train).cuda().float()
labels_test = torch.from_numpy(labels_test).cuda().long()
data_test = torch.from_numpy(data_test).cuda().float()
labels_train = torch.from_numpy(labels_train).cuda().long()

for epoch in range(epochs):
    model_vanilla.train()
    optimizer.zero_grad()
    out = model_vanilla(data_train)
    loss = criter(out, labels_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        model_vanilla.eval()
        acc = (model_vanilla(data_test).max(-1)[1] == labels_test).sum().item() / len(labels_test)
        print('loss: ', loss, 'acc: ', acc)

# plot maha distance distribution.
maha_train = []
train_embed = model_vanilla.forward_inter(data_train).cpu().data.numpy()
labels_train = labels_train.cpu().data.numpy()
mean_list, covariance_list = maha_train_estimation(train_embed, labels_train, length=5)

# breakpoint()
grid_points = grid_points.item()
# calculate the maha distance for the grid points.
gird_points_input = np.concatenate((grid_points.x.reshape(-1, 1), grid_points.y.reshape(-1, 1)), 1)
gird_points_input =  np.concatenate((gird_points_input, grid_points.z.reshape(-1,1)), 1)
grid_points_inter = model_vanilla.forward_inter(torch.from_numpy(gird_points_input).cuda().float())
# breakpoint()
gaussian_score = 0
for i in range(3):
    batch_sample_mean = mean_list[-1][i]
    zero_f = grid_points_inter.cpu() - batch_sample_mean
    term_gau = -0.5*torch.mm(torch.mm(zero_f.float(), covariance_list[-1].float()), zero_f.float().t()).diag()
    if i == 0:
        gaussian_score = term_gau.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

id_score, _ = torch.max(gaussian_score, dim=1)
id_score = id_score.data.numpy()
id_score = id_score.reshape(200, 200)


# cmap = cm.coolwarm
# rescaled = (id_score - id_score.min()) /\
#            (id_score.max() - id_score.min())
# colors = cmap(rescaled)
#
# Sphere().plot_add(data = [samples[0],samples[1],samples[2]],
#                                 fcolors=colors)

# breakpoint()
# code block for calculating the vmf score.
k = kappa[0]
id_score1 = density(np.asarray([0, 0, 1]), k, gird_points_input)
id_score2 = density(np.asarray([np.sqrt(3) / 2, 0., -1 / 2]), k, gird_points_input)
id_score3 = density(np.asarray([-np.sqrt(3) / 2, 0., -1 / 2]), k, gird_points_input)
id_score = torch.stack([id_score1, id_score2, id_score3], 0)
id_score = id_score.max(0)[0]
id_score = id_score.data.numpy()
id_score = id_score.reshape(200, 200)

cmap = cm.coolwarm
rescaled = (id_score - id_score.min()) /\
           (id_score.max() - id_score.min())
colors = cmap(rescaled)

Sphere().plot_add(data = [samples[0],samples[1],samples[2]],
                                fcolors=colors)



# mean_list, covariance_list = maha_train_estimation(data_train.cpu().data.numpy(), labels_train, length=3)
# gaussian_score = 0
# for i in range(3):
#     batch_sample_mean = mean_list[-1][i]
#     zero_f = torch.from_numpy(gird_points_input) - batch_sample_mean
#     term_gau = -0.5*torch.mm(torch.mm(zero_f.float(), covariance_list[-1].float()), zero_f.float().t()).diag()
#     if i == 0:
#         gaussian_score = term_gau.view(-1,1)
#     else:
#         gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
#
# id_score, _ = torch.max(gaussian_score, dim=1)
# id_score = id_score.data.numpy()
# id_score = id_score.reshape(200, 200)
#
# cmap = cm.coolwarm
# rescaled = (id_score - id_score.min()) /\
#            (id_score.max() - id_score.min())
# colors = cmap(rescaled)
#
# Sphere().plot_add(data = [samples[0],samples[1],samples[2]],
#                                 fcolors=colors)
# breakpoint()

# id_score =

# Plot the outliers.
samples_outlier1 = []
samples_outlier2 = []
samples_outlier3 = []
k = kappa[0]
keep_numbers = 50
samples_outlier1.append(Sphere().sample(10000, distribution = 'vMF',
                                       mu = np.asarray([0., 0., 1.0]), kappa = k, label=0))
temp = np.concatenate((samples_outlier1[0].x.reshape(-1, 1), samples_outlier1[0].y.reshape(-1, 1)), 1)
samples_outlier1 = np.concatenate((temp, samples_outlier1[0].z.reshape(-1, 1)), 1)
density1 = density(np.asarray([0, 0, 1]), k, samples_outlier1)
plot_outlier1 = torch.from_numpy(samples_outlier1[(-density1).topk(keep_numbers)[1]]).cuda().float()
plot_outlier1 = samples_outlier1[(-density1).topk(keep_numbers)[1]]
grid_points = Sphere().plot_outliers(data = [samples[0], samples[1], samples[2]], outliers=plot_outlier1)
#
# plot_outlier1 = plot_outlier1.view(1,-1)
breakpoint()



samples_outlier2.append(Sphere().sample(100000, distribution='vMF',
                               mu=np.asarray([np.sqrt(3) / 2, 0., -1 / 2]), kappa=k, label=1))
temp = np.concatenate((samples_outlier2[0].x.reshape(-1, 1), samples_outlier2[0].y.reshape(-1, 1)), 1)
samples_outlier2 = np.concatenate((temp, samples_outlier2[0].z.reshape(-1, 1)), 1)
density2 = density(np.asarray([np.sqrt(3) / 2, 0., -1 / 2]), k, samples_outlier2)
plot_outlier2 = torch.from_numpy(samples_outlier2[(-density2).topk(keep_numbers)[1]]).cuda().float()

plot_outlier2 = plot_outlier2.view(1,-1)
samples_outlier3.append(Sphere().sample(100000, distribution='vMF',
                               mu=np.asarray([-np.sqrt(3) / 2, 0., -1 / 2]), kappa=k, label=2))
temp = np.concatenate((samples_outlier3[0].x.reshape(-1, 1), samples_outlier3[0].y.reshape(-1, 1)), 1)
samples_outlier3 = np.concatenate((temp, samples_outlier3[0].z.reshape(-1, 1)), 1)
density3 = density(np.asarray([-np.sqrt(3) / 2, 0., -1 / 2]), k, samples_outlier3)
plot_outlier3 = torch.from_numpy(samples_outlier3[(-density3).topk(keep_numbers)[1]]).cuda().float()
plot_outlier3 = plot_outlier3.view(1,-1)


from sphere import MLP_add
model_new = MLP_add(3,3).cuda()
optimizer1 = torch.optim.Adam(model_new.parameters(),lr=0.001,weight_decay=0.0005)
epochs = 400
criter1 = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1])).cuda().float())
for epoch in range(epochs):
    model_new.train()
    optimizer1.zero_grad()
    out, reg_id = model_new(data_train)
    _, reg_ood = model_new(torch.cat((torch.cat((plot_outlier1, plot_outlier2), 0), plot_outlier3), 0))
    # breakpoint()
    loss = criter(out, torch.from_numpy(labels_train).cuda().long())
    labels_reg = torch.cat((torch.ones(len(labels_train))[:10], torch.zeros(len(reg_ood))), 0).cuda().long()
    loss1 = criter1(torch.cat((reg_id[:10], reg_ood), 0), labels_reg)
    loss += loss1
    loss.backward()
    optimizer1.step()
    if epoch % 50 == 0:
        model_new.eval()
        acc = (model_new(data_test)[0].max(-1)[1] == labels_test).sum().item() / len(labels_test)
        print('loss: ', loss, 'acc: ', acc)




# plot maha distance distribution.
maha_train = []
train_embed = model_new.forward_inter(data_train).cpu().data.numpy()
mean_list, covariance_list = maha_train_estimation(train_embed, labels_train)

grid_points_inter = model_new.forward_inter(torch.from_numpy(gird_points_input).cuda().float())
gaussian_score = 0
for i in range(3):
    batch_sample_mean = mean_list[-1][i]
    zero_f = grid_points_inter.cpu() - batch_sample_mean
    term_gau = -0.5*torch.mm(torch.mm(zero_f.float(), covariance_list[-1].float()), zero_f.float().t()).diag()
    if i == 0:
        gaussian_score = term_gau.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

id_score, _ = torch.max(gaussian_score, dim=1)
id_score = id_score.data.numpy()
id_score = id_score.reshape(200, 200)


# cmap = cm.coolwarm
# rescaled = (id_score - id_score.min()) /\
#            (id_score.max() - id_score.min())
# colors = cmap(rescaled)
#
# grid_points = Sphere().plot_add(data = [samples[0],samples[1],samples[2]],
#                                 fcolors=colors)


# plot binary prediction scores.

_, id_score = model_new(torch.from_numpy(gird_points_input).cuda().float())
id_score = id_score.softmax(1)[:,1].cpu().data.numpy()

id_score = id_score.reshape(200, 200)
cmap = cm.coolwarm
# rescaled = (id_score - id_score.min()) /\
#            (id_score.max() - id_score.min())
colors = cmap(id_score)

grid_points = Sphere().plot_add(data = [samples[0],samples[1],samples[2]],
                                fcolors=colors)
breakpoint()


# breakpoint()