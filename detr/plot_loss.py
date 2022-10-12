import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 2

# matplotlib.rcParams['mathtext.fontset'] = 'Arial'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial'

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
data  =open('/nobackup-slow/dataset/my_xfdu/detr_out/exps/pascal_center_project_dim_16_weight_0.5_t_0.1_vmf_start_0_reweighted_mc_relu/log.txt','r')
tweets = []
data = data.readlines()
# breakpoint()
for line in data:
    tweets.append(float(line[line.find('train_loss_vmf')+17: line.find(', "train_cardinality_error_unscaled":')]))

plt.figure(figsize=(10,8))
x = [i for i in range(len(tweets))]
plt.plot(x,tweets, label=r'$\mathcal{L}_{\mathrm{uncertainty}}$',color='#184E77',linewidth=3)
plt.xlabel("epochs", fontsize=25)
plt.ylabel("Uncertainty loss", fontsize=25)
plt.xticks(fontsize= 25)
plt.yticks(fontsize= 25)
plt.legend(fontsize=30, frameon=False)
plt.savefig('./mc-relu.jpg', dpi=500)
