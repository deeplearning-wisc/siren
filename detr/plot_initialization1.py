import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from matplotlib import rcParams
rcParams['font.family']='sans-serif'

# plt.grid(linestyle = '--', linewidth = 0.1)
fixed_x = [0.01, 0.05, 0.1, 0.3, 0.5]
init_x = ['const-1', 'const-5', 'const-10', 'const-20', 'const-40', 'Normal', 'Uniform']
fixed = [69.98,74.53, 74.54, 71.56, 69.36]#[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43]
fixed = [85.89, 96.11, 98.70, 83.51, 76.50]
init = [74.07, 73.21, 76.10, 71.00, 70.09, 71.03, 71.31]#[71.90, 73.47,74.04,74.34,73.03,71.03,70.10]



sns.set(font_scale = 1.3)
sns.set_theme(style="ticks")
figure, axes = plt.subplots(1, 3, sharex=False, figsize=(15,3.5))


# sns.set(rc={'text.usetex' : True})

axes[0].set_title(r'(a) Different weight $\alpha$ for $R_{open}$')
data_preproc = pd.DataFrame({
    'Regularization weight': fixed_x,
    'AUROC': fixed})
sub1 = sns.barplot(data=data_preproc,x='Regularization weight',y='AUROC', ax=axes[0], color='#a5b3d2')#palette=sns.color_palette("rocket_r",5))

sub1.set(ylim=(75, 100.5))
# axes[0].set_ylabel("")
widthbars = [0.8, 0.8, 0.8, 0.8, 0.8]
for bar, newwidth in zip(axes[0].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub1.bar_label(sub1.containers[0], size = 11)




loss_weight = [0.01, 0.1, 0.5, 1, 10]
weight_auroc = [71.56, 71.57, 73.42, 76.10, 76.06, 75.64]
weight_auroc = [98.34, 98.70, 98.21, 97.95, 88.82]
axes[1].set_title(r'(b) Different variance $\sigma^2$ for outlier synthesis')
data_preproc = pd.DataFrame({
    'Variance': loss_weight,
    'AUROC': weight_auroc})
sub3 = sns.barplot(data=data_preproc,x='Variance',y='AUROC', ax=axes[1], color='#58508f')#palette=sns.color_palette('YlOrBr',6))

sub3.set(ylim=(75, 100.5))
axes[1].set_ylabel("")
widthbars = [0.8,0.8,0.8,0.8]
for bar, newwidth in zip(axes[1].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub3.bar_label(sub3.containers[0], size = 11)





loss_weight = [100, 200, 300,400,500]
weight_auroc = [71.45, 76.10, 76.04, 76.00, 75.12]
weight_auroc = [98.34, 98.31, 98.36, 98.70, 98.00]
axes[2].set_title('(c) Different K in KNN distance')
data_preproc = pd.DataFrame({
    'K in KNN distance': loss_weight,
    'AUROC': weight_auroc})

sub4 = sns.barplot(data=data_preproc,x='K in KNN distance',y='AUROC', ax=axes[2], color='#483a81')# palette=sns.color_palette('crest',5))

sub4.set(ylim=(75, 100.5))
axes[2].set_ylabel("")
widthbars = [0.8, 0.8, 0.8, 0.8, 0.8]
for bar, newwidth in zip(axes[2].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub4.bar_label(sub4.containers[0], size = 11)








figure.tight_layout(w_pad=1)
figure.savefig('ablation11.pdf')