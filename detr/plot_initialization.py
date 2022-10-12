import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from matplotlib import rcParams
rcParams['font.family']='sans-serif'

# plt.grid(linestyle = '--', linewidth = 0.1)
fixed_x = [1,5,10,20,40]
init_x = ['const-1', 'const-5', 'const-10', 'const-20', 'const-40', 'Normal', 'Uniform']
fixed = [69.98,74.53, 74.54, 71.56, 69.36]#[80.88, 81.76,83.11, 83.29,82.76,81.84,80.43]
fixed = [74.93, 77.25, 77.67, 76.21, 76.54]
init = [74.07, 73.21, 76.10, 71.00, 70.09, 71.03, 71.31]#[71.90, 73.47,74.04,74.34,73.03,71.03,70.10]

sns.set(font_scale = 1.5)
sns.set_theme(style="ticks")

figure, axes = plt.subplots(1, 4, sharex=False, figsize=(20,4.5))


# sns.set(rc={'text.usetex' : True})

axes[0].set_title('(a) Different fixed kappa values')
data_preproc = pd.DataFrame({
    'Fixed kappa': fixed_x,
    'AUROC': fixed})
sub1 = sns.barplot(data=data_preproc,x='Fixed kappa',y='AUROC', ax=axes[0], color='#d58bc5')#palette=sns.color_palette("rocket_r",5))

sub1.set(ylim=(74,78))
# axes[0].set_ylabel("")
widthbars = [1,1,1,1,1]
for bar, newwidth in zip(axes[0].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub1.bar_label(sub1.containers[0], size = 11)


# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
# 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
# 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
# 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
# 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
# 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1',
# 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
# 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
# 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
# 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r',
# 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
# 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
# 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r',
# 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',
# 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic',
# 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
# 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
# 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

# axes[1].set_title(r'(b) Different kappa initializations')
# data_preproc = pd.DataFrame({
#     'Initialization': init_x,
#     'AUROC': init})
# sub2 = sns.barplot(data=data_preproc,x='Initialization',y='AUROC', ax=axes[1], palette=sns.color_palette('dark:salmon_r',7))
# # sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
# sub2.set(ylim=(69,77))
# axes[1].set_ylabel("")
# widthbars = [1,1,1,1,1,1,1]
# for bar, newwidth in zip(axes[1].patches, widthbars):
#     x = bar.get_x()
#     width = bar.get_width()
#     print(x)
#     centre = x #+ width/2.
#     bar.set_x(centre)
#     bar.set_width(newwidth)
# sub2.tick_params(axis='x', which='major', labelsize=8)
# sub2.bar_label(sub2.containers[0], size = 10)


loss_weight = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
weight_auroc = [71.56, 71.57, 73.42, 76.10, 76.06, 75.64]
weight_auroc = [76.01, 76.96, 77.28, 78.23, 78.21, 75.91]
axes[1].set_title(r'(b) Different loss weight for $\mathcal{L}_{\mathrm{SIREN}}$')
data_preproc = pd.DataFrame({
    'Loss weight': loss_weight,
    'AUROC': weight_auroc})
sub3 = sns.barplot(data=data_preproc,x='Loss weight',y='AUROC', ax=axes[1], color='#61a865')#palette=sns.color_palette('YlOrBr',6))

sub3.set(ylim=(75,78.5))
axes[1].set_ylabel("")
widthbars = [1,1,1,1,1,1]
for bar, newwidth in zip(axes[1].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub3.bar_label(sub3.containers[0], size = 11)





loss_weight = [8, 16, 32, 64, 80]
weight_auroc = [71.45, 76.10, 76.04, 76.00, 75.12]
weight_auroc = [77.74, 78.23, 77.90, 76.86, 76.76]
axes[2].set_title('(c) Different Hypersphere dimension')
data_preproc = pd.DataFrame({
    'Hypersphere dimension': loss_weight,
    'AUROC': weight_auroc})

sub4 = sns.barplot(data=data_preproc,x='Hypersphere dimension',y='AUROC', ax=axes[2], color='#5273b2')# palette=sns.color_palette('crest',5))

sub4.set(ylim=(76,78.5))
axes[2].set_ylabel("")
widthbars = [1,1,1,1,1]
for bar, newwidth in zip(axes[2].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub4.bar_label(sub4.containers[0], size = 11)



loss_weight = [1, 5, 10, 20, 50, 100, 200]
weight_auroc = [77.48, 78.10,  78.23, 78.02, 77.93, 77.60, 77.38]
axes[3].set_title('(d) Different k in the KNN score')
data_preproc = pd.DataFrame({
    'k in KNN score': loss_weight,
    'AUROC': weight_auroc})
sub4 = sns.barplot(data=data_preproc,x='k in KNN score',y='AUROC', ax=axes[3], color='#8172b5')#palette=sns.color_palette('crest',7))

sub4.set(ylim=(77,78.4))
axes[3].set_ylabel("")
widthbars = [1,1,1,1,1, 1, 1]
for bar, newwidth in zip(axes[3].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub4.bar_label(sub4.containers[0], size = 11)




figure.tight_layout(w_pad=1)
figure.savefig('ablation1.pdf')