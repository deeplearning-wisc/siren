import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# fixed_x = [1,5,10,20,40]
init_x = ['1<<18', '1<<17', '1<<16',  '1<<15',  '1<<14',  '1<<13' ]
init_y = [20000, 20000, 38500, 39000, 38500, 38300 ]

sns.set(font_scale = 1.5)
sns.set_theme(style="ticks")

figure, axes = plt.subplots(1, 4, sharex=False, figsize=(24,5))




axes[0].set_title('(a) The effect of buffer size per port')
data_preproc = pd.DataFrame({
    'The number of pkts per port': init_x,
    'Pkts per second': init_y})
sub1 = sns.barplot(data=data_preproc,x='The number of pkts per port',y='Pkts per second',
                   ax=axes[0], palette=sns.color_palette('flare',6))

sub1.set(ylim=(19000,40000))
# axes[0].set_ylabel("")
widthbars = [1,1,1,1,1, 1]
for bar, newwidth in zip(axes[0].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub1.bar_label(sub1.containers[0], size = 11)


init_x = [8, 7, 6, 5, 4, 3,2, 1]
init_y = [25600, 27100, 25400, 24000, 22000, 17000, 13500, 8000]
axes[1].set_title(r'(b) The effect of the GPU burst size')
data_preproc = pd.DataFrame({
    'GPU burst x56 pkt': init_x,
    'Pkts per second': init_y})
sub2 = sns.barplot(data=data_preproc,x='GPU burst x56 pkt',y='Pkts per second',
                   ax=axes[1], palette=sns.color_palette('dark:salmon_r',8))
# sub4.set(xticks=[0, 5, 10, 15], yticks= [74,75])
sub2.set(ylim=(7000, 28100))
axes[1].set_ylabel("")
widthbars = [1,1,1,1,1,1,1,1]
for bar, newwidth in zip(axes[1].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub2.tick_params(axis='x', which='major', labelsize=8)
sub2.bar_label(sub2.containers[0], size = 10)


loss_weight = ['1<<18', '1<<17', '1<<16','1<<14']
weight_auroc = [21000, 21000, 20900, 20900]
axes[2].set_title(r'(c) The effect of the GPU ring depth')
data_preproc = pd.DataFrame({
    'GPU ring depth (#pkt)': loss_weight,
    'Pkts per second': weight_auroc})
sub3 = sns.barplot(data=data_preproc,x='GPU ring depth (#pkt)',
                   y='Pkts per second', ax=axes[2], palette=sns.color_palette('YlOrBr',4))

sub3.set(ylim=(20800, 21100))
axes[2].set_ylabel("")
widthbars = [1,1,1,1]
for bar, newwidth in zip(axes[2].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub3.bar_label(sub3.containers[0], size = 11)




loss_weight = [15, 12, 11, 10,9,8,7,6,5,4,3,2,1]
weight_auroc = [39000, 38500, 38700, 21000, 21200, 21200, 21200,21200,
                20400, 20400, 20400, 20400, 25000]
axes[3].set_title('(d) The effect of the number of queue')
data_preproc = pd.DataFrame({
    'The number of the queue': loss_weight,
    'Pkts per second': weight_auroc})
sub4 = sns.barplot(data=data_preproc,x='The number of the queue',
                   y='Pkts per second', ax=axes[3], palette=sns.color_palette('crest',13))

sub4.set(ylim=(20000, 45000))
axes[3].set_ylabel("")
widthbars = [1,1,1,1,1, 1,1,1,1,1,1,1,1]
for bar, newwidth in zip(axes[3].patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    print(x)
    centre = x #+ width/2.
    bar.set_x(centre)
    bar.set_width(newwidth)
sub4.bar_label(sub4.containers[0], size = 6)






figure.tight_layout(w_pad=1)
figure.savefig('740.pdf')