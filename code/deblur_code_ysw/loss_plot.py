import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
df1 = pd.read_csv('./loss_sum_0.8_pre_loglikehood/run-.-tag-Epoch Train KL vh_gauss.csv')  # csv文件所在路径
step1 = df1['Step'].values.tolist()
loss1 = df1['Value'].values.tolist()

df2 = pd.read_csv('./loss_sum_0.8_pre_loglikehood/run-.-tag-Epoch Valid KL vh_gauss.csv')
step2 = df2['Step'].values.tolist()
loss2 = df2['Value'].values.tolist()

plt.plot(step1, loss1, label='Train KL vh Loss')
plt.plot(step2, loss2, label='Valid KL vh Loss')
plt.legend(fontsize=16)  # 图注的大小
plt.savefig("./loss_sum_0.8_pre_loglikehood/loss_KL_vh.png",dpi=300)
plt.show()




