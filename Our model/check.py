import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
import os
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
from scipy.interpolate import interp1d

set_filename = 'PEMS08'
file_path = f'../ASTGNN-main/data/{set_filename}/{set_filename}_r1_d0_w0_flag0.npz'  # 原数据的测试集 flag后一定要是0
file_path12 = f'../ASTGNN-main/experiments/{set_filename}/融合结果.npz'  # 融合结果的地址
save_filename = f'../ASTGNN-main/experiments/{set_filename}/误差.csv'  # 最终误差
flag_save = True  # 是否保存结果

data_text = np.load(file_path)
data12 = np.load(file_path12)

data_text = data_text['test_target']
# data12_pre = data12['prediction']
data12_pre = data12['data']

average_third_dim = data12_pre  # 结果
data_target_tensor = data_text  # 目标值
excel_list = []
prediction_length = average_third_dim.shape[2]

for i in range(prediction_length):
    assert data_target_tensor.shape[0] == average_third_dim.shape[0]
    print('predict %s points' % (i + 1))
    mae = mean_absolute_error(data_target_tensor[:, :, i], average_third_dim[:, :, i, 0])
    rmse = mean_squared_error(data_target_tensor[:, :, i], average_third_dim[:, :, i, 0]) ** 0.5
    mape = masked_mape_np(data_target_tensor[:, :, i], average_third_dim[:, :, i, 0], 0)
    print('MAE: %.2f' % (mae))
    print('RMSE: %.2f' % (rmse))
    print('MAPE: %.2f' % (mape))
    print()
    excel_list.extend([mae, rmse, mape])

    # print overall results
mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), average_third_dim.reshape(-1, 1))
rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), average_third_dim.reshape(-1, 1)) ** 0.5
mape = masked_mape_np(data_target_tensor.reshape(-1, 1), average_third_dim.reshape(-1, 1), 0)
print('all MAE: %.2f' % (mae))
print('all RMSE: %.2f' % (rmse))
print('all MAPE: %.2f' % (mape))
excel_list.extend([mae, rmse, mape])
print(excel_list)

if flag_save:
    import pandas as pd

    excel_list_rounded = [round(value, 4) for value in excel_list]  # 四舍五入

    # 将列表转换为DataFrame
    df = pd.DataFrame({
        'MAE': excel_list_rounded[::3],
        'RMSE': excel_list_rounded[1::3],
        'MAPE': excel_list_rounded[2::3]
    })

    # 将DataFrame保存为CSV文件
    df.to_csv(save_filename, index=False)

# for i in range(0, 12):  # (170, 3567)
#     # 创建一个图形和三个子图
#     fig, axs = plt.subplots(5, 1, figsize=(10, 15))  # 3行1列，可以调整figsize
#
#     # 绘制第三个数组
#     axs[0].plot(results_data[i], color='r')  # 蓝色线条
#     axs[0].set_title('Target value')
#     axs[0].set_xlabel('Index')
#     axs[0].set_ylabel('Value')
#
#     # 绘制第一个数组
#     axs[2].plot(results12[i], color='b')  # 红色线条
#     axs[2].set_title('5-minute interval forecast results')
#     axs[2].set_xlabel('Index')
#     axs[2].set_ylabel('Value')
#
#     # 绘制第二个数组
#     axs[3].plot(results6[i], color='g')  # 绿色线条
#     axs[3].set_title('10-minute interval forecast results')
#     axs[3].set_xlabel('Index')
#     axs[3].set_ylabel('Value')
#
#     # 绘制第三个数组
#     axs[4].plot(results3[i], color='y')  # 蓝色线条
#     axs[4].set_title('15-minute interval forecast results')
#     axs[4].set_xlabel('Index')
#     axs[4].set_ylabel('Value')
#
#     # 绘制第三个数组
#     axs[1].plot(average_array[i], color='y')  # 蓝色线条
#     axs[1].set_title('average_data')
#     axs[1].set_xlabel('Index')
#     axs[1].set_ylabel('Value')
#
#     # 自动调整子图间距
#     plt.tight_layout()
#
#     # 显示图形
#     plt.show()
