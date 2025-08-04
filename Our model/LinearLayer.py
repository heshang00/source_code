import numpy as np

def search_data(sequence_length, num_of_depend, label_start_idx,
                    num_for_predict, units, points_per_hour):
    if points_per_hour < 0:  # 报错
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:  # 查看序列长度
        return None

    x_idx = []  # 循环创建索引列表x_idx
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:  # 如果索引列表的长度不等于依赖数量
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, step, points_per_hour=12):
    # 用于获取样本的索引
    week_sample, day_sample, hour_sample = None, None, None  # 初始化周 天 小时的样本

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j: step]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict: step]  # 隔步长取一个

    return week_sample, day_sample, hour_sample, target


def get_data(data, num_of_weeks=0, num_of_days=0,
             num_of_hours=1, num_for_predict=12, step=1,
             points_per_hour=12):
    data_seq = data
    all_samples = []  # 初始化样本列表

    for idx in range(data_seq.shape[0]):  # 循环每个样本的索引
        # 初始化样本列表，并循环捕获每个样本的索引
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict, step,
                                    points_per_hour)

        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0)
            hour_sample = hour_sample.transpose((0, 2, 3, 1))
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)

    return train_x

def sliding_window_2d(data, window_size):
    # 初始化一个空列表，用于存储每一行的窗口数据
    windows_list = []

    # 对每一行数据应用滑动窗口
    for row in data:
        # 使用滑动窗口的方式，窗口大小为 window_size，步长为 1
        windows = [row[i:i + window_size] for i in range(0, len(row) - window_size + 1, 1)]
        # 将每一行的窗口转换为NumPy数组，并添加到列表中
        windows_array = np.array(windows)
        windows_list.append(windows_array)

    # 将列表转换为NumPy数组，形状为(170, 3567, 12)
    windows_array_3d = np.array(windows_list)
    return windows_array_3d


file_path = 'data/PEMS08/PEMS08_r1_d0_w0_flag0.npz'
data_text = np.load(file_path)
data_text = data_text['test_target']
save_file_path_final = f'../experiments/PEMS08/融合结果.npz'
new_data = np.zeros((data_text.shape[0], data_text.shape[1], data_text.shape[2], 1))
file_path12 = 'experiments/PEMS08/MAE_ASTGNN_h1d0w0_layer4_head8_dm64_channel1_dir2_drop0.00_1.00e-03TcontextScaledSAtSE1TEflag327/output_epoch_54_test.npz'

for t in range(1, 13):
    temp_flag = t
    data_text = np.load(file_path)  # 初始数据
    data12 = np.load(file_path12)  # 预测结果

    data_text = data_text['test_target']  # （170，3578，1）
    data12_pre = data12['prediction']  # （3567，170，12，1）

    data12_pre = np.squeeze(data12_pre.transpose((1, 0, 2, 3)))  # （170，3567，12，1）

    # shape12 = (170, 3578)
    shape12 = (data12_pre.shape[0], data12_pre.shape[1]+11)

    # 创建空数组并用零初始化
    results12 = np.zeros(shape12)
    for k in range(data12_pre.shape[0]):  # 循环第一维度 0~169
        data_textt = data12_pre[k, :, :]   # 提取出第k个样本  （3567，12，1）
        for i in range(data_textt.shape[0] + temp_flag - 1):  # 遍历样本的第一维度 0 ~ 3567
            flag = 0
            temp = 0
            if i < data_textt.shape[0]:
                for j in range(0, temp_flag):
                    if i - j >= 0:  # 确保索引不会变成负数
                        temp += data_textt[i - j, j]
                        flag = flag + 1
            else:
                for j in range(i - data_textt.shape[0], temp_flag):
                    if i - j < data_textt.shape[0]:
                        temp += data_textt[i - j, j]
                        flag = flag + 1

            results12[k][i] = temp / flag

    results12 = sliding_window_2d(results12, 12)
    results12 = results12.transpose((1, 0, 2))

    results12 = np.expand_dims(results12, axis=3)

    print(f'data{t} done!')

    average_third_dim = results12  # 结果
    new_data[:, :, t - 1, :] = average_third_dim[:, :, t - 1, :]

print(new_data.shape)

np.savez(save_file_path_final, data=new_data)
print(f"Data saved to: {save_file_path_final}")
