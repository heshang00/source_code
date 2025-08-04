import os
import numpy as np
import argparse
import configparser


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):  #用于搜索数据的起始索引和结束索引
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data  所有历史数据的长度
    num_of_depend: int,  预测目标的第一个索引
    label_start_idx: int, the first index of predicting target 每个样本将要预测的点数的第一个索引
    num_for_predict: int, the number of points will be predicted for each sample 每个样本将要预测的点数
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data 每小时的点数
    Returns
    ----------
    list[(start_idx, end_idx)] 起始索引，结束索引
    '''

    if points_per_hour < 0:  # 报错
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:  # 查看序列长度
        return None

    x_idx = [] # 循环创建索引列表x_idx
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
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, 预测值开始的那个点的索引 the first index of predicting target,
    num_for_predict: int,  每个样本将要预测的点数
                     the number of points will be predicted for each sample
    points_per_hour: int, 每小时的点数, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None  # 初始化周 天 小时的样本

    # 如果预测的其实索引加上预测点数超过了数据序列的长度
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # 处理周的数量
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    # 如果天的数量大于0，则使用search函数获取天样本的索引，并连接
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    # 如果小时的数量大于0，则使用search函数获取小时样本的索引，并连接
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        # print(hour_indices, "===")  # [(8895, 8907)] ===
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j: step]
                                      for i, j in hour_indices], axis=0)
        # print(hour_sample.shape)  # (12, 170, 1)

    # 获取目标值，即预测值
    # target = data_sequence[label_start_idx: label_start_idx + num_for_predict]  # 原
    # print(target.shape, "==========")

    # print(data_sequence.shape, "===")
    # step = 2
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict: step] # 隔步长取一个
    # print(target.shape, "==========")  # (3, 170, 3)

    # 初始化目标序列
    # target = []
    #
    # # 计算目标序列
    # for i in range(label_start_idx, len(data_sequence) - 1, step):
    #     # 计算步长中的数的平均值
    #     avg = np.mean(data_sequence[i:i + 2])
    #     # 将计算得到的平均值添加到目标序列
    #     target.append(avg)
    #
    # # 将列表转换为NumPy数组
    # target = np.array(target)
    # # 打印结果和数据类型
    # print("目标序列（平均值）:", target)
    # print("数据类型:", type(target))

    # print(target.shape, "==========")
    # print(str(type(target)))
    return week_sample, day_sample, hour_sample, target


def MinMaxnormalization(train, val, test):  # 用于最小最大归一化
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename,
                                              num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict, step, flag,
                                              points_per_hour=12, save=False ):
    # 用于读取和生成编码器——解码器数据集
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, 图信号矩阵文件的路径 path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']  # 加载数据信号矩阵
    # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []  # 初始化样本列表

    for idx in range(data_seq.shape[0]):  # 循环每个样本的索引
        # 初始化样本列表，并循环捕获每个样本的索引
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict, step,
                                    points_per_hour)

        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue
        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            # print(hour_sample.shape, "----")  # (1, 170, 1, 12) ----
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    # print(all_samples[0][0][0][0])
    # 获取分割点
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]



    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)  # 特征数据

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]  # 目标值

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]  # 时间戳

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

    # print("train_x[0][0][0]:", train_x[0][0][0])
    # print("train_target[0][0]:", train_target[0][0])

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '_flag' + str(flag))
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
    return all_data


# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS08.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
step = int(data_config['step'])
flag = int(data_config['flag'])
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)
data['data'].shape

all_data = read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, step, flag, points_per_hour=points_per_hour, save=True)
