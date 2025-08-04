import numpy as np
import pandas as pd

# from model import HA


class HA:
    def __init__(self, window_size):
        self.window_size = window_size  # P: 用于历史平均的窗口大小

    def fit(self, data):
        # 在 HA 模型中，'fit'操作主要用于存储数据，以备后续计算平均值
        self.data = data

    def predict(self, steps):
        # 使用最近 window_size 数据的平均值进行预测
        if len(self.data) < self.window_size:
            avg = np.mean(self.data)  # 如果数据不足，使用所有数据的平均值
        else:
            avg = np.mean(self.data[-self.window_size:])  # 否则使用最后 window_size 数据的平均值
        return np.array([avg] * steps)  # 返回相同的平均值，模拟预测接下来 steps 个时间步的值



def ha_train(P, Q, Traffic):
    predictions_list = []
    actuals_list = []
    results = []
    metrics_summary = {'MAE': [], 'RMSE': [], 'MAPE': [], 'WAPE': []}

    for i in range(Traffic.shape[1]):
        print('遍历到第',i,'个传感器')
        sensor_data = Traffic.iloc[:, i].values

        model = HA(Q)
        predictions = []
        actuals = []

        # Rolling window prediction
        for j in range(0, len(sensor_data) - Q, P):
            train_end = j + Q
            test_end = train_end + P
            if test_end > len(sensor_data):
                break

            model.fit(sensor_data[j:train_end])
            prediction = model.predict(P)
            actual = sensor_data[train_end:test_end]

            predictions.extend(prediction)
            actuals.extend(actual)
        # Scale back to original scale
        original_predictions = np.array(predictions)   # 逆向常数值添加
        original_actuals = np.array(actuals)
        # Calculate metrics
        metrics = calculate_metrics(original_predictions, original_actuals)
        results.append({
            'sensor_id': i,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'WAPE': metrics['WAPE']
        })

        # Add to summary
        for key in metrics_summary:
            metrics_summary[key].append(metrics[key])

        # 将每个传感器的预测和实际结果追加到列表中
        predictions_list.append(pd.Series(original_predictions, name=f'Sensor_{i + 1}_Predictions'))
        actuals_list.append(pd.Series(original_actuals, name=f'Sensor_{i + 1}_Actuals'))

    # Calculate and print average metrics
    average_metrics = {key: np.mean(values) for key, values in metrics_summary.items()}
    print("预测时间为为：",Q*5)
    print("Average Metrics:", average_metrics)
    pd.DataFrame([average_metrics]).to_csv('./results/HA/average_metrics.csv', index=False)
    print("预测完成。")


def calculate_metrics(predictions, actuals):
    # 如果输入是numpy数组，确保它们是扁平化的
    if isinstance(predictions, np.ndarray):
        pred_values = predictions.flatten()
        actual_values = actuals.flatten()
    else:
        pred_values = predictions
        actual_values = actuals

    # 计算度量
    mae = np.mean(np.abs(pred_values - actual_values))
    rmse = np.sqrt(np.mean((pred_values - actual_values)**2))
    epsilon = 1.0  # 避免除以零
    # mape = np.mean(np.abs(pred_values - actual_values) / (actual_values + epsilon)) * 100
    # # 避免除以零的问题，使用修改后的MAPE公式
    mape = np.mean(
        np.abs(pred_values - actual_values) / (0.5 * (np.abs(pred_values) + np.abs(actual_values)) + epsilon)) * 100

    wape = np.sum(np.abs(pred_values - actual_values)) / (np.sum(actual_values) + epsilon) * 100

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape}
if __name__ == '__main__':
    # 加载数据./data/metr-la.csv
    Traffic = pd.read_csv('data/pems08tt.csv', header=0, parse_dates=[0], index_col=0, skiprows=[1])
    Traffic = Traffic.asfreq('5min')  # 假设数据每5分钟记录一次
    Traffic.fillna(method='ffill', inplace=True)
    Traffic.fillna(method='bfill', inplace=True)
    ha_train(2,2, Traffic)