import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm 

# 读取数据
data = pd.read_csv('data/pems04tt.csv', index_col=0, parse_dates=True)
data = data.asfreq('5min')
window_size = 7 # 前一个小时的数据
def preprocess_data(data):
    """预处理数据，填充NaN值并去除无效值"""
    data = data.fillna(method='ffill').fillna(method='bfill')  # 前向和后向填充
    data = data.replace([np.inf, -np.inf], np.nan).dropna()  # 移除无穷大值和NaN值
    return data

data = preprocess_data(data)

def svr_forecast_single_column(data, col, prediction_steps, window_size):
    predictions = pd.Series(index=data.index)
    for t in range(window_size, len(data) - prediction_steps):
        train_data = data[col].iloc[t - window_size:t]
        train_labels = data[col].iloc[t:t + prediction_steps]
        if len(train_data) < window_size or len(train_labels) < prediction_steps:
            continue
        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        model = SVR()
        model.fit(train_data_scaled, train_labels)
        test_data_scaled = scaler.transform(data[col].iloc[t:t + window_size].values.reshape(-1, 1))
        forecast = model.predict(test_data_scaled)
        predictions.iloc[t + prediction_steps] = forecast[-1]
    return col, predictions

def svr_forecast(data, prediction_steps, window_size=12, n_jobs=12):
    results = Parallel(n_jobs=n_jobs)(delayed(svr_forecast_single_column)(data, col, prediction_steps, window_size) for col in tqdm(data.columns, desc="Processing columns"))
    predictions = pd.DataFrame({col: pred for col, pred in results})
    return predictions

if __name__ == '__main__':
    # 使用前一个小时（12个点）的数据来预测未来5分钟或10分钟的数据
    prediction_steps = 7  # 预测步长为5分钟

    # 预测数据
    predictions = svr_forecast(data, prediction_steps, window_size)

    # 移动真实值，使之与预测值对齐
    true_values = data.shift(-prediction_steps)

    # 只保留有效的部分
    valid_idx = slice(window_size, -prediction_steps)
    true_values = true_values.iloc[valid_idx]
    predictions = predictions.iloc[valid_idx]

    # 计算误差
    def calculate_errors(true_values, predictions):
        valid_idx = ~(true_values.isna().any(axis=1) | predictions.isna().any(axis=1))
        true_values = true_values[valid_idx]
        predictions = predictions[valid_idx]
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        non_zero_true = true_values.replace(0, np.nan)
        mape = np.mean(np.abs((non_zero_true - predictions) / non_zero_true).replace([np.inf, -np.inf], np.nan).dropna().values) * 100
        return mae, rmse, mape

    mae, rmse, mape = calculate_errors(true_values, predictions)

    print("预测时间为为：",prediction_steps*5)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
