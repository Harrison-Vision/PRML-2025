import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 负号显示设置

# 1. 读取数据
data = pd.read_csv('LSTM-Multivariate_pollution.csv')

# 2. 数据预处理
# 选择用于模型的特征，不包括 date
features = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data = data[features]

# 编码类别变量 wnd_dir
data['wnd_dir'] = pd.Categorical(data['wnd_dir']).codes

# 处理缺失值（如果有）
data = data.dropna()

# 计算特征的相关性矩阵
correlation_matrix = data.corr()

# 输出相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('特征相关性热图', fontsize=16)
plt.show()

# 选择与 PM2.5 相关性高的特征
threshold = 0.04  # 选择相关性阈值
correlation_with_target = correlation_matrix['pollution']
selected_features = correlation_with_target[abs(correlation_with_target) > threshold].index.tolist()
selected_features.remove('pollution')  # 移除目标特征本身

# 使用选择的特征生成新的数据集
data_selected = data[['pollution'] + selected_features]  # 保证包含目标列

# 归一化
scaler = MinMaxScaler()
scaled_data_selected = scaler.fit_transform(data_selected)

# 3. 创建序列数据
def create_sequences(data, seq_length=24):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 0]  # pollution作为预测目标
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24  # 以过去24小时预测
X, y = create_sequences(scaled_data_selected, seq_length)

# 分割训练集和测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. 构建LSTM模型
model = Sequential()
model.add(Input(shape=(seq_length, X.shape[2])))  # 使用 LSTM 需要的输入形状
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))  # 增加 Dropout 以降低过拟合风险
model.add(LSTM(50, return_sequences=True))  # 使用return_sequences=True，以便可以叠加更多的LSTM层
model.add(Dropout(0.3))  # 增加 Dropout
model.add(LSTM(25))  # 第三层 LSTM
model.add(Dropout(0.3))  # 增加 Dropout
model.add(Dense(1))  # 预测值应为单个数

# 修改学习率
learning_rate = 0.001  # 设定学习率，可以根据需要尝试不同的值
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 5. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 6. 预测
y_pred = model.predict(X_test)

# 逆归一化
# 获取用于逆变换的 scaler 的特征范围
pollution_min = data['pollution'].min()
pollution_max = data['pollution'].max()

# 逆归一化函数
def destandardize(scaled_values, min_value, max_value):
    return scaled_values * (max_value - min_value) + min_value

y_test_orig = destandardize(y_test, pollution_min, pollution_max)
y_pred_orig = destandardize(y_pred.flatten(), pollution_min, pollution_max)

# 7. 结果可视化

# (1) 预测结果与真实值可视化
plt.figure(figsize=(12, 6))
plt.plot(y_test_orig, label='真实值')
plt.plot(y_pred_orig, label='预测值')
plt.legend()
plt.xlabel('时间 (样本索引)')
plt.ylabel('PM2.5 浓度')
plt.title('LSTM PM2.5 预测结果')
plt.show()

# (2) 绘制训练损失和验证损失曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.title('训练和验证损失趋势')
plt.legend()
plt.show()

# (3) 绘制预测值与真实值的散点图
plt.figure(figsize=(12, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')  # 45° 线
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('散点图：预测值 vs 真实值')
plt.show()

# (4) 绘制预测误差的直方图
errors = y_test_orig - y_pred_orig
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=30, alpha=0.7, color='g')
plt.axvline(0, color='r', linestyle='--')  # 红线表示零误差
plt.xlabel('预测误差')
plt.ylabel('样本数')
plt.title('预测误差分布')
plt.show()

# (5) 计算并输出MAE
mae = np.mean(np.abs(errors))
print(f'平均绝对误差 (MAE): {mae:.2f}')