import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class MLPRegressor:
    def __init__(self, x, y, x_test, y_test):
        """
        初始化MLP回归器

        参数:
        x: 训练特征
        y: 训练目标
        x_test: 测试特征
        y_test: 测试目标
        """
        # 数据标准化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.x_scaled = self.scaler_x.fit_transform(x.reshape(-1, 1))
        self.y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))

        self.x_test_scaled = self.scaler_x.transform(x_test.reshape(-1, 1))
        self.y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1))

        # 转换为PyTorch张量
        self.X = torch.FloatTensor(self.x_scaled)
        self.y = torch.FloatTensor(self.y_scaled)
        self.X_test = torch.FloatTensor(self.x_test_scaled)
        self.y_test = torch.FloatTensor(self.y_test_scaled)

        # 原始数据保存
        self.x_original = x
        self.y_original = y
        self.x_test_original = x_test
        self.y_test_original = y_test

    class MLPModel(nn.Module):
        def __init__(self, input_size, hidden_layers):
            """
            构建多层感知器模型

            参数:
            input_size: 输入特征维度
            hidden_layers: 隐藏层神经元配置
            """
            super().__init__()
            layers = []
            prev_size = input_size

            # 动态构建网络层
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size

                # 输出层
            layers.append(nn.Linear(prev_size, 1))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def train_model(self, hidden_layers=(10, 20, 10),
                    learning_rate=0.01,
                    epochs=5000,
                    verbose=100):
        """
        训练MLP模型

        参数:
        hidden_layers: 隐藏层神经元配置
        learning_rate: 学习率
        epochs: 训练轮数
        verbose: 输出频率
        """
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 创建模型
        model = self.MLPModel(input_size=1, hidden_layers=hidden_layers)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练过程
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # 前向传播
            y_pred = model(self.X)

            # 计算损失
            loss = criterion(y_pred, self.y)
            train_losses.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 测试集损失
            with torch.no_grad():
                y_test_pred = model(self.X_test)
                test_loss = criterion(y_test_pred, self.y_test)
                test_losses.append(test_loss.item())

                # 周期性输出
            if epoch % verbose == 0:
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {loss.item():.4f}, '
                      f'Test Loss: {test_loss.item():.4f}')

                # 模型预测
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(self.X_test)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.numpy())

            # 计算指标
        mse = mean_squared_error(self.y_test_original, y_pred)
        r2 = r2_score(self.y_test_original, y_pred)

        # 可视化
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 拟合结果
        plt.subplot(1, 2, 2)
        plt.scatter(self.x_original, self.y_original,
                    color='blue', label='Training points')
        plt.scatter(self.x_test_original, self.y_test_original,
                    color='red', label='Test points')

        # 生成精细预测线
        x_line = np.linspace(self.x_original.min(),
                             self.x_original.max(), 200).reshape(-1, 1)
        x_line_scaled = self.scaler_x.transform(x_line)
        x_line_tensor = torch.FloatTensor(x_line_scaled)

        with torch.no_grad():
            y_line_scaled = model(x_line_tensor).numpy()
        y_line = self.scaler_y.inverse_transform(y_line_scaled)

        plt.plot(x_line, y_line, color='green', label='MLP Prediction')
        plt.title(f'MLP Regression\n'
                  f'MSE: {mse:.4f}, R²: {r2:.4f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return model, mse, r2

    def grid_search_mlp(self):
        """
        网格搜索最佳MLP配置
        """
        # 要尝试的网络配置
        hidden_layer_configs = [

            (20, 10),
            (20, 20, 10),
            (30, 20, 10),
        ]

        learning_rates = [0.01, 0.001]

        best_mse = float('inf')
        best_model = None
        best_config = None

        for hidden_layers in hidden_layer_configs:
            for lr in learning_rates:
                print(f"\n分析配置：")
                print(f"隐藏层: {hidden_layers}")
                print(f"学习率: {lr}")

                try:
                    model, mse, r2 = self.train_model(
                        hidden_layers=hidden_layers,
                        learning_rate=lr,
                        epochs=3000,
                        verbose=500
                    )

                    # 更新最佳模型
                    if mse < best_mse:
                        best_mse = mse
                        best_model = model
                        best_config = {
                            'hidden_layers': hidden_layers,
                            'learning_rate': lr,
                            'mse': mse,
                            'r2': r2
                        }

                except Exception as e:
                    print(f"配置分析失败: {e}")

        print("\n最佳模型配置：")
        print(best_config)
        return best_model, best_config

    # 读取数据


data_training = pd.read_csv('Data4Regression - Training Data.csv')
data_test = pd.read_csv('Data4Regression - Test Data.csv')

# 提取特征和目标
x = data_training['x'].values
y_complex = data_training['y_complex'].values

x_new = data_test['x_new'].values
y_new_complex = data_test['y_new_complex'].values

# 创建MLP回归器
mlp_regressor = MLPRegressor(x, y_complex, x_new, y_new_complex)

# 执行网格搜索
best_model, best_config = mlp_regressor.grid_search_mlp()