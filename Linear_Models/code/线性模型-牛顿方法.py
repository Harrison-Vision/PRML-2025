import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def newton_method(x, y, max_iter=100, tol=1e-6):
    # 将x表示为矩阵形式
    X = x.reshape(len(x), -1)
    X = np.column_stack([np.ones(X.shape[0]), X])

    # 将y表示为矩阵形式
    Y = y.reshape(-1, 1)

    # 初始化参数
    theta = np.zeros((X.shape[1], 1))
    loss_history = []

    for iter in range(max_iter):
        # 计算预测值
        h = X @ theta

        # 计算损失
        loss = np.mean((h - Y) ** 2) / 2
        loss_history.append(loss)

        # 计算一阶导数（梯度）
        gradient = X.T @ (h - Y) / len(y)

        # 计算海森矩阵（二阶导数）
        hessian = X.T @ X / len(y)

        # 使用逆矩阵求解
        delta = np.linalg.inv(hessian) @ gradient
        theta = theta - delta

    # 解包theta
    theta0, theta1 = theta.flatten()

    return theta0, theta1, loss_history


# 读取数据
data_training = pd.read_csv('Data4Regression - Training Data.csv')
data_test = pd.read_csv('Data4Regression - Test Data.csv')

# 提取特征和目标
x = data_training['x'].values
y_complex = data_training['y_complex'].values

x_new = data_test['x_new'].values
y_new_complex = data_test['y_new_complex'].values

# 应用牛顿法
theta0, theta1, loss_history = newton_method(x, y_complex)
print(f"训练结果: theta0 = {theta0}, theta1 = {theta1}")

# Calculate the loss on the test set
Loss_test = (1 / (2 * len(y_new_complex))) * np.sum((y_new_complex - theta0 - theta1 * x_new ) ** 2)
print("测试机上的损失：", Loss_test)


# Plotting the linear regression result
x_line = np.linspace(0, 10, 100)
y_line = theta0 + theta1 * x_line

plt.figure(figsize=(12, 5))

# Plot 1: Linear Regression Result
plt.subplot(1, 2, 1)
plt.scatter(x, y_complex, color='blue', label='Training points')
plt.scatter(x_new, y_new_complex, color='red', label='Test points')
plt.plot(x_line, y_line, color='green', label='Fitted line')
plt.title('Linear Regression via Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Plot 2: Loss Function Over Time
plt.subplot(1, 2, 2)
plt.plot(loss_history, color='green')
plt.title('Loss Function Minimization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

plt.tight_layout()
plt.show()
