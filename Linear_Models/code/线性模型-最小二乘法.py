import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define the classical gradient descent algorithm
def Least_Squares(x, y):
    # 将x表示为矩阵形式
    X = x.reshape(len(x), -1)
    X = np.column_stack([np.ones(X.shape[0]), X])

    # 将y表示为矩阵形式
    Y = y.reshape(-1, 1)
    n = len(y)
    theta = np.zeros((X.shape[1], 1)) # Initial parameters

    # 最小二乘方法拟合数据
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # 解包theta
    theta0, theta1 = theta.flatten()

    return theta0, theta1


# Read the dataset from the CSV file
data_training = pd.read_csv('Data4Regression - Training Data.csv')
data_test = pd.read_csv('Data4Regression - Test Data.csv')

# Extract x and y from the dataframe
x = data_training['x'].values
y_complex = data_training['y_complex'].values

x_new = data_test['x_new'].values
y_new_complex = data_test['y_new_complex'].values


# Apply gradient descent
theta0, theta1 = Least_Squares(x, y_complex)
print(f"训练结果: theta0 = {theta0}, theta1 = {theta1}")

# Calculate the loss on the test set
Loss_test = (1 / (2 * len(y_new_complex))) * np.sum((y_new_complex - theta0 - theta1 * x_new ) ** 2)
print("测试机上的损失：", Loss_test)


# Plotting the linear regression result
x_line = np.linspace(0, 10, 100)
y_line = theta0 + theta1 * x_line

plt.figure(figsize=(6, 5))

# Plot : Linear Regression Result
plt.scatter(x, y_complex, color='blue', label='Training points')
plt.scatter(x_new, y_new_complex, color='red', label='Test points')
plt.plot(x_line, y_line, color='green', label='Fitted line')
plt.title('Linear Regression via Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()  # 显示图形
