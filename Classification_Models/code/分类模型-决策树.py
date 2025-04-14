import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y


# 生成完整数据集
X, y = make_moons_3d(n_samples=1500, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)  # 使用 train_test_split 划分数据集

# 定义迭代次数范围
depths = range(1, 13)
training_errors = []
test_errors = []

# 对不同的迭代次数进行测试
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    # Record the training error
    training_errors.append(1 - clf.score(X_train, y_train))

    # Record the test error
    test_errors.append(1 - clf.score(X_test, y_test))

# 绘制误差曲线图
plt.figure(figsize=(10, 5))
plt.plot(depths, training_errors, label='Training Error', marker='o')
plt.plot(depths, test_errors, label='Test Error', marker='s')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Error')
plt.title('Decision Tree Depth vs. Training and Test Errors on Wine Dataset')
plt.legend()
plt.grid(True)
plt.show()

# 选择最佳深度（使用测试误差最低的深度）
best_depth = depths[np.argmin(test_errors)]
print(f"最佳树深度: {best_depth}")

# 使用最佳深度训练模型
best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_clf.fit(X_train, y_train)

# 预测
y_pred = best_clf.predict(X_test)

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Depth={best_depth})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))

# 额外的性能指标
print("\nDecision Tree性能指标:")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
