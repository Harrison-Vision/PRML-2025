import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X += np.random.normal(scale=noise, size=X.shape)

    return X, y


# 生成完整数据集
X, y = make_moons_3d(n_samples=1500, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

# 定义超参数网格
C_range = [0.1, 1, 10, 100]
kernels = ['linear', 'rbf', 'poly']

# 存储结果的字典
results = {}

# 网格搜索
for kernel in kernels:
    training_errors = []
    test_errors = []

    for C in C_range:
        # 创建SVM分类器进行训练
        clf = SVC(kernel=kernel, C=C, random_state=42)
        clf.fit(X_train, y_train)

        # 记录训练误差和测试误差
        training_errors.append(1 - clf.score(X_train, y_train))
        test_errors.append(1 - clf.score(X_test, y_test))

    # 存储结果
    results[kernel] = {
        'training_errors': training_errors,
        'test_errors': test_errors
    }

# 为每个核函数创建独立的图形窗口
for kernel, result in results.items():
    # 创建新的图形窗口
    plt.figure(figsize=(10, 6))

    # 绘制训练和测试误差
    plt.plot(C_range, result['training_errors'], label='Training Error', marker='o')
    plt.plot(C_range, result['test_errors'], label='Test Error', marker='s')

    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Error')
    plt.title(f'SVM: Kernel = {kernel}')
    plt.xscale('log')  # 对C使用对数尺度
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 选择最佳超参数
best_performance = float('inf')
best_params = {}

for kernel, result in results.items():
    min_test_error = min(result['test_errors'])
    if min_test_error < best_performance:
        best_performance = min_test_error
        best_params['kernel'] = kernel
        best_params['C'] = C_range[result['test_errors'].index(min_test_error)]

print("\n最佳参数:")
print(f"核函数: {best_params['kernel']}")
print(f"正则化参数C: {best_params['C']}")

# 使用最佳参数训练模型
best_clf = SVC(
    kernel=best_params['kernel'],
    C=best_params['C'],
    random_state=42
)
best_clf.fit(X_train, y_train)

# 预测
y_pred = best_clf.predict(X_test)

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Kernel={best_params["kernel"]}, C={best_params["C"]})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))

# 额外的性能指标
print("\nSVM性能指标:")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
