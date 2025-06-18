### <center>Report of Air Pollution Forecasting - LSTM Multivariate</center> 

<center>Yansong He</center> 

<center>2830791284@qq.com</center> 



#### **Abstract**

​	本研究利用长短期记忆网络（LSTM）对多变量空气污染数据进行预测，旨在准确预测PM2.5浓度。研究使用了一个五年内的小时级天气与污染水平数据集，数据包括日期时间、PM2.5浓度、露点、温度、压力、风向、风速及降水情况等特征。通过数据预处理，计算特征相关性并选择高相关性变量以构建模型输入，最终生成序列数据用于训练LSTM模型。模型经过50轮训练，使用Adam优化器和均方误差（MSE）作为损失函数。模型评估结果显示，预测值与真实值呈现出良好的趋势相符，平均绝对误差（MAE）为12.63，表明该模型在空气质量预测任务中的有效性和实用性，为进一步的环境监测和决策支持提供了基础。



#### **Introduction**

​	空气污染已成为全球面临的重大环境问题，对人类健康和生态系统造成了深远影响。准确预测空气污染水平，尤其是PM2.5浓度，对于有效实施环境管理政策和保护公众健康至关重要。近年来，随着深度学习技术的发展，长短期记忆网络（LSTM）作为一种处理时序数据的有效工具，在空气质量预测领域得到了广泛应用。

​	LSTM网络通过其独特的结构，克服了传统递归神经网络（RNN）在处理长序列时的短期记忆问题，能够有效捕捉到长期依赖关系。而在多变量时间序列预测中，LSTM能够同时处理多种气象因素对PM2.5浓度的影响，提供更为准确的预测结果。本研究的目标是构建并评估LSTM模型，通过有效利用历史污染和天气数据，预测未来的PM2.5水平。

​	本报告的结构如下：部分将概述LSTM模型的基本原理；随后介绍研究中使用的数据集及预处理方法；接着详细描述模型的构建与训练过程；最后，对模型的预测结果进行分析与讨论。通过该研究，期望为环境管理提供科学依据，并为未来进一步的研究打下基础。



#### Models: LSTM

##### 1. **长短期记忆网络**:

- **定义**：LSTM是一种特殊类型的RNN，旨在克服传统RNN在处理长序列时的短期记忆问题。它能够有效地捕捉和学习长期依赖关系。
- **动机**：传统RNN在处理长序列时容易遇到梯度消失或爆炸的问题，这使得模型难以保留早期输入的信息。LSTM引入了一种新的结构来保留和更新信息，有效解决这些问题。

##### 2. **LSTM的基本结构**:

​	LSTM单元包括三个主要的门控机制：**输入门、遗忘门和输出门**，这些门控单元控制信息的流入、存储和输出。LSTM的关键是单元状态，它就像一个传送带，在序列中保持信息的流动，只有少量的线性交互。这样使得信息可以高效地在序列中流动。

![dcaaa49eebbe0331d9e5b08ef1c81fb](F:\360MoveData\Users\admin\Desktop\模式识别与机器学习\文档图片\dcaaa49eebbe0331d9e5b08ef1c81fb.png)

<table>  
    <tr>  
        <td><img src="F:\360MoveData\Users\admin\Desktop\模式识别与机器学习\文档图片\886cac80e6dc3a8a4eddc78a3f1eec9.png" alt="886cac80e6dc3a8a4eddc78a3f1eec9" style="zoom: 80%;" /></td>
        <td><img src="F:\360MoveData\Users\admin\Desktop\模式识别与机器学习\文档图片\3bc4ab387ab2cca2d90a14125611c52.png" alt="3bc4ab387ab2cca2d90a14125611c52" style="zoom:80%;" /></td>  
    </tr>  
</table>

<table>  
    <tr>  
        <td><img src="F:\360MoveData\Users\admin\Desktop\模式识别与机器学习\文档图片\a4063f8201b8131f86e95b49f659bd4.png" alt="a4063f8201b8131f86e95b49f659bd4" style="zoom:80%;" /></td>
        <td><img src="F:\360MoveData\Users\admin\Desktop\模式识别与机器学习\文档图片\e77b757db1341b82bfddafb09686cf9.png" alt="e77b757db1341b82bfddafb09686cf9" style="zoom:80%;" /></td>  
    </tr>  
</table>

###### （1）遗忘门 (Forget Gate)

- **功能**：决定上一单元状态中哪些信息应该被遗忘。

- 公式： $$ f_t = \sigma(W_f \cdot [h_{t-1}, X_t] + b_f) $$ 

  其中：

  - $f_t$ 是遗忘门的输出。

  - $W_f$ 是权重矩阵，$b_f$ 是偏置。

  - $h_{t-1}$ 是上一时间步的隐藏状态。

  - $X_t$ 是当前时间步的输入。

###### （2）输入门 (Input Gate)

- **功能**：确定当前时间步的输入中哪些值将被存储到单元状态中。

- **公式**： $$ i_t = \sigma(W_i \cdot [h_{t-1}, X_t] + b_i) $$ 这里，$i_t$ 表示输入门的输出。

###### （3）输出门 (Output Gate)

- **功能**：决定当前单元状态中哪些信息将被输出。
- **公式**： $$ o_t = \sigma(W_o \cdot [h_{t-1}, X_t] + b_o) $$ 其中，$o_t$ 是输出门的输出。

###### （4）单元状态的更新

​	LSTM具有专门的单元状态 $C_t$，负责存储信息并进行更新。更新过程包括以下几步：

- **计算候选单元状态**

  **候选值**：首先，生成当前输入的候选值 $ \tilde{C_t} $，通过tanh激活函数生成： 
  $$
  \tilde{C_t} = \text{tanh}(W_c \cdot [h_{t-1}, X_t] + b_c)
  $$

- **更新单元状态**
  $$
  C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C_t}
  $$

  - $C_{t-1}$ 是前一时间步的单元状态。
  - $f_t$ 控制从前一状态中遗忘多少信息。
  - $i_t$ 控制从当前输入中添加多少新信息。

###### （5）计算输出

- **输出状态**：最后，LSTM的输出状态 $h_t$ 由**当前单元状态**和**输出门**的控制决定： $$ h_t = o_t \cdot \text{tanh}(C_t) $$

##### 3. **Evolution and Usage of LSTMs**：

- **训练复杂性**:
  - LSTM的训练相较于传统的RNN更为复杂，但在处理长序列问题上表现更佳。
- **多种应用**:
  - LSTM被广泛用于各种任务，包括文本生成、机器翻译、语音识别、视频预测和时间序列分析。
- **单向和双向LSTM**:
  - **单向LSTM**只考虑过去的信息，而**双向LSTM**可以同时考虑过去和未来的信息，这在语音识别和自然语言处理等任务中特别有效。



#### Experimental Studies

##### 1.  问题描述：

​	问题提供了一个报告五年内每小时天气和污染水平的数据集，数据包括日期时间、PM2.5 浓度以及露点、温度、压力、风向、风速和雨雪累积小时数的天气信息。使用这些数据来构建一个预测问题，即根据前几个小时的天气状况和污染情况，预测下一个小时的污染情况。

​	原始数据中的完整特征列表如下：

- **`No` : row number**
- **`year` : year of data in this row**
- **`month` : month of data in this row**
- **`day` : day of data in this row**
- **`hour` : hour of data in this row**
- **`pm2.5` : PM2.5 concentration**
- **`DEWP` : Dew Point**
- **`TEMP` : Temperature**
- **`PRES` : Pressure**
- **`cbwd` : Combined wind direction**
- **`Iws` : Cumulated wind speed**
- **`Is` : Cumulated hours of snow**
- **`Ir` : Cumulated hours of rain**



##### 2. 数据预处理：

​	读取文件，创建date变量，编码类型变量（例如 $cbwd$），处理表格中的缺失值，并输出特征相关性热图，代码及结果如下：

```python
# 计算特征的相关性矩阵
correlation_matrix = data.corr()

# 输出相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('特征相关性热图', fontsize=16)
plt.show()
```

<img src="F:\360MoveData\Users\admin\Pictures\Screenshots\屏幕截图 2025-06-18 203456.png" alt="屏幕截图 2025-06-18 203456" style="zoom: 50%;" />

​	选择相关性强的变量进行模型训练，设定阈值 $threshold = 0.04$，将相关性弱的变量筛除，此处筛除了变量 $Is$，并将剩下的变量组成新的数据集并进行归一化处理，代码如下：

```python
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
```

​	从经过归一化的特征数据中生成序列，采用过去 24 小时的特征信息作为输入，将当前时刻的污染值作为预测目标。随后，将生成的序列数据按 80% 的比例分割为训练集和测试集，以支持模型的训练与评估。此过程使得模型能够通过历史数据学习潜在模式，并对未来的污染水平进行有效预测，从而为空气质量监测提供了坚实的基础，代码如下：

```python
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
```



##### 3. 模型构建与训练：

​	模型架构采用了 `Sequential` 模型，首先定义了输入形状以匹配时间序列的特征。接着，添加了三层 LSTM 单元，其中前两层使用 `return_sequences=True` 参数，以便允许进一步的层叠。为了降低过拟合的风险，在每一层 LSTM 后面均添加了 Dropout 层。最后，使用全连接层（`Dense`）输出一个单一的预测值，代码如下：

```python
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
```

​	在模型编译过程中，设置了学习率为 0.001，并选择 Adam 优化器和均方误差（MSE）作为损失函数。设定训练参数 `epochs=50, batch_size=32`，经过 50 轮的训练，模型利用训练集数据进行优化，同时在验证集上进行评估，以监测其泛化能力。最终，使用测试集数据对模型进行了预测。这一模型设计和训练过程为未来的污染水平预测奠定了基础。



##### 4. 模型预测结果：

- 预测结果与真实值比对：

  <img src="F:\360MoveData\Users\admin\Pictures\Screenshots\屏幕截图 2025-06-18 202515.png" alt="屏幕截图 2025-06-18 202515" style="zoom: 40%;" />

  ​	模型的预测结果与真实值之间的比较显示出了一定的准确性。绘制的预测值与真实值的曲线图表明，模型能够有效地捕捉到污染水平的变化趋势。整体趋势上，预测的 PM2.5 浓度与实际值存在较好的对应关系，特别是在高浓度的污染时段，模型的预测能够较为准确地反映出实际的变化。这表明模型在捕捉时间序列数据的动态变化的能力上表现良好。

  

- 训练损失和验证损失曲线：

  <img src="F:\360MoveData\Users\admin\Pictures\Screenshots\屏幕截图 2025-06-18 202533.png" alt="屏幕截图 2025-06-18 202533" style="zoom:40%;" />

  ​	从训练损失和验证损失曲线的趋势来看，训练损失逐渐下降并趋于平稳，而验证损失同样展现了相似的趋势。这一现象表明模型并未出现严重的过拟合现象。通过比较训练集和测试集的损失，验证了模型的良好泛化能力，能够在未见数据上稳定地进行预测。

  

- 预测值与真实值散点图：

  <img src="F:\360MoveData\Users\admin\Pictures\Screenshots\屏幕截图 2025-06-18 202550.png" alt="屏幕截图 2025-06-18 202550" style="zoom:40%;" />

  ​	对于预测误差的分析，通过散点图进行的展示，真实值与预测值之间的关系进一步体现了模型性能的可靠性。散点图中的数据点大多数集中在 45 度参考线附近，表明模型的预测值与真实值的契合度较高。这意味着，模型在大多数情况下能够准确地反映实际的 PM2.5 浓度变化。

  

- 预测误差的直方图：

  <img src="F:\360MoveData\Users\admin\Pictures\Screenshots\屏幕截图 2025-06-18 202600.png" alt="屏幕截图 2025-06-18 202600" style="zoom:40%;" />

- 性能指标：

  平均绝对误差 (MAE)：12.63

  

​	综合上述分析结果，LSTM 模型在 PM2.5 浓度的预测中展现出了良好的性能，能够有效地捕捉和分析时间序列数据。尽管存在一定的预测误差，但整体趋势的把握及较高的相似度表明，该模型在空气质量监测和预警系统中具备实际应用的潜力。未来的研究可以考虑进一步优化模型结构、参数调节，加入更多的外部因素，以提升预测的准确性和稳定性，从而更好地为环境治理和公共健康提供科学依据。