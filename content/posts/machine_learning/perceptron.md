---
head:
  - - link
    - rel: stylesheet
      href: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css

title: 感知机
date: 2025-05-17 15:00:00
categories: 机器学习
cover: [/pics/machine_learning/perceptron/cover.jpg]
sticky: 0
tags: ["机器学习","AI"]
---
![封面](/pics/machine_learning/perceptron/cover.jpg)

# 感知机
&emsp;&emsp;感知机是一种二分类的线性分类模型，属于判别模型，其用于在线性可分的问题空间中划分出一个正负可分的超平面。是神经网络和支持向量机的基础。

&emsp;&emsp;线性可分的一个直观的理解，即在空间中，能够找到一个超平面将不同类的问题直接划分。在二维图像上的表现即能够找到一条直线将具有具有相同特征的点划分在同一个区域。

![原始训练集](/pics/machine_learning/perceptron/Figure_1.png)

![学习效果](/pics/machine_learning/perceptron/Figure_2.png)


## 算法原理
&emsp;&emsp;在感知机算法中，线性可分的问题空间的样本标签被划分为了 +1 -1 两类区域，感知机需要找到能够用于分离这两个样本的超平面。训练集空间为:
$$T={(x_1,y_1), (x_2, y_2), (x_3, y_3),...,(x_N, y_N)}$$
&emsp;&emsp; 其中：$x_i\in X = R^n$，$y_i \in Y = \{-1, +1\}$。（$R^n$ 代表特征空间）

&emsp;&emsp;易得，超平面的方程定义为:
$$H(x) = dot(\vec w, \vec x) + b$$

&emsp;&emsp;向量$\vec w$的几何意义为超平面的法向量，而b则是超平面方程的偏移量。

&emsp;&emsp;在这个超平面的定义下，样本空间中点到超平面的有向距离为：
$$\frac {dot(\vec w, \vec x_i) + b}{||\vec w||}$$

&emsp;&emsp;因此，对于误分类的点，其样本的标签值与公式得到的值一定是负数。令误分类点集合为$M$，所以定义损失函数为：

$$L = -\sum_{\vec x_i \in M} {y_i *(dot(\vec w, \vec x_i) + b)}$$

&emsp;&emsp;关于为什么不考虑 $\frac{1}{||\vec w||}$，我的理解是：感知机的目标是找到**任意一个**能正确分类的超平面。我们需要进行调整的点为误分类的点，而这些点到超平面的有向距离与其标签值的乘积为负。即我们进行判断时，更多的是基于其正负符号判断，其具体值的缩放简化不会影响我们的收敛性计算。

&emsp;&emsp;在确定了损失函数后，我们可以利用梯度下降法优化损失函数以对 w 和 b 值进行调整。

&emsp;&emsp;对w和b分别求偏导，我们可以得到：
$$\nabla{L_w(w, b)} = -\sum_{\vec x_i \in M} {y_i * \vec x_i}$$
$$\nabla{L_b(w, b)} = -\sum_{\vec x_i \in M} {y_i}$$

&emsp;&emsp;即每一次对误分类点进行学习调整后($\eta$表示学习率):
$$\vec w = \vec w + (y_i * \vec x_i) * \eta$$
$$ b = b + y_i*\eta$$

## 具体算法

```python
def perceptron(X: np.ndarray,
               Y: np.ndarray,
               max_epochs: int = 1000,
               learning_rate: float = 1.0
               ) -> (bool, np.ndarray, int):
    """
    训练感知机模型
    :param X: 特征数组，形状 (n_samples（数量）, n_features（特征）)
    :param Y: 标签数组
    :param max_epochs：最大学习次数
    :param learning_rate: 学习率
    :return: (是否收敛, 权重向量, 偏置)
    """
    n_samples, n_features = X.shape

    # 按照特征维度初始化超平面法向量 w
    w = np.zeros(n_features)
    # 初始化截距
    b = 0

    find_answer = False
    for epoch in range(max_epochs):
        has_error = False
        for i in range(n_samples):
            # 计算预测值，检查是否和超平面同侧
            prediction = np.sign(X[i].dot(w) + b)

            if prediction != Y[i]:
                # 如果不是同侧，那么则根据偏导对 w 和 b 进行更新
                w = w + (learning_rate * X[i]) * Y[i]
                b = b + learning_rate * Y[i]
                has_error = True
        if not has_error:
            find_answer = True
            print(f"cost ", epoch, "epochs finish training")
            break
    return find_answer, w, b
    
```

&emsp;&emsp;上述的代码为最基础的感知机算法，$其还有一个优化技巧，即对 $\vec w$ 和 $\vec x_i$进行维度拓展，使用 $\vec W = (w_1, w_2, ..., w_n, b)$和 $X_i = (x_1, x_2, ..., x_n, 1)$ 进行计算。

&emsp;&emsp;原始形式单轮学习的时间复杂度为$O(d*N)$，（d为维度，N为样本数）。而空间复杂度则为 $O(d)$

## 对偶形式
&emsp;&emsp;注意到，由于每一次对超平面方程参数进行调整，参与计算的只有被错误分类的样本。我们假设某样本$x_i$被错误分类了n次，那么这个样本点对参数的贡献分别为：
$$\delta w = \eta n y_i x_i$$
$$\delta w = \eta n y_i$$

&emsp;&emsp;基于此，我们可以将学习的参数对象，从$\vec w$和$b$，转化为 $a = \eta n$，易得$\delta a = \eta$。

&emsp;&emsp;令 alpha 为每个点关于a的数组，样本量为n，其代表着每一个样本点对于参数 ${a}$ 的调整量，那么其每一轮调整后得到的超平面方程为：
$$h(x) = dot((\sum_{j=1}^n y_j * alpha[j] * \vec x_j) \vec x) + \sum_{j=1}^n alpha[j]$$

&emsp;&emsp;注意到，由于每一次对样品点进行评估的时候，都会进行 $dot((\sum_{j=1}^n \vec x_j) \vec x)$的操作，所以我们可以提前先对每一个样品点的$dot((\sum_{j=1}^n \vec x_j) \vec x)$结果进行存储，这结果也称为Gram矩阵。

&emsp;&emsp;因此算法为：
```python
def perceptron_duality(
    X: np.ndarray,
    y: np.ndarray,
    max_epochs: int = 1000,
    learning_rate: float = 1.0
) -> (bool, np.ndarray, float):
    """
    感知机对偶形式算法实现
    :param X: 特征数组，形状 (n_samples, n_features)
    :param y: 标签数组，形状 (n_samples,)，取值 ±1
    :param max_epochs: 最大迭代次数
    :param learning_rate: 学习率
    :return: (是否收敛, 权重向量, 偏置)
    """
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)
    w = np.zeros(n_features)
    b = 0.0
    Gram = X @ X.T  # 预计算 Gram 矩阵（核心优化）

    # 训练循环
    find_answer = False
    for epoch in range(max_epochs):
        has_error = False
        for i in range(n_samples):
            # 计算当前样本的预测值
            sum_predict = np.sum(alpha * y * Gram[:, i]) + b
            if y[i] * sum_predict < 0:
                # 误分类时更新参数
                alpha[i] += learning_rate
                b += learning_rate * y[i]
                has_error = True
        # 所有样本正确分类则提前终止
        if not has_error:
            find_answer = True
            break

    # 计算最终的权重向量 w = Σ(α_i y_i x_i)
    # w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    for j in range(n_samples):
        w += alpha[j] * y[j] * X[j]

    return find_answer, w, b
```

&emsp;&emsp;我们不难发现，对偶形式在每一轮学习的时间复杂度和空间复杂度为$O(N^2)$。因此可以看出，对偶模式在 D > > N 时，效率会比原始模式高很多，但是同样的也存在较大的开销问题。

## 参考
[1]《统计学习方法》(李航)(清华大学出版社)

[2][一文读懂感知机算法](https://zhuanlan.zhihu.com/p/72040253)

[3][机器学习笔记-感知机对偶形式](https://blog.csdn.net/weixin_54814385/article/details/122467399)

[4][【机器学习】感知机原理详解](https://blog.csdn.net/pxhdky/article/details/86360535)

[5][D老师](https://www.deepseek.com)