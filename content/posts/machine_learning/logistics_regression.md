---
head:
  - - link
    - rel: stylesheet
      href: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css

title: 逻辑回归
date: 2025-6-21 22:00:00
categories: 机器学习
cover: [/pics/logistic_regression/cover.jpg]
sticky: 0

tags: ["机器学习","AI"]

---

![封面](/pics/logistic_regression/cover.jpg)

# 逻辑回归

&emsp;&emsp;逻辑回归和朴素贝叶斯有诸多相似，并且证明贝叶斯是线性模型的方式在连续下（高斯朴素贝叶斯），可以见到逻辑回归重要的**sigmod**函数产物。原本打算放在一起写的，但是后来意识到篇幅实在太大，所以就分成了两个文章。

## 逻辑回归和朴素贝叶斯
&emsp;&emsp;朴素贝叶斯是一个生成模型，而逻辑回归是一个判别模型。在朴素贝叶斯进行 $P(x_i| y)$ 预测的时候，其对分别对 $P(x_i | y)$ 和 $P(y)$进行建模。而逻辑回归则是直接对 $P(y | x_i)$ 进行建模。

## 逻辑斯谛 (logistic distribution)

&emsp;&emsp;设$X$是连续随机变量，如果$X$服从逻辑斯谛分布，那么其具有下列分布函数和密度函数：

$$F(x) = P(x \leq x) = \frac{1}{ 1+e^{-(x- \mu)/\gamma } }$$

$$f(X) = F^{'}(x) = \frac{ e^{-(x- \mu) / \gamma} }{\gamma (1+e^{-(x- \mu)/\gamma } ) ^2} $$

&emsp;&emsp;其分布函数为 $F(x)$ 而密度函数为 $f(x)$，其图像如下所示：

![函数图像](/pics/logistic_regression/sigmod.png)


&emsp;&emsp;其图像是一条S形曲线(simoid curve)，该曲线关于点 $(\mu , \frac{1}{2})$ 中心对称，即：

$$F(-x + \mu) - \frac{1}{2} = -F(x - \mu) + \frac{1}{2}$$

&emsp;&emsp;图形两边增长慢，中心增长快，形状参数 $\gamma$ 越小，其中心点附近增长越快。

## 二项逻辑斯谛回归

&emsp;&emsp;二项逻辑回归模型是一个基础的分类模型，由条件概率分布 $P(Y | X)$ 表示，形式为参数化的逻辑斯谛分布。其仅将其仅将随机变量 $Y$ 取值为 1 或0。其中:

$$P(Y = 1 | \vec{x}) = \frac{exp(\vec{w} * \vec{x} + b)}{1+exp(\vec{w} * \vec{x} + b)}$$

$$P(Y = 0 | \vec{x}) = \frac{1}{1+exp(\vec{w} * \vec{x} + b)}$$

&emsp;&emsp;可以通过将权值向量 $\vec{w}$ 和 输入向量 $\vec{x}$ 扩充以省略参数 $b$ ，此时：

$$P(Y = 1 | \vec{x}) = \frac{exp(\vec{w} * \vec{x})}{1+exp(\vec{w} * \vec{x})}$$

$$P(Y = 0 | \vec{x}) = \frac{1}{1+exp(\vec{w} * \vec{x})}$$

&emsp;&emsp;通常来说，为了方便编程与书写，$P(Y = 1 | \vec{x})$ 一般写成：

$$\frac{1}{1+exp(-(\vec{w}*\vec{x}))}$$

&emsp;&emsp;二项分类也可以推广到多项的情况，即变量 $Y$ 的集合为 ${1, 2, ... K}$ 。此时的分布模型如下：

$$P(Y = k | \vec{x}) = \frac{exp(\vec{w}_k * \vec{x})}{1+ \sum_{k=1}^{K-1} exp(\vec{w}_k * \vec{x})}$$

## 参数估计

### 极大似然估计 (MLE)

&emsp;&emsp;极大似然估计，是一种根据已知的样本结果信息，反推最有可能得到此结果的输入值。其数学表达形式如同：

$$MLE(\theta) = \arg\max_{\theta} P(y | \theta)$$

&emsp;&emsp;极大似然估计还有一个很重要的假设，即样本之间是独立的。

&emsp;&emsp;我们采用 $y \in \{-1, 1\}$ ，并且采用维度扩充的写法，即：

$$ P(y |\vec{x}) = \frac{1}{1+exp(-y*(\vec{w}*\vec{x}))} $$

&emsp;&emsp;在逻辑回归中，我们根据统计的结果 $y$ 和输入向量 $x$，对权值向量 $\vec{w}$进行极大似然估计。即基于样本，最大化的概率函数：

$$P(y |\vec{x} ,\vec{w}) = \prod_{i = 0}^n P(y_i | \vec{x_i}, \vec{w})$$

$$P(y |\vec{x} ,\vec{w}) = \prod_{i = 0}^n \frac{1}{1+exp(-y_i*(\vec{w}*\vec{x_i}))} $$

$$log(P(y |\vec{x} ,\vec{w})) = -\sum_{i=0}^nlog(1+exp(-y_i*(\vec{w}*\vec{x_i})))$$

&emsp;&emsp;所以要估计的参数 $\vec{w}$ 即：

$$ \vec{w}_{MLE} = \arg\max_w log(-\sum_{i=0}^n(1+exp(-y_i*(\vec{w}*\vec{x_i})))) $$

$$ \vec{w}_{MLE} = \arg\min_w \sum_{i=0}^nlog(1+exp(-y_i*(\vec{w}*\vec{x_i}))) $$

&emsp;&emsp; 此时采用梯度下降对 $\vec{w}$ 进行估计即可。

### 最大后验估计（MAP）

&emsp;&emsp;最大后验估计可以视为在极大似然估计之上引入了先验分布，因为很多时候采集大量的数据是非常困难的，此时极大似然估计会容易出现过拟合这样的情况。于是引入了先验概率，确定某种数据假定条件，以进一步约束。

&emsp;&emsp;最大后验估计中，我们不仅要使得求得的 $\theta$ 的似然函数最大， $\theta$ 自己的先验概率也要最大，即：

$$\theta^{map} = \arg\max_{\theta}P(\theta|D) $$

&emsp;&emsp;回到逻辑回归，将 $\vec{w}$ 看作是一个随机变量，并且假设它服从某种先验概率。

$$P(\vec{w} | D) = P(\vec{w} | X,y) \propto P(y | X, \vec{w}) * P(\vec{w}) $$

$$\vec{w}_{MAP} = \arg\max_{ \vec{w} }log(P(y | X, \vec{w})P(\vec{w}))$$

$$\vec{w}_{MAP} = \arg\min_{ \vec{w} } \sum_{i=0}^nlog(1+e^{-y_i (\vec{w} * \vec{x_i})}) + log(P(\vec {w}))$$

&emsp;&emsp;$log(P(\vec {w}))$ 为正则项，其具体形式和问题相关的分布有关，比如在高斯分布中，其形式为 $\lambda ||\vec{w}||^2$，$\lambda = \frac{1}{2\sigma}$。

&emsp;&emsp;在最后，会使用梯度下降法进行寻找近似的参数 w 解。

&emsp;&emsp;注意，和朴素贝叶斯不同，逻辑回归中方差 $\sigma$ 用于控制模型的离散程度，是先验分布的参数。而高斯朴素贝叶斯中的方差则是会直接参与后验概率的计算。所以往往在逻辑回归的实现中，方差是作为学习的参数进行传递的，而不像高斯朴素贝叶斯是通过样本计算得到的。

&emsp;&emsp;以下是一个基于MAP的代码实现：

```python
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, l2_lambda=0.1, verbose=False):
        """
        初始化逻辑回归模型
        :param learning_rate: 学习率 (α)
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值（梯度范数小于此值停止）
        :param l2_lambda: L2正则化强度（相当于MAP中的高斯先验）
        :param verbose: 是否打印训练过程
        """
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.weights = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid函数实现"""
        # 数值稳定版本，防止大z溢出
        # sigmod 函数中 1-sigmod(z) = sigmod(-z)
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _add_intercept(self, X):
        """添加偏置项（截距）到特征矩阵"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _compute_loss(self, X, y):
        """计算损失函数（带L2正则化的对数损失）"""
        z = X.dot(self.weights)
        h = self._sigmoid(z)

        # 核心损失计算：sigmoid 对数损失
        log_loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        # 正则项损失
        reg_loss = 0.5 * self.l2_lambda * np.sum(self.weights[1:] ** 2)  # 忽略偏置项

        return log_loss + reg_loss

    def _compute_gradient(self, X, y):
        """计算梯度（含正则化项）"""
        n = X.shape[0]
        z = X.dot(self.weights)
        h = self._sigmoid(z)

        # 核心梯度计算
        error = h - y
        gradient = (X.T.dot(error)) / n

        # 添加L2正则化梯度（偏置项不参与正则化）
        reg_gradient = np.concatenate(([0], self.l2_lambda * self.weights[1:]))

        return gradient + reg_gradient

    def fit(self, X, y):
        """
        训练逻辑回归模型
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签向量 (n_samples,)，取值{0,1}
        """
        # 添加偏置项并初始化权重
        X = self._add_intercept(X)
        self.weights = np.zeros(X.shape[1])

        # 梯度下降主循环
        for i in range(self.max_iter):
            # 计算梯度和损失
            gradient = self._compute_gradient(X, y)
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            # 更新权重
            self.weights -= self.lr * gradient

            # 检查收敛（梯度范数小于阈值）
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, gradient norm: {grad_norm:.6f}")
                break

            # 每100次迭代打印进度
            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: Loss={loss:.4f}, ||Grad||={grad_norm:.6f}")

    def predict_proba(self, X):
        """预测概率 P(y=1|x)"""
        X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.weights))

    def predict(self, X, threshold=0.5):
        """预测类别（默认阈值0.5）"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def decision_function(self, X):
        """计算决策函数值 w^T x + b"""
        X = self._add_intercept(X)
        return X.dot(self.weights)
```