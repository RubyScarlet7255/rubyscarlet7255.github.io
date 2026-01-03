---
head:
  - - link
    - rel: stylesheet
      href: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css

title: 朴素贝叶斯
date: 2025-05-19 22:00:00
categories: 机器学习
cover: [/pics/machine_learning/normal_bayes_and_logistic_regression/cover.jpg]
sticky: 0

tags: ["机器学习","AI"]

---

![封面](/pics/machine_learning/normal_bayes_and_logistic_regression/cover.jpg)

# 朴素贝叶斯

&emsp;&emsp;贝叶斯分类是一种基于贝叶斯定理的分类学习方法，一句话概括其根据样本的特征计算相关的后验概率，然后根据后验概率最大的输出作为其预测标签。而朴素贝叶斯则在此基础上拥有**样本独立性假设**，这使得其使用会更加的简单，然而这并不妨碍它是个非常实用的方法。由于它的原理是学习生成数据的机制，所以它是一种生成模型。

&emsp;&emsp;小提一嘴，因为小站对于latex的支持并没有太好，所以有些地方会避免使用中文，请见谅。


## 联合概率分布

&emsp;&emsp;在开始之前，先简单复习一下联合概率分布。研究多维的随机变量分布时，不仅想要包含每个随机变量组各自的分布关系，也需要包含不同维度之间的互相关系的信息。而这种分布称为联合分布。

&emsp;&emsp;拿二维随机变量$(X，Y)$来说，我们定义$P(X \leq x, Y \leq y) = P(\{X \leq x\} \cap \{Y \leq y \}) = P((X,Y) \in D_{xy})$。

**定义** 设$(X_1, X_2,..., X_n)$为n维随机变量，对于**任意**$(x_1, x_2, ..., x_n) \in R^n$，称下式：
$$F(x_1, x_2, ..., x_n) = P(X_1 \leq x_1, X_2 \leq x_2,..., X_n \leq x_n)$$ 

为随机变量 $(X_1, X_2,..., X_n)$ 的联合分布函数。

&emsp;&emsp;举个具体的例子，想象你要预测明天是否会下雨（随机变量$X$）和是否要带伞（随机变量$Y$）。  
- **单独概率**：  
  - $P(X=雨)$：明天下雨的概率  
  - $P(Y=带伞)$：你带伞的概率  
- **联合分布**：  
  - $P(X=雨, Y=带伞)$：**同时考虑**天气和你行为的概率

联合分布就是描述多个变量**共同发生**的概率规律,而联合分布函数，则会**描述**所有你带伞的情况下下雨的概率，和所有下雨情况下你带伞的概率。

### 二维离散型分布随机变量及其联合分布律

&emsp;&emsp;二维离散型分布随机变量及其联合分布律定义如下，多维情况可以类比理解。

&emsp;&emsp;**定义**：如果二维随机变量$(X,Y)$只可能取到有限多或者无限多个可列值，那么$(X,Y)$则称为二维离散型随机变量。

&emsp;&emsp;而其联合分布律定义为：

&emsp;&emsp;**定义**：$P(X = x_i, Y = y_j) = p_{ij}, i, j = 1,2,...$。其中，$p_{ij} \geq 0, i,j = 1,2,...,  \sum_i \sum_j p_{ij} = 1$

&emsp;&emsp;再拿下雨的例子说明，联合分布律是枚举出所有下雨和你是否带伞的情况。

&emsp;&emsp;假设$X$表示天气（晴=0, 雨=1），$Y$表示是否带伞（不带=0, 带=1），其联合分布律可能为：
|       | $Y=0$ | $Y=1$ |
|-------|-------|-------|
| $X=0$ | 0.3   | 0.1   |
| $X=1$ | 0.2   | 0.4   |



### 二维连续型随机变量及其联合概率密度函数

&emsp;&emsp;**定义**：设二维随机变量$(X,Y)$的联合分布函数为 $F(x,y)$，如果存在一个二元非负实值函数 $f(x,y)$，使得任意的 $(x, y) \in R^2$ 满足：
$$F(x,y) = \int_{- \infty}^{x}\int_{- \infty}^{y}f(u,v)d_ud_v$$
&emsp;&emsp;那我们称 $(X,Y)$为二维随机变量，$f(x，y)$为二维随机变量$(X,Y)$的联合概率密度函数。

&emsp;&emsp;高维用法类推。

&emsp;&emsp;想象你在玩一个游乐园里面打沙包的游戏，你面前的墙是一个凹槽均匀过渡，绘制着不同大小孔洞。你把球扔到洞中就可以获得奖励。同时洞越小奖励越高。总之，概率密度函数描述的是连续随机变量空间中，某个具体点的概率。


## 朴素贝叶斯

### 朴素贝叶斯原理

&emsp;&emsp;朴素贝叶斯建立在最基本的**特征条件独立假设**之上，设输入空间$X \subseteq R^n$，输出空间为类标记为集合$Y={c_1,c_2,c_3,...,c_k}$。定义输入特征向量为$x \in X$, 输出类标签为$y \in Y$。即$x$为输入空间$X$的随机向量，而$y$是输出空间在$Y$中的随机标签值。训练集定义为：

$$T=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$$

&emsp;&emsp;$P(X, Y)$ 是输入特征 $X$ 和类别标签 $Y$ 的联合概率分布，基于我们的**条件特征独立假设**，在给定类别标签 $Y = c_k$ 的条件下，输入特征的各个维度 $X^{(j)}$ 是条件独立的，即表示为：

$$P(X=x | Y = c_k) = P(X^{(1)} = x^{(1)}, ..., X^{(n)} = x^{(n)} | Y = c_k)$$
$$= \prod_{j=1}^n P(X^{(j)} = x^{(j)} | Y = c_k)$$

&emsp;&emsp;而朴素贝叶斯方法，则是通过学习输入x对于y的生成模式，根据计算的后验概率：

$$P(Y = c_k | X = x) 
= \frac {P(X = x | Y = c_k)P(Y=c_k)} {\sum_{k=1}^K P(X = x | Y = c_k)P(Y = c_k)}$$
$$= \frac{P(Y=c_k) \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_{k=1}^K P(Y=c_k) \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)},   k = 1,2,...,K$$

&emsp;&emsp;找到${X=x}$时，后验概率最大的$y=c_k$。所以朴素贝叶斯分类器可以表示为：

$$ y = \arg\max_c P(Y=c_k)  \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)$$

### 多项式特征（示例：垃圾邮件问题）
&emsp;&emsp;垃圾邮件分类是一个朴素贝叶斯的常见示例，基于朴素贝叶斯的垃圾邮件分类是根据不同词在垃圾邮箱中出现的频率学习，然后进行预测的一种方式。尽管朴素贝叶斯对于垃圾邮箱分类的方式比较直接，但是其仍然具有较好的分类效果。

&emsp;&emsp;在垃圾邮箱分类问题中，特征维度 $x_a = j$ 表示文本词典 $x$ 的第 $\alpha$ 个单词在词典中出现了 $j$ 次。其基本思想是，如果一个词语在垃圾邮件中出现了 $j$ 次，那么在一个该词语出现了 $j$ 次或者更多次的邮件更有可能是个垃圾邮件。

&emsp;&emsp;令 $m$ 代表着一个邮件文本中所有单词的技术，$d$ 代表着整个词典大小（词典是根据所有邮件的所有不同单词生成的，不存在重复）。那么这个问题可多项式分布表示为 $P(x|y)$：

$$P(x|m, y=c)= \frac{m!}{x_1!*x_2!*...*x_d!} \prod_{\alpha=1}^d (\theta_{ac})^{x_a}$$

&emsp;&emsp;其中，多项式左侧代表着文件中这些单词在这个文本中出现的所有排列组合情况。而右侧则代表着这些序列为垃圾邮件的等效概率。而 $\theta_{\alpha c}$ 则代表选择特征维度 $x_{\alpha}$ 的概率。$\theta_{\alpha c}$ 的参数估计如下。

$$\theta_{\alpha c} = \frac{\sum_{i=1}^{n}I(y_i=c)x_{i\alpha} + l}{
  \sum_{i=1}^{n}I(y_i=c)m_i+ l*d
}$$ 

&emsp;&emsp;$m_i$代表着文本 $i$ 中单词的总数，而 $I(y_i=c)$ 则是判别函数，其含义为:

$$I_{\{y_i = c\}} = \begin{cases}

0, & y_i = c \\

1, & y_i \neq c

\end{cases}$$


&emsp;&emsp;$l$ 代表平滑参数，当 $l = 1$ 时我们称为拉普拉斯平滑。平滑参数可以有效防止因为某些内容没有在垃圾邮件文本中出现，导致 $P(x|m, y=c)$ 为 0 以使得无法准确判断的情况。

&emsp;&emsp;回到 $P(x|m, y=c)$ 中，由于多项式左侧是常数项，所以我们只需要学习 $\prod_{\alpha=1}^d (\theta_{ac})^{x_a}$ 。又因为判别函数的特点，$I(y_i=c)x_{i\alpha}$ 和 $I(y_i=c)m_i$ 则可以完全通过计数计算。因此我们实际上的特征维度代表的含义为：

$$\frac{ \text{ total appearences of word x in class C} }{\text{word counts in all text of class C} }$$

&emsp;&emsp;由此在训练得到了联合分布律之后，预测方法可以表示为：
$$\arg\max_{c} P(Y=c) \propto \arg\max_{c} P(Y=c) \prod_{\alpha=1}^d \theta_{\alpha c} $$

#### 代码示例
&emsp;&emsp;训练的数据集采用的来自kaggle的[sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)，这是一个垃圾短信的训练集，其对于新接触python的同学来说非常友好。

```python
import math

from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

df = pd.read_csv("./data/spam.csv", delimiter=',',encoding='latin-1')
dataSet = df.values
y = dataSet[:, 0]
X = dataSet[:, 1]

# 正确划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,  # 使用转换后的标签数组
    test_size=0.3,
    random_state=42,
)

class NaiveBayesSpamClassifier:
    """
    @param alpha: 平滑系数
    """
    def __init__(self, alpha: int = 1):
        self.alpha = alpha
        # 词典
        self.wordDict = set()
        # 类型概率（后验概率）
        self.class_probs = {}
        # 用于进行不同类别的单词出现次数计数
        self.class_word_prop = {'spam': {}, 'ham': {}}

        # log 处理前置概率避免下溢
        self.log_prop_spam = 0
        self.log_prop_ham = 0


    def preprocess_data(self, text: str):
        return text.split()

    def train(self, texts: list[str], labels: list[str]):
        ham_word_dict_counter = defaultdict(int)
        spam_word_dict_counter = defaultdict(int)

        total_ham_words = 0
        total_spam_words = 0

        spam_count = sum(1 for label in labels if label == 'spam')
        ham_count = len(labels) - spam_count


        self.log_prop_spam = math.log(spam_count / len(labels))
        self.log_prop_ham = math.log(ham_count / len(labels))

        # 遍历所有文本，统计词数
        for text, label in zip(texts, labels):
            words = self.preprocess_data(text)
            self.wordDict.update(words)  # 更新词汇表

            if label == 'spam':
                for word in words:
                    spam_word_dict_counter[word] += 1
                    total_spam_words += 1
            else:
                for word in words:
                    ham_word_dict_counter[word] += 1
                    total_ham_words += 1

        dict_size = len(self.wordDict)
        for word in self.wordDict:
            # 计算每个单词的 分布，log防止下溢
            word_prop_ham = (ham_word_dict_counter[word] + self.alpha)/(total_ham_words + self.alpha * dict_size)
            word_prop_spam = (spam_word_dict_counter[word] + self.alpha)/(total_spam_words + self.alpha * dict_size)
            self.class_word_prop["ham"][word] = math.log(word_prop_ham)
            self.class_word_prop["spam"][word] = math.log(word_prop_spam)

    def predict(self, text) -> str:
        words = self.preprocess_data(text)
        prop_ham = self.log_prop_ham
        prop_spam = self.log_prop_spam

        for word in words:
            if word in self.wordDict:
                prop_spam += self.class_word_prop["spam"][word]
                prop_ham += self.class_word_prop["ham"][word]

        if prop_spam > prop_ham:
            return "spam"
        return "ham"

    def evaluate(self, texts: list[str], labels: list[str]) -> float:
        correct_cnt = 0
        for text, label in zip(texts, labels):
            predict_res = self.predict(text)
            if predict_res == label:
                correct_cnt += 1

        return correct_cnt / len(labels)


classifier = NaiveBayesSpamClassifier()
classifier.train(X_train, y_train)
correct_rate = classifier.evaluate(X_test, y_test)
print("预测准确率为", correct_rate)
```

### 连续性特征（用例：鸢尾花分类问题）

&emsp;&emsp;当数据表现为连续特征时，常见的如自然状态下的各种可观测属性。通常使用高斯分布来进行表征。即：

$$X_\alpha \in R$$

$$P(x_\alpha | y = c) = N(\mu_{\alpha c}, \sigma^2_{\alpha c}) = \frac{1}{\sqrt{2 \pi} \sigma_{\alpha c} } e^{ -\frac{1}{2} (\frac{x_\alpha - \mu_{\alpha c} }{\sigma_{\alpha c} })^2 }$$

&emsp;&emsp;即面对连续性特征时的，我们根据每种类型不同特征的正态分布来计算每一维度的条件概率 $P(x_\alpha | y = c)$。

&emsp;&emsp;连续性特征的一个常用示例为鸢尾花问题，鸢尾花作为一种自然存在的物种，同品类每个个体之间的生物性状都符合某种状态分布。

&emsp;&emsp;值得注意的是，在写实际代码时，需要防止出现计算出方差为0的情况。

#### 代码示例

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# 加载莺尾花数据集
from sklearn import datasets
from sklearn.model_selection import train_test_split

class GaussianNaiveBayes:

    def __init__(self):

        # 标签分类
        self.labels: list[any] = None

        # 每个标签的前置概率
        self.label_prop: list[int] = None

        # 每个类别不同维度属性的均值
        self.class_means: list[int, float] = None

        # 每个类别不同属性的方差
        self.class_vars: list[int, float] = None

    def train(self, X, y):

        self.labels = np.unique(y)
        n_classes = len(self.labels)
        n_features = X.shape[1]

        # 初始化存储结构
        self.label_prop = np.zeros(n_classes)
        self.class_means = np.zeros((n_classes, n_features))
        self.class_vars = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.labels):
            X_c = X[y == c]
            # 计算每一类的后验概率
            self.label_prop[i] = X_c.shape[0] / X.shape[0]
            # 计算同品种每一类特征的期望
            self.class_means[i, :] = X_c.mean(axis=0)
            # 计算同品种每一类特征的方差
            self.class_vars[i, :] = X_c.var(axis=0, ddof=1)

    def calculate_predict(self, x, mean, var):
        """计算高斯概率密度"""
        # 防止方差为0（添加小epsilon）
        var = np.maximum(var, 1e-9)
        exponent = -((x - mean) ** 2) / (2 * var)
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(exponent)

    def predict(self, X):
        log_probs = []
        for i, c in enumerate(self.labels):
            prior = np.log(self.label_prop[i])
            # 对每个特征独立计算条件概率
            likelihood = 0
            for j in range(len(X)):
                likelihood += np.log(self.calculate_predict(
                    X[j],
                    self.class_means[i, j],
                    self.class_vars[i, j]
                ))
            log_probs.append(prior + likelihood)

        # 选择最大概率类别
        return self.labels[np.argmax(np.array(log_probs), axis=0)]

    def evaluate(self, X, y) -> float:
        correct_cnt = 0
        for X_c, label in zip(X, y):
            predict_res = self.predict(X_c)
            if predict_res == label:
                correct_cnt += 1
        return correct_cnt / len(y)


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = GaussianNaiveBayes()
classifier.train(X_train, y_train)
correct_rate = classifier.evaluate(X_test, y_test)

print("预测准确率为", correct_rate)
```

### 朴素贝叶斯是线性分类器

#### 多项式形式证明
&emsp;&emsp;假设 $y_i \in \{-1, +1\}$，并且特征是多项式形式的。根据前文可以得到预测函数即为：
$$h(x) = \arg\max_c P(Y=c_k)  \prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)$$

&emsp;&emsp;欲证明其符合线性分类器特征即通过证明：
$$w^Tx+b > 0 \Longleftrightarrow h(x) = +1$$

**证**：

&emsp;&emsp;定义 $P(x_{\alpha} | y = +1) \propto \theta_{\alpha_+}^{x_\alpha}$,定义$P(x_{\alpha} | y = -1) \propto \theta_{\alpha_-}^{x_\alpha}$, $P(Y = +1) = \pi_+$，$P(Y = -1) = \pi_-$

&emsp;&emsp;令：
$$[w]_\alpha = log(\theta_{\alpha+}) - log(\theta_{\alpha-})$$
$$b = log(\pi_+) - log(\pi_-)$$

&emsp;&emsp;则：

$$w^T + b > 0 \Longleftrightarrow \sum_{\alpha = 1}^d[x]_{\alpha}[w]_{\alpha}+b > 0$$

$$\Longleftrightarrow exp(\sum_{\alpha = 1}^d[x]_{\alpha}[w]_{\alpha}+b) > 1$$

$$\Longleftrightarrow exp(\sum_{\alpha = 1}^d[x]_{\alpha}(log(\theta_{\alpha+}) - log(\theta_{\alpha-}))+log(\pi_+) - log(\pi_-)) > 1$$
$$\Longleftrightarrow \frac{\pi_+}{\pi_-}\prod_{\alpha = 1}^d\frac{ \theta_{\alpha_+}^{[x]_\alpha} }{ \theta_{\alpha_-}^{[x]_\alpha} } > 1$$
$$\Longleftrightarrow \frac{\pi_+}{\pi_-}\prod_{\alpha = 1}^d\frac{ P([x]_{\alpha}|Y=+1)}{ P([x]_{\alpha}|Y=-1) } > 1$$
$$\Longleftrightarrow \frac{P(x|Y=+1)\pi_+}{P(x|Y= -1)\pi_-} > 1$$
$$\Longleftrightarrow \arg\max_cP(Y=y|x) = +1$$

## 参考

[1]《统计学习方法》(李航)(清华大学出版社)

[2]《概率论与数理统计》 (同济大学数学系)（人民邮电出版社）

[3]《机器学习讲义（何琨）》

[4][D老师](https://www.deepseek.com)