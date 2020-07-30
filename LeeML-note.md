# L1 Introduction

![](./images/HW.png)

- 机器学习就是自动找函数（任务决定函数），使得loss最低。

## Three Steps
1. Function set(Model)
2. Goodness of a function
3. Find the best function

## 机器学习分类
- Regression
- Classification
- Generation: Seq2seq, GAN
- ...

# L2 Regression
## 定义
- 回归：找到一个function，通过输入特征x，输出数值scaler。
- 应用：Stock market forecast, Self-driving Car, Recommendation

## 模型步骤
1. 假设-选择框架（线性）
2. 评估-判断模型好坏（Loss func）
3. 优化-筛选最优模型（梯度下降）

## Regularization
- 调整平滑程度，对输入不敏感
- 不需要正则化常数项，因为它不影响平滑

## Error
- bias 偏差 -> underfitting
  - redesign model
- variance 方差 -> overfitting
  - more data(collect or generate)
  - regularization

## Model Selection
- training set
- validation set

- N-folf cross validation

# L3 Gradient Descent
## Adagrad
- 这次梯度与之前的反差
- 与最小值点距离 ~ 一次微分/二次微分
- adagrad在没有增加太大负担的情况下估算了二次微分

## Stochastic Gradient Descent
- Gradient Descent see all one time
- SGD see one example one time

## Feature Scaling
- 向着最低点走，gradient descent faster

## 梯度下降理论

# L4 Classification
- 不能用regression来分类，因为会惩罚太正确的值（比如>>1），并且多元分类。
  
## Generative Model
1. 假设每个class sampled from Gaussian distribution (mean and covariance matrix)
2. Maximum Likelihood 找出某个model（$\mu, \Sigma$） sample出已知样本的可能性最大(posterior probability)，就认为是其分布。微分得零点即为结果。
3. 贝叶斯公式算出属于某个类比的概率。

## Modifying Model
  - share covariance matrix，防止overfit。通过样本数加权得到。
  - 变形得到$P(C_1|x)=\sigma(w·x + b)$，share产生了linear model
  - 既然线性？为什么不直接算w和b？Logistic regression

## Probability Distribution
  - 分布适合数据
  - 比如二元数据假设来自 Bernoulli distribution
  - Naive Bayes Classifier：假设变量全independent

# L5 Logistic Regression
## Three Steps
1. Model: Discriminative （没有假设Gaussian等）
2. Goodness: Cross entropy of two Bernoulli distribution
3. Best function: Gradient Descent计算式（竟然）和Linear一样

> ### Cross Entropy
> - 信息量：$I(x) = -log(p(x))$，概率大，信息少
> - Entropy：所有信息量的期望$H(x)=E[I(x)]=-\sum_{i=1}^n p(x_i) \log(p(x_i))$
> - Bernoulli distribution 熵：$H(x)=-p(x) \log (p(x))-(1-p(x)) \log (1-p(x)$
> - Cross Entropy：$H(p, q)=-\mathrm{E}_{p}[\log q]$
> - Relative entropy (Kullback–Leibler divergence)：$D_{K L}(p \| q)=\sum_{i=1}^{n} p(x_{i}) \log (\frac{p(x_{i})}{q(x_{i})})$
>   - In the context of machine learning, DKL(P‖Q) is often called the information gain achieved if P is used instead of Q.
>   - KL散度（距离）越小，p、q分布越接近
>   - $D_{K L}(p \| q) =\sum_{i=1}^{n} p(x_{i}) \log (p(x_{i}))-\sum_{i=1}^{n} p(x_{i}) \log (q(x_{i})) =-H(p)+H(p,q)$
>   - ML中评估label与predicts的差距，KL散度就够了，因为H(p)为常数
  

## Multi-class classification
- softmax: 强化最大值。两种推导
  - gaussian distribution（概率论角度）
  - maximum entropy(信息论角度)


## Limitation
- boundary 为 linear，不能实现异或
- feature transformation -> not always easy
- cascading logistic regression -> deep learning

# L6 Brief Introduction of Deep Learning
model: Neural Network with parameter -> function, network structure -> function set
- Neural Network 只是转化了问题：Feature Engineering -> design network structure。
- DL适合语言和影像，用于NLP相对改进较少（因为语言设计rule较简单）

# L7 Backpropagation
- backpropagation: 一种比较有效的Gradient descent演算方法。
- Calculous: Chain rule
- 2 steps:
  - forward pass：向前算出输出
  - backward pass：向后分配误差（链式）

# L8 “Hello world" of DL in Keras
- 搭积木
- mini batch
  - trade-off between speed and stability,
  - batch不能太大：并行运算无法支持太大、容易卡住

# L9 Tips for Training DNN
Recipy of Deep Learning
## Do not always blame overfitting
  - train good and test bad is overfitting
  - 都不好可能是没有train好
  - overfit才dropout，否则更差
## 改进神经网络
### 新激活函数
sigmoid问题
- gradient vanishing
- output not zero-centered
- 幂运算耗时

改进
- Hyperbolic Tangent(tanh)：zero-centered
- Rectified Linear Unit(ReLU)
    - faster：threshold max(0,x)
    - biological reason（左右脑），稀疏后模型更好挖掘特征以拟合
    - infinite sigmoid with different biases的结果
    - 解决vanishing gradient
    - 不是zero-centered，存在Dead ReLU Problem
- Leaky ReLU：解决Dead ReLU
- Parametric ReLU
- Exponential Linear Units(ELU)：解决Dead ReLU + zero-centered，计算量稍大
- Maxout
  - Learnable activation function
  - 不同input，不同structure
  - ReLU is a special class of Maxout


### Adaptive Learning Rate
- Adagrad
- RMSProp
- Momentum
- Adam = RMSProp + Momentum

### Early Stopping
validation set error最小时停下。

### Regularization
- L2：乘。Weight decey
- L1：减。

### Dropout
- train -> dropout, test -> no dropdout
- dropout rate p%,则test时weitht乘1-p%
- dropout是一个终极的ensemble(训练多个模型取平均，以中和误差)

# L10 CNN
- why  CNN for Image
  - some patterns are small -> convolution
  - same patterns appear in different regions -> convolutionjj
  - subsampling will not change the object -> Max Pooling
- CNN架构
  - (Convolution -> Max Pooling) * 多次 -> Flatten -> Fully Connected Feedforward network
- 简化DNN，省去不必要的连接、weight sharing
- 图形变换：在CNN前面加一个network，对特定部分缩放、旋转。

# L11 Why Deep?
## Deep -> Modularization
共用前面的Neural，简化了模型（eg.先分男女、发长，再组合），所以瘦高的DNN比矮胖的网络效果更好
- Universality Theorem: shallow network(one hidden layer) can represent any function
- However, Deep is more effective, using less data
- 机器自动学习模组化（？）
- Analogy
  - 两层logic gates can represent any boolean function，但通过多层次共用可以减少逻辑门
  - 剪窗花

## Modularization - Speech
- 语音识别概念：Phoneme, Tri-phone, state
- 过程：acoustic feature(piece of wave phone) -> state -> phone, phoneme -> words -> 同音字等
- 传统方法：HMM-GMM
  - 参数太多
  - state is independent（不符合发音方式）
- DNN：All states use same DNN, all phonemes share same detectors

## End-to-end learning
- Speech Recognition: 步骤大都是hand-crafted，而使用DNN可以learn all functions
  - Google一篇paper：input target main声音讯号，output文字，打平了Fourier transform
- Image Recognition

# L12