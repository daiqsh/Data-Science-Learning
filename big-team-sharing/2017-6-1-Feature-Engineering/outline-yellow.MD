# 回顾
## 什么是数据集、样本、特征、维度

# ML基本步骤
1. （Task before here）
2. 选择数据(Select Data): 整合数据，将数据规范化成一个数据集，收集起来.
3. 数据预处理（Preprocess Data）: 数据格式化，数据清理，采样等.
4. 数据转换（Transform Data）: 这个阶段做特征工程.
5. 数据建模（Model Data）: 建立模型，评估模型并逐步优化.
6. Tasks after here…)

# 特征工程
## 俗话说
- 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已
- 纵观Kaggle、KDD等国内外大大小小的比赛，每个竞赛的冠军其实并没有用到很高深的算法，大多数都是在特征工程这个环节做出了出色的工作，然后使用一些常见的算法，比如LR，就能得到出色的性能。

##  定义
最大限度地从原始数据中提取特征以供算法和模型使用

## 特征工程在ML中
![ML flow](https://raw.githubusercontent.com/daiqsh/Data-Science-Learning/master/big-team-sharing/2017-6-1-Feature-Engineering/ML_flow.png "ML flow")

## 预处理
1. 去重：去掉重复的数据
2. 过滤：把反常值（极高/极低）用平滑的值代替
3. 标准化：把数据缩放到同样的范围
4. 缺失值处理
5. 离散化
6. 特征构造(特征组合or特征切分："20170522" -> 切成3个特征"年"、"月"、"日")

## 特征提取
### 分类变量特征提取
1. One-Hot encoding

### 文字特征提取
1. 词袋法
2. 停用词过滤
3. 词根还原与词形还原
4. 带TF-IDF权重

### 图片特征提取
1. SIFT和SURF

## 特征选择
1. Filter：Pearson相关系数，Gini-index（基尼指数），IG（信息增益）等
2. Wrapper：用一个基分类器递归地计算不同特征组合
3. Embedded：（集成方法）随机森林、深度神经网络等
