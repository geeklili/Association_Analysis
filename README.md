# 关联分析

## 例子
#### 根据水果商店的每个人的购买清单, 进行关联分析，类似于啤酒纸尿裤的挖掘，分别计算水果之间的支持度，置信度，提升度

* 其中，I表示总事务集。num()表示求事务集里特定项集出现的次数;

* 比如，num(I)表示总事务集的个数;

* num(X∪Y)表示含有{X,Y}的事务集的个数（个数也叫次数）。


* 1.支持度（Support）
    * 支持度表示项集{X,Y}在总项集里出现的概率。公式为：
    * Support(X→Y) = P(X,Y) / P(I) = P(X∪Y) / P(I) = num(XUY) / num(I)

* 2.置信度 （Confidence）
   * 置信度表示在先决条件X发生的情况下，由关联规则”X→Y“推出Y的概率。即在含有X的项集中，含有Y的可能性，公式为：
   * Confidence(X→Y) = P(Y|X)  = P(X,Y) / P(X) = P(XUY) / P(X)

* 3.提升度（Lift）
    * 提升度表示含有X的条件下，同时含有Y的概率，与不含X的条件下却含Y的概率之比。
    * Lift(X→Y) = P(Y|X) / P(Y)

### association_analysis.py
### 表头的水果顺序
```
{'桃子': 0, '桔子': 1, '榴莲': 2, '甘蔗': 3, '芒果': 4, '苹果': 5, '草莓': 6, '香蕉': 7}
```

### 每行代表一个人的消费记录，1代表购买了该水果，0代表未购买该水果
```
    [[0 1 0 0 0 1 0 1]
     [0 1 0 1 1 0 0 0]
     [0 0 0 0 1 0 0 1]
     [0 0 1 0 1 1 1 0]
     [1 1 0 0 0 1 0 1]
     [0 1 1 0 0 1 0 1]]
```

### 原始数据类型可以是：
```
苹果,桔子,香蕉
桔子,芒果,甘蔗
香蕉,芒果
苹果,芒果,榴莲,草莓
桔子,桃子,苹果,香蕉
桔子,香蕉,苹果,榴莲
```

### association_analysis_pro.py 文件是升级版关联分析的计算，适合大数据
文件准备的类型为列表：
```
['html', 'redirect', 'seo', 'google-search']
['prolog', 'swi-prolog']
['javascript', 'math', 'colors', 'html5-canvas']
['c#', 'asp.net-mvc']
```