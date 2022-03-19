===============
XGBoost调优指南
===============

:Date:   2019-08-02T21:05:56+08:00

原文地址：\ `Complete Guide to Parameter Tuning in
XGBoost <https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/>`__
by Aarshay Jain

.. _1-简介:

**1. 简介**
===========

如果你的预测模型表现得有些不尽如人意，那就用XGBoost吧。XGBoost算法现在已经成为很多数据工程师的重要武器。它是一种十分精致的算法，\ **可以处理各种不规则的数据。**

构造一个使用XGBoost的模型十分简单。但是，提高这个模型的表现就有些困难(至少我觉得十分纠结)。这个算法使用了好几个参数。所以为了提高模型的表现，参数的调整十分必要。\ **在解决实际问题的时候，有些问题是很难回答的——你需要调整哪些参数？这些参数要调到什么值，才能达到理想的输出？**

这篇文章最适合刚刚接触XGBoost的人阅读。在这篇文章中，我们会学到参数调优的技巧，以及XGboost相关的一些有用的知识。以及，我们会用Python在一个数据集上实践一下这个算法。

.. _2-你需要知道的:

**2. 你需要知道的**
===================

XGBoost(eXtreme Gradient Boosting)是Gradient
Boosting算法的一个优化的版本。在前文章中，基于Python的Gradient
Boosting算法参数调整完全指南，里面已经涵盖了Gradient
Boosting算法的很多细节了。我强烈建议大家在读本篇文章之前，把那篇文章好好读一遍。它会帮助你对Boosting算法有一个宏观的理解，同时也会对GBM的参数调整有更好的体会。

.. _3-内容列表:

**3. 内容列表**
===============

| 1、\ **XGBoost的优势**
| 2、\ **理解XGBoost的参数**
| 3、\ **调参示例**

.. _4-xgboost的优势:

**4. XGBoost的优势**
====================

XGBoost算法可以给预测模型带来能力的提升。当我对它的表现有更多了解的时候，当我对它的高准确率背后的原理有更多了解的时候，我发现它具有很多优势：

.. _41-正则化:

**4.1 正则化**
--------------

-  | 标准GBM的实现没有像XGBoost这样的正则化步骤。\ **``正则化对减少过拟合也是有帮助的。``**
   | |image1|

-  实际上，XGBoost以\ **``正则化提升(regularized boosting)``**\ 技术而闻名。

.. _42-并行处理:

**4.2 并行处理**
----------------

-  XGBoost可以实现\ ``并行处理，相比GBM有了速度的飞跃``\ 。

-  | 不过，众所周知，\ ``Boosting算法是顺序处理的，它怎么可能并行呢?``\ 每一课树的构造都依赖于前一棵树，那具体是什么让我们能用\ ``多核处理器``\ 去构造一个树呢？我希望你理解了这句话的意思。如果你希望了解更多，点击这个\ `链接 <http://zhanpengfang.github.io/418home.html>`__\ 。(**构造决策树的结构时，样本分割点位置，可以使用并行计算**)
   | |image2|

-  XGBoost 也支持\ ``Hadoop``\ 实现。

.. _43-高度的灵活性:

**4.3 高度的灵活性**
--------------------

-  XGBoost 允许用户定义\ **``自定义优化目标函数和评价标准``**

-  它对模型增加了一个\ ``全新的维度``\ ，\ ``所以我们的处理不会受到任何限制``\ 。

.. _44-缺失值处理:

**4.4 缺失值处理**
------------------

-  **XGBoost\ ``内置处理缺失值的规则``**\ 。

-  用户需要提供一个和其它样本不同的值，然后把它作为一个参数传进去，以此来作为缺失值的取值。\ **XGBoost在不同节点遇到缺失值时采用不同的处理方法，并且会学习未来遇到缺失值时的处理方法。**

.. _45-剪枝:

**4.5 剪枝**
------------

-  当分裂时遇到一个负损失时，GBM会停止分裂。因此GBM实际上是一个\ **贪心算法**\ 。

-  XGBoost会\ **一直分裂到指定的最大深度(``max_depth``)**\ ，然后\ **回过头来剪枝**\ 。如果\ **某个节点之后不再有正值，它会去除这个分裂**\ 。

-  这种做法的优点，\ **当一个负损失（如-2）后面有个正损失（如+10）的时候，就显现出来了。GBM会在-2处停下来，因为它遇到了一个负值。但是XGBoost会继续分裂，然后发现这两个分裂综合起来会得到+8，因此会保留这两个分裂**\ 。

.. _46-内置交叉验证:

**4.6 内置交叉验证**
--------------------

-  XGBoost允许在\ **每一轮\ ``boosting``\ 迭代中使用\ ``交叉验证``**\ 。因此，\ **可以方便地获得最优\ ``boosting``\ 迭代次数**\ 。

-  而GBM使用\ ``网格搜索``\ ，\ **只能检测有限个值**\ 。

.. _47在已有的模型基础上继续:

**4.7、在已有的模型基础上继续**
-------------------------------

-  XGBoost可以\ ``在上一轮的结果上继续训练``\ 。这个特性在某些特定的应用上是一个巨大的优势。

-  ``sklearn``\ 中的\ ``GBM``\ 的实现也有这个功能，两种算法在这一点上是一致的。

.. _5-xgboost的参数重要:

**5. XGBoost的参数**\ (重要)
============================

XGBoost的作者把所有的参数分成了三类：

1. **通用参数：宏观函数控制。**

2. **Booster参数：控制每一步的\ ``booster(tree/regression)``\ 。**

3. **学习目标参数：控制训练目标的表现。**

在这里我会类比GBM来讲解

.. _51-通用参数:

**5.1 通用参数**
----------------

这些参数用来控制XGBoost的宏观功能。

.. _1booster默认gbtree:

**1、booster[默认gbtree]**
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  | 选择每次迭代的模型，有两种选择：
   | ``gbtree``\ ：基于树的模型
   | ``gbliner``\ ：线性模型

.. _2silent默认0:

**2、silent[默认0]**
~~~~~~~~~~~~~~~~~~~~

-  当这个参数值为\ ``1``\ 时，\ ``静默模式开启，不会输出任何信息``\ 。

-  一般这个参数就\ ``保持默认的0，因为这样能帮我们更好地理解模型``\ 。

.. _3nthread默认值为最大可能的线程数:

**3、nthread[默认值为\ ``最大可能的线程数``]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  这个参数用来进行\ ``多线程控制``\ ，应当\ ``输入系统的核数``\ 。

-  如果你希望\ ``使用CPU全部的核``\ ，那就不要输入这个参数，\ ``算法会自动检测它``\ 。

还有两个参数，XGBoost会自动设置，目前你不用管它。接下来咱们一起看\ ``booster``\ 参数。

.. _52-booster参数:

**5.2 booster参数**
-------------------

尽管有两种booster可供选择，我这里只介绍\ **``tree booster``**\ ，因为它的表现远远胜过\ **``linear booster``**\ ，所以\ **``linear booster``**\ 很少用到。

.. _1eta默认03:

**1、eta[默认0.3]**
~~~~~~~~~~~~~~~~~~~

-  和\ ``GBM``\ 中的\ ``learning rate``\ 参数类似。

-  通过\ ``减少每一步的权重``\ ，可以\ ``提高模型的鲁棒性``\ 。

-  典型值为\ ``0.01-0.2``\ 。

.. _2minchildweight默认1:

**2、min_child_weight[默认1]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  决定\ ``最小叶子节点样本权重和``\ 。

-  和GBM的
   ``min_child_leaf``\ 参数类似，但不完全一样。XGBoost的这个参数是\ **``最小样本权重的和``**\ ，而GBM参数是\ ``最小叶子节点样本数``\ 。

-  这个参数\ ``用于避免过拟合``\ 。当它的值较大时，可以\ ``避免模型学习到局部的特殊样本``\ 。

-  但是如果这个值过高，会\ ``导致欠拟合``\ 。这个参数需要使用\ ``CV``\ 来调整。

.. _3maxdepth默认6:

**3、max_depth[默认6]**
~~~~~~~~~~~~~~~~~~~~~~~

-  和GBM中的参数相同，这个值为\ ``树的最大深度``\ 。

-  这个值也是用来\ ``避免过拟合的``\ 。\ ``max_depth``\ 越大，模型会学到更具体更局部的样本。

-  需要使用\ ``CV``\ 函数来进行调优。

-  典型值：\ ``3-10``

.. _4maxleafnodes:

**4、max_leaf_nodes**
~~~~~~~~~~~~~~~~~~~~~

-  树上\ ``最大的节点``\ 或\ ``叶子的数量``\ 。

-  可以替代\ ``max_depth``\ 的作用。因为如果生成的是\ ``二叉树``\ ，一个深度为\ ``n``\ 的树最多生成\ ``n平方``\ 个叶子。

-  如果定义了这个参数，\ ``GBM``\ 会忽略\ ``max_depth``\ 参数。

.. _5gamma默认0:

**5、gamma[默认0]**
~~~~~~~~~~~~~~~~~~~

-  在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。\ ``Gamma``\ 指定了\ ``节点分裂所需的最小损失函数下降值``\ 。

-  这个参数的值越大，算法越保守。这个参数的值和\ ``损失函数息息相关``\ ，所以是需要调整的。

.. _6maxdeltastep默认0:

**6、max_delta_step[默认0]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  这参数\ ``限制每棵树权重改变的最大步长``\ 。如果这个参数的值为\ ``0``\ ，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。

-  通常，这个参数不需要设置。但是\ ``当各类别的样本十分不平衡时``\ ，它对逻辑回归是很有帮助的。

-  这个参数一般用不到，但是你可以挖掘出来它更多的用处。

.. _7subsample默认1:

**7、subsample[默认1]**
~~~~~~~~~~~~~~~~~~~~~~~

-  和GBM中的\ ``subsample``\ 参数一模一样。这个参数控制对于每棵树，\ ``随机采样的比例。``

-  ``减小这个参数的值，算法会更加保守，避免过拟合``\ 。但是，如果这个值设置得过小，它可能会导致\ ``欠拟合``\ 。

-  典型值：\ ``0.5-1``

.. _8colsamplebytree默认1:

**8、colsample_bytree[默认1]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  和GBM里面的\ ``max_features``\ 参数类似。用来控制\ ``每棵随机采样的列数的占比(每一列是一个特征)。``

-  典型值：\ ``0.5-1``

.. _9colsamplebylevel默认1:

**9、colsample_bylevel[默认1]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  用来\ ``控制树的每一级的每一次分裂，对列数的采样的占比``\ 。

-  我个人一般不太用这个参数，因为\ ``subsample``\ 参数和\ ``colsample_bytree``\ 参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处。

.. _10lambda默认1:

**10、lambda[默认1]**
~~~~~~~~~~~~~~~~~~~~~

-  权重的\ ``L2``\ 正则化项。(和\ ``Ridge regression``\ 类似)。

-  这个参数是用来控制\ ``XGBoost``\ 的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在\ ``减少过拟合上还是可以挖掘出更多用处的。``

.. _11alpha默认1:

**11、alpha[默认1]**
~~~~~~~~~~~~~~~~~~~~

-  权重的\ ``L1``\ 正则化项。(和\ ``Lasso regression``\ 类似)。

-  可以应用在很\ ``高维度的情况下，使得算法的速度更快``\ 。

.. _12scaleposweight默认1:

**12、scale_pos_weight[默认1]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。``

.. _53-学习目标参数:

**5.3 学习目标参数**
--------------------

这个参数用来控制理想的优化目标和每一步结果的度量方法。

.. _1objective默认reglinear:

**1、objective[默认reg:linear]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  这个参数定义需要被\ ``最小化的损失函数``\ 。最常用的值有：

   -  ``binary:logistic``
      二分类的逻辑回归，返回预测的\ ``概率``\ (不是类别)。

   -  | ``multi:softmax``
        使用softmax的多分类器，返回预测的\ ``类别``\ (不是概率)。
      | 在这种情况下，你还需要多设一个参数：\ ``num_class``\ (类别数目)。

   -  ``multi:softprob``
      和\ ``multi:softmax``\ 参数一样，但是返回的是\ ``每个数据属于各个类别的概率``\ 。

.. _2evalmetric默认值取决于objective参数的取值:

**2、eval_metric[默认值取决于objective参数的取值]**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  对于有效数据的度量方法。

-  对于\ ``回归问题``\ ，默认值是\ ``rmse``\ ，对于\ ``分类问题``\ ，默认值是\ ``error``\ 。

-  典型值有：

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592681-b68ad055-a022-412a-9541-2393a8de61dc.png#align=left&display=inline&height=314&originHeight=314&originWidth=341&size=0&status=done&width=341
   :alt: 

.. _3seed默认0:

**3、seed(默认0)**
~~~~~~~~~~~~~~~~~~

-  随机数的种子

-  设置它可以复现随机数据的结果，也可以用于调整参数

如果你之前用的是\ ``scikit-learn``,你可能不太熟悉这些参数。但是有个好消息，\ ``python``\ 的\ ``XGBoost``\ 模块有一个\ ``sklearn``\ 包，\ ``XGBClassifier``\ 。这个包中的参数是按\ ``sklearn``\ 风格命名的。会改变的函数名是：

| 1、\ **``eta -> learning_rate``**
| 2、\ **``lambda -> reg_lambda``**
| 3、\ **``alpha -> reg_alpha``**

| 你肯定在疑惑为啥咱们没有介绍和\ ``GBM``\ 中的\ ``n_estimators``\ 类似的参数。\ ``XGBClassifier``\ 中确实有一个类似的参数，但是，是在标准\ ``XGBoost``\ 实现中调用拟合函数时，把它作为\ ``num_boosting_rounds``\ 参数传入。
| ``XGBoost Guide``\ 的一些部分是我强烈推荐大家阅读的，通过它可以对代码和参数有一个更好的了解：

| `XGBoost Parameters (official
  guide) <http://xgboost.readthedocs.org/en/latest/parameter.html#general-parameters>`__
| `XGBoost Demo Codes (xgboost GitHub
  repository) <https://github.com/dmlc/xgboost/tree/master/demo/guide-python>`__
| `Python API Reference (official
  guide) <http://xgboost.readthedocs.org/en/latest/python/python_api.html>`__

例子1
-----

.. code:: python

   import xgboost as xgb
   import numpy as np

   # 1、xgBoost的基本使用
   # 2、自定义损失函数的梯度和二阶导
   # 3、binary:logistic/logitraw



   # 自定义损失函数
   # 定义f: theta * x
   def log_reg(y_hat, y):
       p = 1.0 / (1.0 + np.exp(-y_hat))
       g = p - y.get_label()   #  目标函数的一阶导数
       h = p * (1.0-p)         #  目标函数的二阶导数
       return g, h

   # 稀疏数据的存储方案
   # 126个特征，1代表毒蘑菇，0好蘑菇

   # 自定义错误率
   def error_rate(y_hat, y):
       return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


   if __name__ == "__main__":
       # 读取数据，xgb.DMatrix格式数据
       data_train = xgb.DMatrix('./12.agaricus_train.txt')
       data_test = xgb.DMatrix('./12.agaricus_test.txt')

       # 设置参数
       param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw'}  # logitraw
       #  param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}

       watchlist = [(data_test, 'eval'), (data_train, 'train')]

       # GBM中的 n_estimators 类似的参数
       n_round = 3

       # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
       bst = xgb.train(param, \
                       data_train, \
                       num_boost_round=n_round, \
                       evals=watchlist, \
                       obj=log_reg, \
                       feval=error_rate)

       # 计算错误率
       y_hat = bst.predict(data_test)
       y = data_test.get_label()
       print(y_hat)
       print(y)
       error = sum(y != (y_hat > 0))
       error_rate = float(error) / len(y_hat)
       print('样本总数:', len(y_hat))
       print('错误数目:{0}'.format(error))
       print('错误率:{0:.2%}'.format(error_rate))

运行结果：

.. code:: 

   [09:32:13] 6513x126 matrix with 143286 entries loaded from ./12.agaricus_train.txt
   [09:32:13] 1611x126 matrix with 35442 entries loaded from ./12.agaricus_test.txt
   [0]     eval-auc:0.960373       train-auc:0.958228      eval-error:0.042831     train-error:0.046522
   [1]     eval-auc:0.97993        train-auc:0.981413      eval-error:0.021726     train-error:0.022263
   [2]     eval-auc:0.998518       train-auc:0.99707       eval-error:0.018001     train-error:0.0152
   [-1.70713    1.7054877 -1.70713   ...  3.1556199 -3.7006462  3.1556199]
   [0. 1. 0. ... 1. 0. 1.]
   样本总数: 1611
   错误数目:10
   错误率:0.62%

例子2
-----

.. code:: python

   import xgboost as xgb
   import numpy as np
   from sklearn.model_selection import train_test_split   # cross_validation
   from sklearn.linear_model import LogisticRegression
   from sklearn.preprocessing import StandardScaler


   def show_accuracy(a, b, tip):
       acc = a.ravel() == b.ravel()
       print(acc)
       print(tip + '正确率:', float(acc.sum()) / a.size)


   if __name__ == "__main__":
       data = np.loadtxt('./12.wine.data', dtype=float, delimiter=',')
       y, x = np.split(data, (1,), axis=1)
       x = StandardScaler().fit_transform(x)
       x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

       # Logistic回归
       lr = LogisticRegression(penalty='l2')
       lr.fit(x_train, y_train.ravel())
       y_hat = lr.predict(x_test)
       show_accuracy(y_hat, y_test, 'Logistic回归 ')

       # XGBoost
       y_train[y_train == 3] = 0
       y_test[y_test == 3] = 0
       data_train = xgb.DMatrix(x_train, label=y_train)
       data_test = xgb.DMatrix(x_test, label=y_test)
       watch_list = [(data_test, 'eval'), (data_train, 'train')]
       param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
       bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
       y_hat = bst.predict(data_test)
       show_accuracy(y_hat, y_test, 'XGBoost ')

运行结果：

.. code:: 

   [ True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True False  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True]
   Logistic回归 正确率: 0.9887640449438202
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 0 pruned nodes, max_depth=3
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 0 pruned nodes, max_depth=3
   [0]     eval-merror:0.011236    train-merror:0
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 4 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 0 pruned nodes, max_depth=3
   [1]     eval-merror:0   train-merror:0
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 4 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 0 pruned nodes, max_depth=3
   [2]     eval-merror:0.011236    train-merror:0
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 4 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 4 extra nodes, 0 pruned nodes, max_depth=2
   [09:36:10] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 0 pruned nodes, max_depth=3
   [3]     eval-merror:0.011236    train-merror:0
   [ True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True False
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True  True  True  True  True  True  True  True
     True  True  True  True  True]
   XGBoost 正确率: 0.9887640449438202

.. _6-调参示例:

**6. 调参示例**
===============

| 我们从Data Hackathon 3.x AV版的hackathon中获得数据集，和\ `GBM
  介绍文章 <http://blog.csdn.net/han_xiaoyang/article/details/52663170>`__\ 中是一样的。更多的细节可以参考\ `competition
  page <http://datahack.analyticsvidhya.com/contest/data-hackathon-3x>`__
| 数据集可以从\ `这里 <http://www.analyticsvidhya.com/wp-content/uploads/2016/02/Dataset.rar>`__\ 下载。我已经对这些数据进行了一些处理：

-  ``City``\ 变量，因为类别太多，所以删掉了一些类别。

-  ``DOB``\ 变量换算成年龄，并删除了一些数据。

-  增加了 ``EMI_Loan_Submitted_Missing``
   变量。如果\ ``EMI_Loan_Submitted``\ 变量的数据缺失，则这个参数的值为1。否则为0。删除了原先的\ ``EMI_Loan_Submitted``\ 变量。

-  ``EmployerName``\ 变量，因为类别太多，所以删掉了一些类别。

-  因为\ ``Existing_EMI``\ 变量只有111个值缺失，所以缺失值补充为中位数0。

-  增加了 ``Interest_Rate_Missing``
   变量。如果\ ``Interest_Rate``\ 变量的数据缺失，则这个参数的值为1。否则为0。删除了原先的\ ``Interest_Rate``\ 变量。

-  删除了\ ``Lead_Creation_Date``\ ，从直觉上这个特征就对最终结果没什么帮助。

-  ``Loan_Amount_Applied, Loan_Tenure_Applied``
   两个变量的缺项用中位数补足。

-  增加了 ``Loan_Amount_Submitted_Missing``
   变量。如果\ ``Loan_Amount_Submitted``\ 变量的数据缺失，则这个参数的值为1。否则为0。删除了原先的\ ``Loan_Amount_Submitted``\ 变量。

-  增加了 ``Loan_Tenure_Submitted_Missing`` 变量。如果
   ``Loan_Tenure_Submitted``
   变量的数据缺失，则这个参数的值为1。否则为0。删除了原先的
   ``Loan_Tenure_Submitted`` 变量。

-  删除了\ ``LoggedIn``, ``Salary_Account`` 两个变量

-  增加了 ``Processing_Fee_Missing`` 变量。如果 ``Processing_Fee``
   变量的数据缺失，则这个参数的值为1。否则为0。删除了原先的
   ``Processing_Fee`` 变量。

-  ``Source``\ 前两位不变，其它分成不同的类别。

-  进行了离散化和独热编码(一位有效编码)。

如果你有原始数据，可以从资源库里面下载\ ``data_preparation``\ 的\ ``Ipython notebook``
文件，然后自己过一遍这些步骤。

首先，\ ``import``\ 必要的库，然后加载数据。

.. code:: python

   #Import libraries:
   import pandas as pd
   import numpy as np
   import xgboost as xgb
   from xgboost.sklearn import XGBClassifier
   from sklearn import cross_validation, metrics   #Additional     scklearn functions
   from sklearn.grid_search import GridSearchCV   #Perforing grid search

   import matplotlib.pylab as plt
   %matplotlib inline
   from matplotlib.pylab import rcParams
   rcParams['figure.figsize'] = 12, 4

   train = pd.read_csv('train_modified.csv')
   target = 'Disbursed'
   IDcol = 'ID'

注意我\ ``import``\ 了两种\ ``XGBoost``\ ：

-  ``xgb`` -
   直接引用\ ``xgboost``\ 。接下来会用到其中的“\ ``cv``\ ”函数。

-  ``XGBClassifier`` -
   是\ ``xgboost``\ 的\ ``sklearn``\ 包。这个包允许我们像\ ``GBM``\ 一样使用\ ``Grid Search``
   和\ ``并行处理``\ 。

在向下进行之前，我们先定义一个函数，它可以帮助我们建立\ ``XGBoost models``
并进行交叉验证。好消息是你可以直接用下面的函数，以后再自己的\ ``models``\ 中也可以使用它。

.. code:: python

   def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
       if useTrainCV:
           xgb_param = alg.get_xgb_params()
           
           xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
           
           cvresult = xgb.cv(xgb_param, \
                             xgtrain, \
                             num_boost_round=alg.get_params()['n_estimators'], \
                             nfold=cv_folds,\
                             metrics='auc', \
                             early_stopping_rounds=early_stopping_rounds)
           
           alg.set_params(n_estimators=cvresult.shape[0])

       #Fit the algorithm on the data
       alg.fit(dtrain[predictors], \
               dtrain['Disbursed'], \
               eval_metric='auc')

       #Predict training set:
       dtrain_predictions = alg.predict(dtrain[predictors])
       dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

       #Print model report:
       print ("Model Report")
       print ("Accuracy : {0:.4f}".format(metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)))
       print ("AUC Score (Train): {0:.2f}".format(metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)))

       #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
       #feat_imp.plot(kind='bar', title='Feature Importances')
       #plt.ylabel('Feature Importance Score')

这个函数和\ ``GBM``\ 中使用的有些许不同。不过本文章的重点是讲解重要的概念，而不是写代码。如果哪里有不理解的地方，请在下面评论，不要有压力。注意\ ``xgboost``\ 的\ ``sklearn``\ 包没有\ ``feature_importance``\ 这个量度，但是\ ``get_fscore()``\ 函数有相同的功能。

.. _61-参数调优的一般方法:

**6.1 参数调优的一般方法**
--------------------------

我们会使用和\ ``GBM``\ 中相似的方法。需要进行如下步骤：

1. 选择较高的\ **学习速率(``learning rate``)**\ 。一般情况下，学习速率的值为\ ``0.1``\ 。但是，对于不同的问题，理想的学习速率有时候会在\ ``0.05到0.3``\ 之间波动。选择\ **对应于此学习速率的理想决策树数量**\ 。\ ``XGBoost``\ 有一个很有用的函数“\ ``cv``\ ”，这个函数可以在\ ``每一次迭代中使用交叉验证，并返回理想的决策树数量``\ 。

2. 对于给定的\ ``学习速率和决策树数量``\ ，进行\ **决策树特定参数调优**\ (``max_depth``,
   ``min_child_weight``, ``gamma``, ``subsample``,
   ``colsample_bytree``)。在确定一棵树的过程中，我们可以选择不同的参数，待会儿我会举例说明。

3. ``xgboost``\ 的\ **正则化参数**\ 的调优。(``lambda``,
   ``alpha``)。这些参数可以\ ``降低模型的复杂度``\ ，从而\ ``提高模型的表现``\ 。

4. 降低学习速率，确定理想参数。

咱们一起详细地一步步进行这些操作。

.. _第一步确定learning-rate和treebased-参数调优的估计器数目:

**第一步：确定\ ``learning rate``\ 和\ ``tree_based`` 参数调优的估计器数目**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了确定\ ``boosting``\ 参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：

1、\ ``max_depth`` = 5
:这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。

2、\ ``min_child_weight`` =
1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小。

3、\ ``gamma`` = 0:
起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。

4、\ ``subsample, colsample_bytree`` = 0.8:
这个是最常见的初始值了。典型值的范围在0.5-0.9之间。

| 5、\ ``scale_pos_weight`` = 1: 这个值是因为类别十分不平衡。
| 注意哦，上面这些参数的值只是一个初始的估计值，后继需要调优。这里把学习速率就设成默认的\ ``0.1``\ 。然后用\ ``xgboost``\ 中的\ ``cv``\ 函数来确定最佳的决策树数量。前文中的函数可以完成这个工作。

.. code:: 

   #Choose all predictors except target & IDcols
   predictors = [x for x in train.columns if x not in [target,IDcol]]

   xgb1 = XGBClassifier(learning_rate =0.1,\
                        n_estimators=1000,\
                        max_depth=5,\
                        min_child_weight=1,\
                        gamma=0,\
                        subsample=0.8,\
                        colsample_bytree=0.8,\
                        objective= 'binary:logistic',\
                        nthread=4,\
                        scale_pos_weight=1,\
                        seed=27)

   modelfit(xgb1, train, predictors)

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592948-4acb645c-2b27-4918-a233-086611670327.png#align=left&display=inline&height=999&originHeight=999&originWidth=1240&size=0&status=done&width=1240
   :alt: 

从输出结果可以看出，在学习速率为0.1时，理想的决策树数目是140。这个数字对你而言可能比较高，当然这也取决于你的系统的性能。

   注意：在AUC(test)这里你可以看到测试集的AUC值。但是如果你在自己的系统上运行这些命令，并不会出现这个值。因为数据并不公开。这里提供的值仅供参考。生成这个值的代码部分已经被删掉了。

.. _第二步-maxdepth-和-minweight-参数调优:

**第二步： max_depth 和 min_weight 参数调优**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| 我们先对这两个参数调优，是因为它们对最终结果有很大的影响。首先，我们先大范围地粗调参数，然后再小范围地微调。
| 注意：在这一节我会进行高负荷的栅格搜索(grid
  search)，这个过程大约需要15-30分钟甚至更久，具体取决于你系统的性能。你也可以根据自己系统的性能选择不同的值。

.. code:: python

   param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
   }
   gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=140, max_depth=5,
   min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
    objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27), 
    param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch1.fit(train[predictors],train[target])
   gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592771-cb7fb77f-c35b-4061-b7a7-b5287afc81d4.png#align=left&display=inline&height=448&originHeight=448&originWidth=1240&size=0&status=done&width=1240
   :alt: 

至此，我们对于数值进行了较大跨度的12中不同的排列组合，可以看出理想的max_depth值为5，理想的min_child_weight值为5。在这个值附近我们可以再进一步调整，来找出理想值。我们把上下范围各拓展1，因为之前我们进行组合的时候，参数调整的步长是2。

.. code:: python

   param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weight':[4,5,6]
   }
   gsearch2 = GridSearchCV(estimator = XGBClassifier(     learning_rate=0.1, n_estimators=140, max_depth=5,
    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
    param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
   gsearch2.fit(train[predictors],train[target])
   gsearch2.grid_scores_, gsearch2.best_params_,     gsearch2.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592632-e93ff500-3523-4c30-9841-5d83164ff635.png#align=left&display=inline&height=346&originHeight=346&originWidth=1240&size=0&status=done&width=1240
   :alt: 

至此，我们得到max_depth的理想取值为4，min_child_weight的理想取值为6。同时，我们还能看到cv的得分有了小小一点提高。需要注意的一点是，随着模型表现的提升，进一步提升的难度是指数级上升的，尤其是你的表现已经接近完美的时候。当然啦，你会发现，虽然min_child_weight的理想取值是6，但是我们还没尝试过大于6的取值。像下面这样，就可以尝试其它值。

.. code:: python

   param_test2b = {
    'min_child_weight':[6,8,10,12]
    }
   gsearch2b = GridSearchCV(estimator = XGBClassifier(     learning_rate=0.1, n_estimators=140, max_depth=4,
    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch2b.fit(train[predictors],train[target])

   modelfit(gsearch3.best_estimator_, train, predictors)

   gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592460-325fcd90-efc6-41d7-bba4-44353cbcfd4a.png#align=left&display=inline&height=222&originHeight=222&originWidth=1116&size=0&status=done&width=1116
   :alt: 

我们可以看出，6确确实实是理想的取值了。

**第三步：gamma参数调优**
~~~~~~~~~~~~~~~~~~~~~~~~~

在已经调整好其它参数的基础上，我们可以进行gamma参数的调优了。Gamma参数取值范围可以很大，我这里把取值范围设置为5了。你其实也可以取更精确的gamma值。

.. code:: python

   param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
   }
   gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch3.fit(train[predictors],train[target])
   gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

| 从这里可以看出来，我们在第一步调参时设置的初始gamma值就是比较合适的。也就是说，理想的gamma值为0。在这个过程开始之前，最好重新调整boosting回合，因为参数都有变化。
| |image3|

从这里，可以看出，得分提高了。所以，最终得到的参数是：

.. code:: python

   xgb2 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
   scale_pos_weight=1,
   seed=27)
   modelfit(xgb2, train, predictors)

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097593200-1a7c6731-d726-4e40-bd2b-162ac2da19d0.png#align=left&display=inline&height=994&originHeight=994&originWidth=1240&size=0&status=done&width=1240
   :alt: 

.. _第四步调整subsample-和-colsamplebytree-参数:

第四步：调整subsample 和 colsample_bytree 参数
==============================================

下一步是尝试不同的subsample 和 colsample_bytree
参数。我们分两个阶段来进行这个步骤。这两个步骤都取0.6,0.7,0.8,0.9作为起始值。

.. code:: python

   param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
   }

   gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3, min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch4.fit(train[predictors],train[target])
   gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592835-9c58fe4d-c1c4-4915-b7a5-946836e20424.png#align=left&display=inline&height=536&originHeight=536&originWidth=1240&size=0&status=done&width=1240
   :alt: 

从这里可以看出来，subsample 和 colsample_bytree
参数的理想取值都是0.8。现在，我们以0.05为步长，在这个值附近尝试取值。

.. code:: python

   param_test5 = {
    'subsample':[i/100.0 for i in range(75,90,5)],
    'colsample_bytree':[i/100.0 for i in range(75,90,5)]
   }

   gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch5.fit(train[predictors],train[target])

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592716-f01441b9-23ae-42b2-8864-afea0a14ede3.png#align=left&display=inline&height=319&originHeight=319&originWidth=1240&size=0&status=done&width=1240
   :alt: 

我们得到的理想取值还是原来的值。因此，最终的理想取值是:

-  subsample: 0.8

-  colsample_bytree: 0.8

**第五步：正则化参数调优**
--------------------------

下一步是应用正则化来降低过拟合。由于gamma函数提供了一种更加有效地降低过拟合的方法，大部分人很少会用到这个参数。但是我们在这里也可以尝试用一下这个参数。我会在这里调整’reg_alpha’参数，然后’reg_lambda’参数留给你来完成。

.. code:: python

   param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
   }
   gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch6.fit(train[predictors],train[target])
   gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592554-a17d3f4d-6098-46c2-8143-eabf9b43c105.png#align=left&display=inline&height=250&originHeight=250&originWidth=1038&size=0&status=done&width=1038
   :alt: 

我们可以看到，相比之前的结果，CV的得分甚至还降低了。但是我们之前使用的取值是十分粗糙的，我们在这里选取一个比较靠近理想值(0.01)的取值，来看看是否有更好的表现。

.. code:: python

   param_test7 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
   }
   gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

   gsearch7.fit(train[predictors],train[target])
   gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592569-5651aab7-21ac-4859-9052-c8d0383c07a0.png#align=left&display=inline&height=254&originHeight=254&originWidth=1032&size=0&status=done&width=1032
   :alt: 

可以看到，CV的得分提高了。现在，我们在模型中来使用正则化参数，来看看这个参数的影响。

.. code:: python

   xgb3 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
   modelfit(xgb3, train, predictors)

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592989-9d857ed9-f734-4761-a195-555b73371255.png#align=left&display=inline&height=989&originHeight=989&originWidth=1240&size=0&status=done&width=1240
   :alt: 

然后我们发现性能有了小幅度提高。

**第6步：降低学习速率**
-----------------------

最后，我们使用较低的学习速率，以及使用更多的决策树。我们可以用XGBoost中的CV函数来进行这一步工作。

.. code:: python

   xgb4 = XGBClassifier(
    learning_rate =0.01,
    n_estimators=5000,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
   modelfit(xgb4, train, predictors)

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097593017-0eaed657-a2dd-4316-93c5-fdd510112852.png#align=left&display=inline&height=972&originHeight=972&originWidth=1240&size=0&status=done&width=1240
   :alt: 

至此，你可以看到模型的表现有了大幅提升，调整每个参数带来的影响也更加清楚了。

在文章的末尾，我想分享两个重要的思想：

1、仅仅靠参数的调整和模型的小幅优化，想要让模型的表现有个大幅度提升是不可能的。GBM的最高得分是\ ``0.8487``\ ，XGBoost的最高得分是\ ``0.8494``\ 。确实是有一定的提升，但是没有达到质的飞跃。

2、要想让模型的表现有一个质的飞跃，需要依靠其他的手段，诸如，特征工程(``feature egineering``)
，模型组合(``ensemble of model``),以及堆叠(``stacking``)等。

结束语
======

这篇文章主要讲了如何提升XGBoost模型的表现。首先，我们介绍了相比于GBM，为何XGBoost可以取得这么好的表现。紧接着，我们介绍了每个参数的细节。我们定义了一个可以重复使用的构造模型的函数。

最后，我们讨论了使用XGBoost解决问题的一般方法，在AV Data Hackathon 3.x
problem数据上实践了这些方法。

希望看过这篇文章之后，你能有所收获，下次使用XGBoost解决问题的时候可以更有信心哦~

.. code:: python

   Init signature: XGBClassifier(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,\
    objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,\
    max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, \
   reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, \
   missing=None, **kwargs)

   Docstring:
   Implementation of the scikit-learn API for XGBoost classification.

       Parameters
   ----------
   max_depth : int
       Maximum tree depth for base learners.
   learning_rate : float
       Boosting learning rate (xgb's "eta")
   n_estimators : int
       Number of boosted trees to fit.
   silent : boolean
       Whether to print messages while running boosting.
   objective : string or callable
       Specify the learning task and the corresponding learning objective or
       a custom objective function to be used (see note below).
   booster: string
       Specify which booster to use: gbtree, gblinear or dart.
   nthread : int
       Number of parallel threads used to run xgboost.  (Deprecated, please use n_jobs)
   n_jobs : int
       Number of parallel threads used to run xgboost.  (replaces nthread)
   gamma : float
       Minimum loss reduction required to make a further partition on a leaf node of the tree.
   min_child_weight : int
       Minimum sum of instance weight(hessian) needed in a child.
   max_delta_step : int
       Maximum delta step we allow each tree's weight estimation to be.
   subsample : float
       Subsample ratio of the training instance.
   colsample_bytree : float
       Subsample ratio of columns when constructing each tree.
   colsample_bylevel : float
       Subsample ratio of columns for each split, in each level.
   reg_alpha : float (xgb's alpha)
       L1 regularization term on weights
   reg_lambda : float (xgb's lambda)
       L2 regularization term on weights
   scale_pos_weight : float
       Balancing of positive and negative weights.
   base_score:
       The initial prediction score of all instances, global bias.
   seed : int
       Random number seed.  (Deprecated, please use random_state)
   random_state : int
       Random number seed.  (replaces seed)
   missing : float, optional
       Value in the data which needs to be present as a missing value. If
       None, defaults to np.nan.
   **kwargs : dict, optional
       Keyword arguments for XGBoost Booster object.  Full documentation of parameters can
       be found here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md.
       Attempting to set a parameter via the constructor args and **kwargs dict simultaneously
       will result in a TypeError.
       Note:
           **kwargs is unsupported by Sklearn.  We do not guarantee that parameters passed via
           this argument will interact properly with Sklearn.

   Note
   ----
   A custom objective function can be provided for the ``objective``
   parameter. In this case, it should have the signature
   ``objective(y_true, y_pred) -> grad, hess``:

   y_true: array_like of shape [n_samples]
       The target values
   y_pred: array_like of shape [n_samples]
       The predicted values

   grad: array_like of shape [n_samples]
       The value of the gradient for each sample point.
   hess: array_like of shape [n_samples]
       The value of the second derivative for each sample point
   File:           /opt/anaconda3/lib/python3.5/site-packages/xgboost/sklearn.py
   Type:           type

.. |image1| image:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592738-0070839f-8a41-4bef-99d4-5673fc9f28b9.png#align=left&display=inline&height=569&originHeight=569&originWidth=831&size=0&status=done&width=831
.. |image2| image:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097593170-f291939e-b48c-4cdf-a153-f82ed8ee3f9b.png#align=left&display=inline&height=756&originHeight=756&originWidth=1170&size=0&status=done&width=1170
.. |image3| image:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564097592531-73d31aa8-9d80-44c5-ac6d-0cd5b05bd049.png#align=left&display=inline&height=260&originHeight=260&originWidth=940&size=0&status=done&width=940
