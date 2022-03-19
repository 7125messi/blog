============
GBDT调优详解
============

:Date:   2019-08-02T21:08:50+08:00

原文地址：\ `Complete Guide to Parameter Tuning in Gradient Boosting
(GBM) in
Python <https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/>`__
by Aarshay Jain

.. _1前言:

**1.前言**
==========

如果一直以来你只把\ ``GBM``\ 当作黑匣子，只知调用却不明就里，是时候来打开这个黑匣子一探究竟了！

**不像\ ``bagging``\ 算法只能改善模型高方差（\ ``high variance``\ ）情况**\ ，\ **``Boosting``\ 算法对同时控制偏差（\ ``bias``\ ）和方差（\ ``variance``\ ）都有非常好的效果，而且更加高效**\ 。

如果你需要同时处理模型中的方差和偏差，认真理解这篇文章一定会对你大有帮助，

本文会用\ ``Python``\ 阐明\ ``GBM``\ 算法，更重要的是会介绍如何对\ ``GBM``\ 调参，而恰当的参数往往能令结果大不相同。

.. _2目录:

**2.目录**
==========

1. ``Boosing``\ 是怎么工作的？

2. 理解\ ``GBM``\ 模型中的参数

3. 学会调参（附详例）

.. _3boosting是如何工作的:

**3.Boosting是如何工作的？**
============================

``Boosting``\ 可以将一系列\ **弱学习因子（\ ``weak learners``\ ）相结合来提升总体模型的预测准确度**\ 。\ **在任意时间\ ``t``\ ，根据\ ``t-1``\ 时刻得到的结果我们给当前结果赋予一个权重**\ 。\ **之前正确预测的结果获得\ ``较小权重``\ ，错误分类的结果得到\ ``较大权重``**\ 。回归问题的处理方法也是相似的。

让我们用图像帮助理解：

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-3f8d12774890930c.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

1. 图一： 第一个弱学习因子的预测结果（从左至右）

-  一开始所有的点具有相同的权重（以点的尺寸表示）。

-  分类线正确地分类了两个正极和五个负极的点。

1. 图二： 第二个弱学习因子的预测结果

-  在图一中被正确预测的点有较小的权重（尺寸较小），而\ **``被预测错误的点则有较大的权重``**\ 。

-  这时候模型就会\ **更加注重具有大权重的点的预测结果**\ ，即\ **上一轮分类错误的点**\ ，现在这些点被正确归类了，\ **但其他点中的一些点却归类错误**\ 。

对图3的输出结果的理解也是类似的。这个算法一直如此持续进行直到\ **``所有的学习模型根据它们的预测结果都被赋予了一个权重，这样我们就得到了一个总体上更为准确的预测模型``**\ 。

现在你是否对Boosting更感兴趣了？不妨看看下面这些文章（主要讨论GBM）：

-  `Learn Gradient Boosting Algorithm for better predictions (with codes
   in
   R) <http://www.analyticsvidhya.com/blog/2015/09/complete-guide-boosting-methods/>`__

-  `Quick Introduction to Boosting Algorithms in Machine
   Learning <http://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/>`__

-  `Getting smart with Machine Learning – AdaBoost and Gradient
   Boost <http://www.analyticsvidhya.com/blog/2015/05/boosting-algorithms-simplified/>`__

.. _4gbm参数:

**4.GBM参数**
=============

总的来说GBM的参数可以被归为三类：

1. **``树参数``\ ：**\ 调节模型中每个决定树的性质

2. **``Boosting参数``\ ：**\ 调节模型中\ ``boosting``\ 的操作

3. **``其他模型参数``\ ：**\ 调节\ ``模型总体的各项运作``

.. _1）树参数:

（1）树参数
===========

从树参数开始，首先一个决策树的大致结构是这样的：

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-ad4bb2383701c6d0.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

1. **min\_ samples_split**

-  **定义了树中一个\ ``中间节点所需要用来分裂的最少样本数``\ 。**

-  可以\ **避免过拟合(``over-fitting``)。如果用于分类的样本数太小**\ ，模型可能只适用于用来训练的样本的分类，而用较多的样本数则可以避免这个问题。

-  但是如果设定的值过大，就可能出现\ **欠拟合现象(``under-fitting``)**\ 。因此我们可以用\ **``CV``\ 值（离散系数）考量调节效果。**

1. **min\_ samples_leaf**

-  **定义了树中\ ``叶子节点所需要的最少的样本数``\ 。**

-  同样，它也可以用来\ **``防止过度拟合``\ 。**

-  在\ **不均等分类问题中(``imbalanced class problems``)，一般这个参数需要被设定为\ ``较小``\ 的值，因为大部分少数类别（\ ``minority class``\ ）含有的样本都\ ``比较小``\ 。**

1. **min\_ weight\_ fraction_leaf**

-  和上面\ **``min_ samples_ leaf``**\ 很像，不同的是这里需要的\ ``是一个比例而不是绝对数值``\ ：\ **``叶子节点所需的样本数占总样本数的比值``\ 。**

-  2和3只需要定义一个就行了

1. **max\_ depth**

-  **定义了\ ``树的最大深度``\ 。**

-  它也可以\ **``控制过度拟合，因为分类树越深就越可能过度拟合``\ 。**

-  当然也应该用\ ``CV值检验``\ 。

1. **max\_ leaf\_ nodes**

-  **定义了\ ``决策树里最多能有多少个叶子节点``\ 。**

-  这个属性有可能在上面max\_
   depth里就被定义了。比如深度为\ ``n``\ 的二叉树就有最多\ ``2^n``\ 个终点节点。

-  如果我们定义了\ ``max_ leaf_ nodes``\ ，\ ``GBM``\ 就会忽略前面的\ ``max_depth``\ 。

1. **max\_ features**

-  **决定了用于\ ``分类的特征数``\ ，是\ ``人为随机定义的``\ 。**

-  **根据经验一般选择\ ``总特征数的平方根``\ 就可以工作得很好了，但还是应该用不同的值尝试，最多可以尝试总特征数的\ ``30%-40%``.**

-  **``过多的分类特征可能也会导致过度拟合``\ 。**

在继续介绍其他参数前，我们先看一个简单的\ ``GBM``\ 二分类伪代码：

.. code:: 

   1. 初始分类目标的参数值
   2. 对所有的分类树进行迭代：
       2.1 根据前一轮分类树的结果更新分类目标的权重值（被错误分类的有更高的权重）
       2.2 用训练的子样本建模
       2.3 用所得模型对所有的样本进行预测
       2.4 再次根据分类结果更新权重值
   3. 返回最终结果

.. _2）boosting参数:

（2）Boosting参数
=================

以上步骤是一个极度简化的\ ``GBM``\ 模型，而目前我们所提到的参数会影响
``2.2 用训练的子样本建模``\ 这一步，即建模的过程。现在我们来看看影响\ ``boosting``\ 过程的参数：

1. **``learning_ rate(学习率)``**

-  这个参数决定着\ **每一个决策树对于最终结果（步骤2.4
   ``更新权重值``\ ）的影响**\ 。\ ``GBM``\ 设定了初始的\ ``权重值``\ 之后，\ ``每一次树分类都会更新这个值``\ ，而\ **``learning_ rate控制着每次更新的幅度``\ 。**\ （即\ `8
   提升GBDT <https://www.jianshu.com/p/3a94e853d87a>`__\ ``Shrinkage因子``\ ）

-  **一般来说这个值\ ``不应该设的比较大``**\ ，因为\ **``较小的learning rate使得模型对不同的树更加稳健，就能更好地综合它们的结果。``**

1. **n\_ estimators**

-  **定义了需要使用到的\ ``决策树的数量``\ （步骤2）**

-  虽然GBM即使在有\ **较多决策树时仍然能保持稳健，但还是可能发生过度拟合**\ 。所以也需要针对\ ``learning rate``\ 用\ ``CV``\ 值检验。

1. **subsample**

-  **``训练每个决策树所用到的子样本占总样本的比例``\ ，而对于\ ``子样本的选择是随机的``\ 。**

-  用\ **``稍小于1的值能够使模型更稳健，因为这样减少了方差``**\ 。

-  一把来说用\ ``~0.8``\ 就行了，更好的结果可以用调参获得。

.. _3）其他模型参数:

（3）其他模型参数
=================

好了，现在我们已经介绍了\ **``树参数``\ 和\ ``boosting参数``**\ ，此外还有第三类参数，它们\ **能影响到\ ``模型的总体功能``**

1. **loss**

-  指的是\ **``每一次节点分裂所要最小化的损失函数``\ (loss function)**

-  对于\ **分类和回归模型**\ 可以有不同的值。\ ``一般来说不用更改，用默认值就可以了，除非你对它及它对模型的影响很清楚``\ 。

1. **init**

-  它影响了\ **``输出参数的起始化过程``**

-  **如果我们有一个模型，它的输出结果会用来作为\ ``GBM``\ 模型的起始估计，这个时候就可以用\ ``init``**

1. **random\_ state**

-  作为每次产生随机数的随机种子

-  **``使用随机种子对于调参过程是很重要的，因为如果我们每次都用不同的随机种子，即使参数值没变每次出来的结果也会不同，这样不利于比较不同模型的结果``**\ 。

-  **``任一个随机样本都有可能导致过度拟合``\ ，可以用\ ``不同的随机样本建模来减少过度拟合的可能``\ ，但这样计算上也会昂贵很多，因而我们很少这样用**

1. **verbose**

-  决定建模完成后对输出的打印方式：

.. code:: 

   - 0：不输出任何结果（默认）

   - 1：打印特定区域的树的输出结果

   - **`>1：打印所有结果`**

1. **warm\_ start**

-  这个参数的效果很有趣，有效地使用它可以省很多事

-  **使用它我们就可以\ ``用一个建好的模型来训练额外的决策树，能节省大量的时间``\ ，对于高阶应用我们应该多多探索这个选项。**

1. **presort**

-  **决定是否对数据进行预排序，可以使得\ ``树分裂地更快``\ 。**

-  默认情况下是\ **自动选择的，当然你可以对其更改**

.. _5参数调节实例:

**5.参数调节实例**
==================

接下来要用的数据集来自Data Hackathon 3.x AV
hackathon。比赛的细节可以在比赛网站上找到（\ http://datahack.analyticsvidhya.com/contest/data-hackathon-3x\ ），数据可以从这里下载：\ http://www.analyticsvidhya.com/wp-content/uploads/2016/02/Dataset.rar\ 。我对数据做了一些清洗：

-  City这个变量已经被我舍弃了，因为有太多种类了。

-  DOB转为Age|DOB,舍弃了DOB

-  创建了\ ``EMI_Loan_Submitted_Missing``\ 这个变量，当\ ``EMI_Loan_Submitted``
   变量值缺失时它的值为1，否则为0。然后舍弃了\ ``EMI_Loan_Submitted``\ 。

-  EmployerName的值也太多了，我把它也舍弃了

-  Existing_EMI的缺失值被填补为0（中位数），因为只有111个缺失值

-  创建了\ ``Interest_Rate_Missing``\ 变量，类似于#3，当\ ``Interest_Rate``\ 有值时它的值为0，反之为1，原来的Interest_Rate变量被舍弃了

-  Lead_Creation_Date也被舍弃了，因为对结果看起来没什么影响

-  用\ ``Loan_Amount_Applied``\ 和
   ``Loan_Tenure_Applied``\ 的中位数填补了缺失值

-  创建了\ ``Loan_Amount_Submitted_Missing``\ 变量，当\ ``Loan_Amount_Submitted``\ 有缺失值时为1，反之为0，原本的\ ``Loan_Amount_Submitted``\ 变量被舍弃

-  创建了\ ``Loan_Tenure_Submitted_Missing``\ 变量，当\ ``Loan_Tenure_Submitted``\ 有缺失值时为1，反之为0，原本的\ ``Loan_Tenure_Submitted``\ 变量被舍弃

-  舍弃了LoggedIn,和Salary_Account

-  创建了\ ``Processing_Fee_Missing``\ 变量，当\ ``Processing_Fee``\ 有缺失值时为1，反之为0，原本的\ ``Processing_Fee``\ 变量被舍弃

-  Source-top保留了2个，其他组合成了不同的类别

-  对一些变量采取了数值化和独热编码（One-Hot-Coding）操作

你们可以从\ ``GitHub``\ 里\ ``data_preparation iPython notebook``\ 中看到这些改变。

首先，我们加载需要的库和数据：

.. code:: python

   import pandas as pd
   import numpy as np
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn import cross_validation, metrics
   from sklearn.model_selection import GridSearchCV

   import matplotlib.pylab as plt

   %matplotlib inline
   from matplotlib.pylab import rcParams
   rcParams['figure.figsize'] = 16, 9

   train = pd.read_csv('train_modified.csv')
   target = 'Disbursed'
   IDcol = 'ID'

.. code:: python

   def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
       # 训练模型
       alg.fit(dtrain[predictors], dtrain['Disbursed'])

       # 预测训练集
       dtrain_predictions = alg.predict(dtrain[predictors])
       dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

       # cross-validation
       if performCV:
           cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], \
                                                       dtrain['Disbursed'], \
                                                       cv=cv_folds, \
                                                       scoring='roc_auc')

       # 打印模型报告
       print("Model Report")
       print("Accuracy : {0:.4}".format(metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)))
       print("AUC Score (Train): {0:.4}".format(metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)))

       if performCV:
           print("CV Score : Mean - {0:.7} | Std - {1:.7} | Min - {2:.7} | Max - {3:.7}".\
                 format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

       # 打印重要特征值
       if printFeatureImportance:
           feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
           feat_imp.plot(kind='bar', title='Feature Importances')
           plt.ylabel('Feature Importance Score')

.. code:: 

   """
   接着就要创建一个基线模型（baseline model）。
   这里我们用AUC来作为衡量标准，所以用常数的话AUC就是0.5。
   一般来说用默认参数设置的GBM模型就是一个很好的基线模型，我们来看看这个模型的输出和特征重要性
   """
   #Choose all predictors except target & IDcols
   predictors = [x for x in train.columns if x not in [target, IDcol]]
   gbm0 = GradientBoostingClassifier(random_state=10) # 建模
   modelfit(gbm0, train, predictors)     # alg==gbm0 dtrain==train

.. code:: 

   Model Report
   Accuracy : 0.9856
   AUC Score (Train): 0.8623
   CV Score : Mean - 0.8318589 | Std - 0.008756969 | Min - 0.820805 | Max - 0.8438558

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-c0d8a414cb1df8f5.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

从图上看出，\ ``CV``\ 的平均值是\ ``0.8319``\ ，后面调整的模型会做得比这个更好。

.. _51-参数调节的一般方法:

**5.1 参数调节的一般方法**
--------------------------

之前说过，我们要调节的参数有两种：\ **``树参数``\ 和\ ``boosting参数``**\ 。\ **``learning rate``\ 没有什么特别的调节方法，因为只要我们\ ``训练的树足够多learning rate总是小值``\ 来得好。**

虽然随着\ **决策树的增多GBM并不会明显得过度拟合，高learing
rate还是会导致这个问题**\ ，但如果我们\ **一味地减小learning
rate、增多树,计算就会非常昂贵而且需要运行很长时间**\ 。了解了这些问题，我们决定采取以下方法调参策略：

   1. 选择一个相对来说\ **稍微高一点的\ ``learning rate``**\ 。\ **一般默认的值是\ ``0.1``\ ，不过针对不同的问题，\ ``0.05``\ 到\ ``0.2``\ 之间都可以**

   1. 决定\ **当前\ ``learning rate``\ 下最优的\ ``决策树数量``**\ 。它的值应该在\ **``40-70``**\ 之间。记得选择一个你的电脑还能快速运行的值，因为之后这些树会用来做很多测试和调参。

   1. 接着\ **调节树参数**\ 来调整learning
      rate和树的数量。我们可以选择不同的参数来定义一个决策树，后面会有这方面的例子

   1. **降低learning
      rate**\ ，同时会\ ``增加相应的决策树数量使得模型更加稳健``

.. _52-固定-learning-rate和需要估测的决策树数量:

**5.2 固定 ``learning rate``\ 和需要估测的\ ``决策树数量``**
------------------------------------------------------------

为了决定\ ``boosting``\ 参数，我们得先设定一些参数的初始值，可以像下面这样：

1. **``min_ samples_ split=500``:**
   这个值应该在\ **总样本数的\ ``0.5-1%``\ 之间**\ ，由于我们研究的是\ **``不均等分类问题``\ ，我们可以\ ``取这个区间里一个比较小的数``\ ，\ ``500``\ 。**

2. **``min_ samples_ leaf=50``:**
   可以凭感觉选一个合适的数，\ **只要不会造成过度拟合**\ 。同样因为\ ``不均等分类的原因``\ ，这里我们选择一个\ ``比较小的值``\ 。

3. **``max_ depth=8``:**
   根据\ ``观察数和自变量数``\ ，这个值应该在\ ``5-8``\ 之间。这里我们的数据有\ ``87000``\ 行，\ ``49``\ 列，所以我们先选\ ``深度为8``\ 。

4. **``max_ features=’sqrt’``:** 经验上一般都\ **``选择平方根``**\ 。

5. **``subsample=0.8``:** 开始的时候一般就用\ ``0.8``

注意我们目前定的都是初始值，最终这些参数的值应该是多少还要靠调参决定。

现在我们可以根据\ ``learning rate``\ 的默认值\ ``0.1``\ 来找到所需要的最佳的\ ``决策树数量``\ ，可以利用\ **``网格搜索（grid search）实现``\ ，以10个数递增，从\ ``20测到80``\ 。**\ (先找到决策树数量)

.. code:: python

   #利用网格搜索（grid search）实现，以10个数递增，从20测到80
   #Choose all predictors except target & IDcols

   predictors = [x for x in train.columns if x not in [target, IDcol]]
   param_test1 = {'n_estimators':range(20,81,10)}
   gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, \
                                                                  min_samples_split=500,\
                                                                  min_samples_leaf=50,\
                                                                  max_depth=8,\
                                                                  max_features='sqrt',\
                                                                  subsample=0.8,\
                                                                  random_state=10),\
                           param_grid = param_test1, \
                           scoring='roc_auc',\
                           n_jobs=4,\
                           iid=False, \
                           cv=5)

   gsearch1.fit(train[predictors],train[target])

.. code:: python

   GridSearchCV(cv=5, error_score='raise',
          estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='deviance', max_depth=8,
                 max_features='sqrt', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=50, min_samples_split=500,
                 min_weight_fraction_leaf=0.0, n_estimators=100,
                 presort='auto', random_state=10, subsample=0.8, verbose=0,
                 warm_start=False),
          fit_params=None, iid=False, n_jobs=4,
          param_grid={'n_estimators': range(20, 81, 10)},
          pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
          scoring='roc_auc', verbose=0)

来看一下输出结果：

.. code:: python

   gsearch1.grid_scores_
   [mean: 0.83337, std: 0.00991, params: {'n_estimators': 20},
    mean: 0.83697, std: 0.00994, params: {'n_estimators': 30},
    mean: 0.83832, std: 0.01050, params: {'n_estimators': 40},
    mean: 0.83867, std: 0.01081, params: {'n_estimators': 50},
    mean: 0.83939, std: 0.01077, params: {'n_estimators': 60},
    mean: 0.83891, std: 0.01044, params: {'n_estimators': 70},
    mean: 0.83807, std: 0.01093, params: {'n_estimators': 80}]

.. code:: python

   gsearch1.best_params_
   {'n_estimators': 60}

.. code:: python

   gsearch1.best_score_
   0.83938752161776975

可以看出对于\ ``0.1``\ 的\ ``learning rate``,
``60个树是最佳的``\ ，而且\ ``60也是一个合理的决策树数量``\ ，所以我们就直接用\ ``60``\ 。但在一些情况下上面这段代码给出的结果可能不是我们想要的，比如：

1. 如果给出的输出是\ ``20``\ ，可能就要降低我们的\ ``learning rate``\ 到\ ``0.05``\ ，然后再搜索一遍。

2. 如果\ ``输出值太高``\ ，比如\ ``100``\ ，因为调节其他参数需要很长时间，这时候可以把\ ``learniing rate``\ 稍微\ ``调高一点``\ 。

.. _53-调节树参数:

**5.3 调节树参数**
------------------

树参数可以按照这些步骤调节：

1. 调节\ ``max_depth``\ 和 ``min_samples_split``

2. 调节\ ``min_samples_leaf``

3. 调节\ ``max_features``

需要注意一下\ **调参顺序**\ ，对结果影响最大的参数应该优先调节，就像\ ``max_depth``\ 和\ ``min_samples_split``\ 。

**重要提示：接着我会做比较久的\ ``网格搜索(grid search)``\ ，可能会花上15-30分钟。你在自己尝试的时候应该根据电脑情况适当调整需要测试的值。**

``max_depth``\ 可以相隔两个数从5测到15，而\ ``min_samples_split``\ 可以按相隔200从200测到1000。这些完全凭经验和直觉，如果先测更大的范围再用迭代去缩小范围也是可行的。

.. code:: python

   # 调节树参数
   param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
   gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, \
                                                                  n_estimators=60, \
                                                                  max_features='sqrt', \
                                                                  subsample=0.8, \
                                                                  random_state=10), \
                           param_grid = param_test2, \
                           scoring='roc_auc',\
                           n_jobs=4,\
                           iid=False, \
                           cv=5)

   gsearch2.fit(train[predictors],train[target])

   gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

.. code:: python

   ([mean: 0.83297, std: 0.01226, params: {'max_depth': 5, 'min_samples_split': 200},
     mean: 0.83251, std: 0.01054, params: {'max_depth': 5, 'min_samples_split': 400},
     mean: 0.83386, std: 0.01415, params: {'max_depth': 5, 'min_samples_split': 600},
     mean: 0.83379, std: 0.01169, params: {'max_depth': 5, 'min_samples_split': 800},
     mean: 0.83339, std: 0.01266, params: {'max_depth': 5, 'min_samples_split': 1000},
     mean: 0.83392, std: 0.00758, params: {'max_depth': 7, 'min_samples_split': 200},
     mean: 0.83663, std: 0.00991, params: {'max_depth': 7, 'min_samples_split': 400},
     mean: 0.83481, std: 0.00826, params: {'max_depth': 7, 'min_samples_split': 600},
     mean: 0.83786, std: 0.01067, params: {'max_depth': 7, 'min_samples_split': 800},
     mean: 0.83769, std: 0.01060, params: {'max_depth': 7, 'min_samples_split': 1000},
     mean: 0.83581, std: 0.01003, params: {'max_depth': 9, 'min_samples_split': 200},
     mean: 0.83729, std: 0.00959, params: {'max_depth': 9, 'min_samples_split': 400},
     mean: 0.83317, std: 0.00881, params: {'max_depth': 9, 'min_samples_split': 600},
     mean: 0.83831, std: 0.00953, params: {'max_depth': 9, 'min_samples_split': 800},
     mean: 0.83753, std: 0.01012, params: {'max_depth': 9, 'min_samples_split': 1000},
     mean: 0.82978, std: 0.00888, params: {'max_depth': 11, 'min_samples_split': 200},
     mean: 0.82951, std: 0.00621, params: {'max_depth': 11, 'min_samples_split': 400},
     mean: 0.83305, std: 0.01017, params: {'max_depth': 11, 'min_samples_split': 600},
     mean: 0.83192, std: 0.00844, params: {'max_depth': 11, 'min_samples_split': 800},
     mean: 0.83566, std: 0.01018, params: {'max_depth': 11, 'min_samples_split': 1000},
     mean: 0.82438, std: 0.01078, params: {'max_depth': 13, 'min_samples_split': 200},
     mean: 0.83010, std: 0.00862, params: {'max_depth': 13, 'min_samples_split': 400},
     mean: 0.83228, std: 0.01020, params: {'max_depth': 13, 'min_samples_split': 600},
     mean: 0.83480, std: 0.01193, params: {'max_depth': 13, 'min_samples_split': 800},
     mean: 0.83372, std: 0.00844, params: {'max_depth': 13, 'min_samples_split': 1000},
     mean: 0.82056, std: 0.00913, params: {'max_depth': 15, 'min_samples_split': 200},
     mean: 0.82217, std: 0.00961, params: {'max_depth': 15, 'min_samples_split': 400},
     mean: 0.82916, std: 0.00927, params: {'max_depth': 15, 'min_samples_split': 600},
     mean: 0.82900, std: 0.01046, params: {'max_depth': 15, 'min_samples_split': 800},
     mean: 0.83320, std: 0.01389, params: {'max_depth': 15, 'min_samples_split': 1000}],
    {'max_depth': 9, 'min_samples_split': 800},
    0.8383109442669946)

从结果可以看出，我们从\ ``30``\ 种组合中找出最佳的\ ``max_depth``\ 是\ ``9``\ ，而最佳的\ ``min_smaples_split``\ 是\ ``1000``\ 。\ ``1000``\ 是我们设定的范围里的最大值，有可能真正的最佳值比\ ``1000``\ 还要大，所以我们还要继续增加\ ``min_smaples_split``\ 。树深就用\ ``9``\ 。接着就来调节\ ``min_samples_leaf``\ ，可以测\ ``30，40，50，60，70``\ 这五个值，同时我们也试着调大\ ``min_samples_leaf``\ 的值。

.. code:: python

   param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
   gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, \
                                                                  n_estimators=60,\
                                                                  max_depth=9,\
                                                                  max_features='sqrt', \
                                                                  subsample=0.8, \
                                                                  random_state=10), \
                           param_grid = param_test3, \
                           scoring='roc_auc',\
                           n_jobs=4,\
                           iid=False, \
                           cv=5)

   gsearch3.fit(train[predictors],train[target])

   gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

.. code:: python

   ([mean: 0.83821, std: 0.01092, params: {'min_samples_split': 1000, 'min_samples_leaf': 30},
     mean: 0.83889, std: 0.01271, params: {'min_samples_split': 1200, 'min_samples_leaf': 30},
     mean: 0.83552, std: 0.01024, params: {'min_samples_split': 1400, 'min_samples_leaf': 30},
     mean: 0.83683, std: 0.01429, params: {'min_samples_split': 1600, 'min_samples_leaf': 30},
     mean: 0.83958, std: 0.01233, params: {'min_samples_split': 1800, 'min_samples_leaf': 30},
     mean: 0.83852, std: 0.01097, params: {'min_samples_split': 2000, 'min_samples_leaf': 30},
     mean: 0.83851, std: 0.00908, params: {'min_samples_split': 1000, 'min_samples_leaf': 40},
     mean: 0.83757, std: 0.01274, params: {'min_samples_split': 1200, 'min_samples_leaf': 40},
     mean: 0.83757, std: 0.01074, params: {'min_samples_split': 1400, 'min_samples_leaf': 40},
     mean: 0.83779, std: 0.01199, params: {'min_samples_split': 1600, 'min_samples_leaf': 40},
     mean: 0.83764, std: 0.01366, params: {'min_samples_split': 1800, 'min_samples_leaf': 40},
     mean: 0.83759, std: 0.01222, params: {'min_samples_split': 2000, 'min_samples_leaf': 40},
     mean: 0.83650, std: 0.00983, params: {'min_samples_split': 1000, 'min_samples_leaf': 50},
     mean: 0.83784, std: 0.01169, params: {'min_samples_split': 1200, 'min_samples_leaf': 50},
     mean: 0.83892, std: 0.01234, params: {'min_samples_split': 1400, 'min_samples_leaf': 50},
     mean: 0.83825, std: 0.01371, params: {'min_samples_split': 1600, 'min_samples_leaf': 50},
     mean: 0.83806, std: 0.01099, params: {'min_samples_split': 1800, 'min_samples_leaf': 50},
     mean: 0.83821, std: 0.01014, params: {'min_samples_split': 2000, 'min_samples_leaf': 50},
     mean: 0.83636, std: 0.01118, params: {'min_samples_split': 1000, 'min_samples_leaf': 60},
     mean: 0.83976, std: 0.00994, params: {'min_samples_split': 1200, 'min_samples_leaf': 60},
     mean: 0.83735, std: 0.01217, params: {'min_samples_split': 1400, 'min_samples_leaf': 60},
     mean: 0.83685, std: 0.01325, params: {'min_samples_split': 1600, 'min_samples_leaf': 60},
     mean: 0.83626, std: 0.01153, params: {'min_samples_split': 1800, 'min_samples_leaf': 60},
     mean: 0.83788, std: 0.01147, params: {'min_samples_split': 2000, 'min_samples_leaf': 60},
     mean: 0.83751, std: 0.01027, params: {'min_samples_split': 1000, 'min_samples_leaf': 70},
     mean: 0.83854, std: 0.01111, params: {'min_samples_split': 1200, 'min_samples_leaf': 70},
     mean: 0.83777, std: 0.01186, params: {'min_samples_split': 1400, 'min_samples_leaf': 70},
     mean: 0.83796, std: 0.01093, params: {'min_samples_split': 1600, 'min_samples_leaf': 70},
     mean: 0.83816, std: 0.01052, params: {'min_samples_split': 1800, 'min_samples_leaf': 70},
     mean: 0.83677, std: 0.01164, params: {'min_samples_split': 2000, 'min_samples_leaf': 70}],
    {'min_samples_leaf': 60, 'min_samples_split': 1200},
    0.83975976288429499)

这样\ ``min_samples_split``\ 的最佳值是\ ``1200``\ ，而\ ``min_samples_leaf``\ 的最佳值是\ ``60``\ 。注意现在\ ``CV``\ 值增加到了\ ``0.8398``\ 。现在我们就根据这个结果来重新建模，并再次评估特征的重要性。

.. code:: python

   modelfit(gsearch3.best_estimator_, train, predictors)

.. code:: python

   Model Report
   Accuracy : 0.9854
   AUC Score (Train): 0.8965
   CV Score : Mean - 0.8397598 | Std - 0.009936017 | Min - 0.8255474 | Max - 0.8527672

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-40a960da0b1e8bd9.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

比较之前的\ ``基线模型``\ 结果可以看出，现在我们的\ ``模型用了更多的特征``\ ，并且\ ``基线模型里少数特征的重要性评估值过高，分布偏斜明显，现在分布得更加均匀了``\ 。

接下来就剩下最后的树参数\ ``max_features``\ 了，可以每隔两个数从\ ``7``\ 测到\ ``19``\ 。

.. code:: python

   param_test4 = {'max_features':range(7,20,2)}
   gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, \
                                                                  n_estimators=60,\
                                                                  max_depth=9, \
                                                                  min_samples_split=1200, \
                                                                  min_samples_leaf=60, \
                                                                  subsample=0.8, \
                                                                  random_state=10),\
                           param_grid = param_test4, \
                           scoring='roc_auc',\
                           n_jobs=4,\
                           iid=False, \
                           cv=5)

   gsearch4.fit(train[predictors],train[target])

   gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

.. code:: python

   ([mean: 0.83976, std: 0.00994, params: {'max_features': 7},
     mean: 0.83648, std: 0.00988, params: {'max_features': 9},
     mean: 0.83919, std: 0.01042, params: {'max_features': 11},
     mean: 0.83738, std: 0.01017, params: {'max_features': 13},
     mean: 0.83898, std: 0.01101, params: {'max_features': 15},
     mean: 0.83495, std: 0.00931, params: {'max_features': 17},
     mean: 0.83524, std: 0.01018, params: {'max_features': 19}],
    {'max_features': 7},
    0.83975976288429499)

最佳的结果是\ ``7``\ ，正好就是我们设定的初始值（平方根）。当然你可能还想测测小于7的值，我也鼓励你这么做。而按照我们的设定，现在的树参数是这样的：

-  ``min_samples_split``: 1200

-  ``min_samples_leaf``: 60

-  ``max_depth``: 9

-  ``max_features``: 7

.. _54-调节子样本比例来降低learning-rate:

**5.4 调节子样本比例来降低learning rate**
-----------------------------------------

接下来就可以调节子样本占总样本的比例，我准备尝试这些值：\ ``0.6,0.7,0.75,0.8,0.85,0.9``\ 。

.. code:: python

   # 调节子样本比例来降低learning rate
   param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
   gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, \
                                                                  n_estimators=60,\
                                                                  max_depth=9,\
                                                                  min_samples_split=1200, \
                                                                  min_samples_leaf=60, \
                                                                  subsample=0.8, \
                                                                  random_state=10,\
                                                                  max_features=7),\
                           param_grid = param_test5, \
                           scoring='roc_auc',\
                           n_jobs=4,\
                           iid=False, \
                           cv=5)

   gsearch5.fit(train[predictors],train[target])

   gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

.. code:: python

   ([mean: 0.83645, std: 0.00942, params: {'subsample': 0.6},
     mean: 0.83629, std: 0.01185, params: {'subsample': 0.7},
     mean: 0.83601, std: 0.01074, params: {'subsample': 0.75},
     mean: 0.83976, std: 0.00994, params: {'subsample': 0.8},
     mean: 0.84086, std: 0.00997, params: {'subsample': 0.85},
     mean: 0.83828, std: 0.00984, params: {'subsample': 0.9}],
    {'subsample': 0.85},
    0.84085800832187396)

给出的结果是\ ``0.85``\ 。这样所有的参数都设定好了，现在我们要做的就是进一步减少\ ``learning rate``\ ，就相应地增加了树的数量。需要注意的是树的个数是被动改变的，可能不是最佳的，但也很合适。随着树个数的增加，找到最佳值和\ ``CV``\ 的计算量也会加大，为了看出模型执行效率，我还提供了我每个模型在比赛的排行分数（\ ``leaderboard score``\ ），怎么得到这个数据不是公开的，你很难重现这个数字，它只是为了更好地帮助我们理解模型表现。

现在我们先把\ ``learning rate``\ 降一半，至\ ``0.05``\ ，这样树的个数就相应地加倍到\ ``120``\ 。

.. code:: python

   predictors = [x for x in train.columns if x not in [target, IDcol]]
   gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, \
                                            n_estimators=120,\
                                            max_depth=9, \
                                            min_samples_split=1200,\
                                            min_samples_leaf=60, \
                                            subsample=0.85, \
                                            random_state=10, \
                                            max_features=7)

   modelfit(gbm_tuned_1, train, predictors)

.. code:: python

   Model Report
   Accuracy : 0.9854
   AUC Score (Train): 0.8976
   CV Score : Mean - 0.8391332 | Std - 0.009437997 | Min - 0.8271238 | Max - 0.8511221

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-db3946a242dbf9e3.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

**排行得分：0.844139**

接下来我们把\ ``learning rate``\ 进一步减小到原值的\ **十分之一**,即\ ``0.01``\ ，相应地，树的个数变为\ ``600``\ 。

.. code:: python

   predictors = [x for x in train.columns if x not in [target, IDcol]]
   gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, \
                                            n_estimators=600,\
                                            max_depth=9, \
                                            min_samples_split=1200,\
                                            min_samples_leaf=60, \
                                            subsample=0.85, \
                                            random_state=10, \
                                            max_features=7)

   modelfit(gbm_tuned_2, train, predictors)

.. code:: python

   Model Report
   Accuracy : 0.9854
   AUC Score (Train): 0.9
   CV Score : Mean - 0.8407913 | Std - 0.01011421 | Min - 0.8255379 | Max - 0.8522251

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-2e15ebec61a16b4e.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

**排行得分：0.848145**

继续把\ ``learning rate``\ 缩小至\ **二十分之一**\ ，即\ ``0.005``,这时候我们有\ ``1200``\ 个树。

.. code:: python

   predictors = [x for x in train.columns if x not in [target, IDcol]]
   gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.005, \
                                            n_estimators=1200,\
                                            max_depth=9, \
                                            min_samples_split=1200, \
                                            min_samples_leaf=60, \
                                            subsample=0.85, \
                                            random_state=10, \
                                            max_features=7,\
                                            warm_start=True)

   modelfit(gbm_tuned_3, train, predictors, performCV=False)

.. code:: python

   Model Report
   Accuracy : 0.9854
   AUC Score (Train): 0.9007

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-1c956b4875bd8f3c.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

**排行得分：0.848112**

排行得分稍微降低了，我们停止减少\ ``learning rate``\ ，只单方面增加树的个数，试试\ ``1500``\ 个树。

.. code:: python

   predictors = [x for x in train.columns if x not in [target, IDcol]]
   gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.005, \
                                            n_estimators=1500,\
                                            max_depth=9, \
                                            min_samples_split=1200, \
                                            min_samples_leaf=60, \
                                            subsample=0.85, \
                                            random_state=10, \
                                            max_features=7,\
                                            warm_start=True)

   modelfit(gbm_tuned_4, train, predictors, performCV=False)

.. code:: python

   Model Report
   Accuracy : 0.9854
   AUC Score (Train): 0.9063

.. figure:: https://upload-images.jianshu.io/upload_images/8885151-701b1da5dcd56019.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240#width=
   :alt: 

**排行得分：0.848747**

看，就这么简单，排行得分已经从\ ``0.844``\ 升高到\ ``0.849``\ 了，这可是一个很大的提升。

还有一个技巧就是用“\ ``warm_start``\ ”选项。这样每次用不同个数的树都不用重新开始。

.. _6总结:

**6.总结**
==========

这篇文章详细地介绍了GBM模型。我们首先了解了\ **何为boosting**\ ，然后详细介绍了\ **各种参数**\ 。
这些参数可以被分为3类：树参数，boosting参数，和其他影响模型的参数。最后我们提到了用GBM解决问题的
**一般方法**\ ，并且用\ **AV Data Hackathon 3.x
problem**\ 数据运用了这些方法。最后，希望这篇文章确实帮助你更好地理解了GBM，在下次运用GBM解决问题的时候也更有信心。

.. _7附录官方帮助文档:

**7.附录**\ ：官方帮助文档
==========================

.. code:: python

   In [1]: from sklearn.ensemble import GradientBoostingClassifier

   In [2]: GradientBoostingClassifier?
   Init signature: GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, verbose=0, max_leaf_
   nodes=None, warm_start=False, presort='auto')
   Docstring:
   Gradient Boosting for classification.

   GB builds an additive model in a
   forward stage-wise fashion; it allows for the optimization of
   arbitrary differentiable loss functions. In each stage ``n_classes_``
   regression trees are fit on the negative gradient of the
   binomial or multinomial deviance loss function. Binary classification
   is a special case where only a single regression tree is induced.

   Read more in the :ref:`User Guide <gradient_boosting>`.

   Parameters
   ----------
   loss : {'deviance', 'exponential'}, optional (default='deviance')
       loss function to be optimized. 'deviance' refers to
       deviance (= logistic regression) for classification
       with probabilistic outputs. For loss 'exponential' gradient
       boosting recovers the AdaBoost algorithm.

   learning_rate : float, optional (default=0.1)
       learning rate shrinks the contribution of each tree by `learning_rate`.
       There is a trade-off between learning_rate and n_estimators.

   n_estimators : int (default=100)
       The number of boosting stages to perform. Gradient boosting
       is fairly robust to over-fitting so a large number usually
       results in better performance.

   max_depth : integer, optional (default=3)
       maximum depth of the individual regression estimators. The maximum
       depth limits the number of nodes in the tree. Tune this parameter
       for best performance; the best value depends on the interaction
       of the input variables.

   criterion : string, optional (default="friedman_mse")
       The function to measure the quality of a split. Supported criteria
       are "friedman_mse" for the mean squared error with improvement
       score by Friedman, "mse" for mean squared error, and "mae" for
       the mean absolute error. The default value of "friedman_mse" is
       generally the best as it can provide a better approximation in
       some cases.

       .. versionadded:: 0.18

   min_samples_split : int, float, optional (default=2)
       The minimum number of samples required to split an internal node:

       - If int, then consider `min_samples_split` as the minimum number.
       - If float, then `min_samples_split` is a percentage and
         `ceil(min_samples_split * n_samples)` are the minimum
         number of samples for each split.

       .. versionchanged:: 0.18
          Added float values for percentages.

   min_samples_leaf : int, float, optional (default=1)
       The minimum number of samples required to be at a leaf node:

       - If int, then consider `min_samples_leaf` as the minimum number.
       - If float, then `min_samples_leaf` is a percentage and
         `ceil(min_samples_leaf * n_samples)` are the minimum
         number of samples for each node.

       .. versionchanged:: 0.18
          Added float values for percentages.

   min_weight_fraction_leaf : float, optional (default=0.)
       The minimum weighted fraction of the sum total of weights (of all
       the input samples) required to be at a leaf node. Samples have
       equal weight when sample_weight is not provided.

   subsample : float, optional (default=1.0)
       The fraction of samples to be used for fitting the individual base
       learners. If smaller than 1.0 this results in Stochastic Gradient
       Boosting. `subsample` interacts with the parameter `n_estimators`.
       Choosing `subsample < 1.0` leads to a reduction of variance
       and an increase in bias.

   max_features : int, float, string or None, optional (default=None)
       The number of features to consider when looking for the best split:

       - If int, then consider `max_features` features at each split.
       - If float, then `max_features` is a percentage and
         `int(max_features * n_features)` features are considered at each
         split.
       - If "auto", then `max_features=sqrt(n_features)`.
       - If "sqrt", then `max_features=sqrt(n_features)`.
       - If "log2", then `max_features=log2(n_features)`.
       - If None, then `max_features=n_features`.

       Choosing `max_features < n_features` leads to a reduction of variance
       and an increase in bias.

       Note: the search for a split does not stop until at least one
       valid partition of the node samples is found, even if it requires to
       effectively inspect more than ``max_features`` features.

   max_leaf_nodes : int or None, optional (default=None)
       Grow trees with ``max_leaf_nodes`` in best-first fashion.
       Best nodes are defined as relative reduction in impurity.
       If None then unlimited number of leaf nodes.

   min_impurity_split : float, optional (default=1e-7)
       Threshold for early stopping in tree growth. A node will split
       if its impurity is above the threshold, otherwise it is a leaf.

       .. versionadded:: 0.18

   init : BaseEstimator, None, optional (default=None)
       An estimator object that is used to compute the initial
       predictions. ``init`` has to provide ``fit`` and ``predict``.
       If None it uses ``loss.init_estimator``.

   verbose : int, default: 0
       Enable verbose output. If 1 then it prints progress and performance
       once in a while (the more trees the lower the frequency). If greater
       than 1 then it prints progress and performance for every tree.

   warm_start : bool, default: False
       When set to ``True``, reuse the solution of the previous call to fit
       and add more estimators to the ensemble, otherwise, just erase the
       previous solution.

   random_state : int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.

   presort : bool or 'auto', optional (default='auto')
       Whether to presort the data to speed up the finding of best splits in
       fitting. Auto mode by default will use presorting on dense data and
       default to normal sorting on sparse data. Setting presort to true on
       sparse data will raise an error.

       .. versionadded:: 0.17
          *presort* parameter.

   Attributes
   ----------
   feature_importances_ : array, shape = [n_features]
       The feature importances (the higher, the more important the feature).

   oob_improvement_ : array, shape = [n_estimators]
       The improvement in loss (= deviance) on the out-of-bag samples
       relative to the previous iteration.
       ``oob_improvement_[0]`` is the improvement in
       loss of the first stage over the ``init`` estimator.

   train_score_ : array, shape = [n_estimators]
       The i-th score ``train_score_[i]`` is the deviance (= loss) of the
       model at iteration ``i`` on the in-bag sample.
       If ``subsample == 1`` this is the deviance on the training data.

   loss_ : LossFunction
       The concrete ``LossFunction`` object.

   init : BaseEstimator
       The estimator that provides the initial predictions.
       Set via the ``init`` argument or ``loss.init_estimator``.

   estimators_ : ndarray of DecisionTreeRegressor, shape = [n_estimators, ``loss_.K``]
       The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
       classification, otherwise n_classes.


   See also
   --------
   sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
   AdaBoostClassifier

   References
   ----------
   J. Friedman, Greedy Function Approximation: A Gradient Boosting
   Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

   J. Friedman, Stochastic Gradient Boosting, 1999

   T. Hastie, R. Tibshirani and J. Friedman.
   Elements of Statistical Learning Ed. 2, Springer, 2009.
