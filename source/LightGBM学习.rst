============
LightGBM学习
============

:Date:   2019-08-02T20:57:29+08:00

[参考整理]

.. _1-lightgbm原理br-:

1 LightGBM原理
==============

.. _11-gbdt和-lightgbm对比:

1.1 GBDT和 LightGBM对比
-----------------------

GBDT (Gradient Boosting Decision Tree)
是机器学习中一个长盛不衰的模型，其主要思想是利用弱分类器（决策树）迭代训练以得到最优模型，该模型具有训练效果好、不易过拟合等优点。GBDT
在工业界应用广泛，通常被用于点击率预测，搜索排序等任务。GBDT
也是各种数据挖掘竞赛的致命武器，据统计 Kaggle
上的比赛有一半以上的冠军方案都是基于 GBDT。

LightGBM（Light Gradient Boosting
Machine）同样是一款基于决策树算法的分布式梯度提升框架。为了满足工业界缩短模型计算时间的需求，LightGBM的设计思路主要是两点：

1. 减小数据对内存的使用，保证单个机器在不牺牲速度的情况下，尽可能地用上更多的数据；

2. 减小通信的代价，提升多机并行时的效率，实现在计算上的线性加速。

由此可见，LightGBM的设计初衷就是提供一个快速高效、低内存占用、高准确度、支持并行和大规模数据处理的数据科学工具。

| XGBoost和LightGBM都是基于决策树提升(Tree
  Boosting)的工具，都拥有对输入要求不敏感、计算复杂度不高和效果好的特点，适合在工业界中进行大量的应用。
| 主页地址：\ `http://lightgbm.apachecn.org <http://lightgbm.apachecn.org/>`__

LightGBM （Light Gradient Boosting Machine）是一个实现 GBDT
算法的框架，支持高效率的并行训练，并且具有以下优点：

-  更快的训练速度

-  更低的内存消耗

-  更好的准确率

-  分布式支持，可以快速处理海量数据

如下图，在 Higgs 数据集上 LightGBM 比 XGBoost 快将近 10
倍，内存占用率大约为 XGBoost 的1/6，并且准确率也有提升。

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137735531-af6b0c3b-d170-4b45-9cd4-f46055949bed.png#align=left&display=inline&height=391&originHeight=391&originWidth=679&size=0&status=done&width=679#align=left&display=inline&height=391&originHeight=391&originWidth=679&status=done&width=679
   :alt: 

.. _12-lightgbm-的动机:

1.2 LightGBM 的动机
-------------------

常用的机器学习算法，例如神经网络等算法，都可以以 mini-batch
的方式训练，训练数据的大小不会受到内存限制。

而 GBDT
在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的
GBDT 算法是不能满足其需求的。

   LightGBM 提出的主要原因就是为了解决 GBDT 在海量数据遇到的问题，让
   GBDT 可以更好更快地用于工业实践。

.. _13-xgboost-原理:

1.3 Xgboost 原理
----------------

目前已有的 GBDT
工具基本都是基于\ **预排序的方法（pre-sorted）的决策树算法(如
xgboost)**\ 。这种构建决策树的算法基本思想是：

-  首先，对所有特征都按照特征的数值进行预排序。

-  其次，在遍历分割点的时候用O(#data)的代价找到一个特征上的最好分割点。

-  最后，找到一个特征的分割点后，将数据分裂成左右子节点。

-  **这样的预排序算法的优点是：能精确地找到分割点。**

缺点：

-  每轮迭代时，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。

-  预排序方法（pre-sorted）：

   -  首先，空间消耗大。\ **这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如排序后的索引，为了后续快速的计算分割点），这里需要消耗训练数据两倍的内存。**

   -  其次时间上也有较大的开销，\ **在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。**

-  对Cache优化不友好。\ **在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化**\ 。同时，\ **在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache
   miss**\ 。

.. _14-lightgbm-优化:

1.4 LightGBM 优化
-----------------

LightGBM 优化部分包含以下：

-  基于 Histogram 的决策树算法

-  带深度限制的 Leaf-wise 的叶子生长策略

-  直方图做差加速

-  直接支持类别特征(Categorical Feature)

-  Cache 命中率优化

-  基于直方图的稀疏特征优化

-  多线程优化。

下面主要介绍 Histogram 算法、带深度限制的 Leaf-wise
的叶子生长策略和直方图做差加速。

.. _141-histogram-算法:

1.4.1 Histogram 算法
~~~~~~~~~~~~~~~~~~~~

| 直方图算法的基本思想：\ **先把连续的浮点特征值离散化成k个整数，同时构造一个宽度为k的直方图。遍历数据时，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。**
| **GBDT 虽然是个强力的模型，但却有着一个致命的缺陷，**\ 不能用类似 mini
  batch
  的方式来训练，需要对数据进行无数次的遍历。如果想要速度，就需要把数据都预加载在内存中，但这样数据就会受限于内存的大小；如果想要训练更多的数据，就要使用外存版本的决策树算法\ **。虽然外存算法也有较多优化，SSD
  也在普及，**\ 但在频繁的 IO 下，速度还是比较慢的\ **。为了能让 GBDT
  高效地用上更多的数据，我们把思路转向了**\ 分布式 GBDT，
  然后就有了LightGBM**。

设计的思路主要是两点，

-  **单个机器在不牺牲速度的情况下，尽可能多地用上更多的数据**\ ；

-  多机并行的时候，\ **通信的代价尽可能地低，并且在计算上可以做到线性加速。**

基于这两个需求，LightGBM 选择了基于 histogram
的决策树算法。相比于另一个主流的算法 pre-sorted（如 xgboost 中的 exact
算法），histogram 在内存消耗和计算代价上都有不少优势。

Pre-sorted 算法需要的内存约是训练数据的两倍(2 \_ #data \_ #features\*
4Bytes)，它需要用32位浮点来保存 feature
value，并且对每一列特征，都需要一个额外的排好序的索引，这也需要32位的存储空间。

对于 histogram 算法，则只需要(#data\_ #features \_
1Bytes)的内存消耗，\ **仅为 pre-sorted算法的1/8**\ 。因为 **histogram
算法仅需要存储 featurebin value (离散化后的数值)，不需要原始的 feature
value，也不用排序，而 binvalue 用 uint8_t (256bins)
的类型一般也就足够了。**

| 在计算上的优势则主要体现在“数据分割”。\ **决策树算法有两个主要操作组成，一个是“寻找分割点”，另一个是“数据分割”。**
| \*\*

-  从算法时间复杂度来看，Histogram 算法和 pre-sorted
   算法在“寻找分割点”的代价是一样的，都是O(#feature*#data)。

-  | 而在“数据分割”时，pre-sorted 算法需要O(#feature*#data)，而
     histogram 算法是O(#data)。因为 pre-sorted
     算法的每一列特征的顺序都不一样，分割的时候需要对每个特征单独进行一次分割。\ **Histogram算法不需要排序，所有特征共享同一个索引表，分割的时候仅需对这个索引表操作一次就可以**\ 。（更新:
     这一点不完全正确，pre-sorted 与 level-wise
     结合的时候，其实可以共用一个索引表(row_idx_to_tree_node_idx)。然后在寻找分割点的时候，同时操作同一层的节点，省去分割的步骤。但这样做的问题是会有非常多随机访问，有很大的chche
     miss，速度依然很慢）。
   |  另一个计算上的优势则是\ **大幅减少了计算分割点增益的次数**\ 。

-  对于一个特征，\ **pre-sorted
   需要对每一个不同特征值都计算一次分割增益**;

-  而 **histogram 只需要计算 #bin (histogram 的横轴的数量) 次。**

最后，\ **在数据并行的时候，用 histgoram 可以大幅降低通信代价。用
pre-sorted 算法的话，通信代价是非常大的（几乎是没办法用的）。所以
xgoobst 在并行的时候也使用 histogram 进行通信**\ 。

histogram算法缺点

**histogram 算法也有缺点，它不能找到很精确的分割点，训练误差没有
pre-sorted 好**\ 。但从实验结果来看， histogram 算法在测试集的误差和
pre-sorted
算法差异并不是很大，甚至有时候效果更好。\ **实际上可能决策树对于分割点的精确程度并不太敏感，而且较“粗”的分割点也自带正则化的效果。**

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137735714-c82baf97-d820-4f83-8ea1-3a4112505be0.png#align=left&display=inline&height=253&originHeight=253&originWidth=557&size=0&status=done&width=557#align=left&display=inline&height=253&originHeight=253&originWidth=557&status=done&width=557
   :alt: 

| 使用直方图算法有很多优点。*\*
  首先，最明显就是内存消耗的降低，直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用
  8 位整型存储就足够了，内存消耗可以降低为原来的1/8**。
| |image1|
| 然后在计算上的代价也大幅降低，预排序算法每遍历一个特征值就需要计算一次分裂的增益，而直方图算法只需要计算k次（k可以认为是常数），时间复杂度从O(#data\ *#feature)优化到O(k*\ #features)。
| 当然，Histogram
  算法并不是完美的。由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但在不同的数据集上的结果表明，离散化的分割点对最终的精度影响并不是很大，甚至有时候会更好一点。

   原因是决策树本来就是弱模型，分割点是不是精确并不是太重要；较粗的分割点也有正则化的效果，可以有效地防止过拟合；即使单棵树的训练误差比精确分割的算法稍大，但在梯度提升（Gradient
   Boosting）的框架下没有太大的影响。

.. _142-带深度限制的-leaf-wise-的叶子生长策略:

1.4.2 带深度限制的 Leaf-wise 的叶子生长策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 Histogram 算法之上，LightGBM 进行进一步的优化。首先它抛弃了大多数
GBDT 工具使用的按层生长 (level-wise)
的决策树生长策略，而使用了带有深度限制的按叶子生长 (leaf-wise)
算法。Level-wise
过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上
Level-wise
是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564710919791-ffb53669-ed67-4998-9572-8108fa135cc3.png#align=left&display=inline&height=253&originHeight=253&originWidth=640&size=0&status=done&width=640#align=left&display=inline&height=253&originHeight=253&originWidth=640&status=done&width=640
   :alt: 

Leaf-wise
则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同
Level-wise 相比，在分裂次数相同的情况下，Leaf-wise
可以降低更多的误差，得到更好的精度。Leaf-wise
的缺点是可能会长出比较深的决策树，产生过拟合。因此 LightGBM 在 Leaf-wise
之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564710919814-995b586c-bf4c-476b-8711-9bae8d2bee8e.png#align=left&display=inline&height=223&originHeight=223&originWidth=640&size=0&status=done&width=640#align=left&display=inline&height=223&originHeight=223&originWidth=640&status=done&width=640
   :alt: 

.. _143-直方图加速:

1.4.3 直方图加速
~~~~~~~~~~~~~~~~

LightGBM 另一个优化是
Histogram（直方图）做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。利用这个方法，LightGBM
可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137736159-40ecc782-d67a-476e-bce8-20faf4cc7430.png#align=left&display=inline&height=185&originHeight=230&originWidth=928&size=0&status=done&width=746#align=left&display=inline&height=230&originHeight=230&originWidth=928&status=done&width=928
   :alt: 

.. _144-直接支持类别特征:

1.4.4 直接支持类别特征
~~~~~~~~~~~~~~~~~~~~~~

实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化到多维的0/1
特征，降低了空间和时间的效率。而类别特征的使用是在实践中很常用的。基于这个考虑，LightGBM
优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1
展开。并在决策树算法上增加了类别特征的决策规则。在 Expo
数据集上的实验，相比0/1 展开的方法，训练速度可以加速 8
倍，并且精度一致。据我们所知，LightGBM 是第一个直接支持类别特征的 GBDT
工具。

LightGBM 的单机版本还有很多其他细节上的优化，比如 cache
访问优化，多线程优化，稀疏特征优化等等。优化汇总如下：

.. figure:: https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1553137736435-88814b5a-134f-4a66-8533-6c20199dadd1.jpeg#align=left&display=inline&height=341&originHeight=494&originWidth=1080&size=0&status=done&width=746#align=left&display=inline&height=494&originHeight=494&originWidth=1080&status=done&width=1080
   :alt: 

.. _145-lightgbm并行优化:

1.4.5 LightGBM并行优化
~~~~~~~~~~~~~~~~~~~~~~

LightGBM 还具有支持高效并行的优点。LightGBM
原生支持并行学习，目前支持特征并行和数据并行的两种。

-  特征并行的主要思想是在不同机器在不同的特征集合上分别寻找最优的分割点，然后在机器间同步最优的分割点。

-  数据并行则是让不同的机器先在本地构造直方图，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点。

LightGBM 针对这两种并行方法都做了优化：

-  在特征并行算法中，通过在本地保存全部数据避免对数据切分结果的通信；

-  在数据并行中使用分散规约 (Reduce scatter)
   把直方图合并的任务分摊到不同的机器，降低通信和计算，并利用直方图做差，进一步减少了一半的通信量。基于投票的数据并行则进一步优化数据并行中的通信代价，使通信代价变成常数级别。在数据量很大的时候，使用投票并行可以得到非常好的加速效果。

特征并行 feature
^^^^^^^^^^^^^^^^

-  传统算法

传统的特征并行算法旨在于在并行化决策树中的“ Find Best
Split.主要流程如下:

-  垂直划分数据（不同的机器有不同的特征集）

-  在本地特征集寻找最佳划分点 {特征, 阈值}

-  本地进行各个划分的通信整合并得到最佳划分

-  以最佳划分方法对数据进行划分，并将数据划分结果传递给其他线程

-  其他线程对接受到的数据进一步划分

传统的特征并行方法主要不足:

存在计算上的局限，传统特征并行无法加速 “split”（时间复杂度为
“O（#data）”）。
因此，当数据量很大的时候，难以加速。需要对划分的结果进行通信整合，其额外的时间复杂度约为
“O（#data/8）”（一个数据一个字节）

-  LightGBM 中的特征并行

既然在数据量很大时，传统数据并行方法无法有效地加速，我们做了一些改变：\ **不再垂直划分数据，即每个线程都持有全部数据。
因此，LighetGBM中没有数据划分结果之间通信的开销，各个线程都知道如何划分数据**\ 。
而且，“#data” 不会变得更大，所以，在使每台机器都持有全部数据是合理的。

LightGBM 中特征并行的流程如下：

-  每个线程都在本地数据集上寻找最佳划分点｛特征， 阈值｝

-  本地进行各个划分的通信整合并得到最佳划分

-  执行最佳划分

然而，\ **特征并行算法在数据量很大时仍然存在计算上的局限。因此，建议在数据量很大时使用数据并行。**

**数据并行 data**
^^^^^^^^^^^^^^^^^

-  传统算法

数据并行旨在于并行化整个决策学习过程。数据并行的主要流程如下：

-  水平划分数据

-  线程以本地数据构建本地直方图

-  将本地直方图整合成全局整合图

-  在全局直方图中寻找最佳划分，然后执行此划分

传统数据划分的不足：

-  高通讯开销。 如果使用点对点的通讯算法，一个机器的通讯开销大约为
   “O(#machine #feature #bin)”

-  如果使用集成的通讯算法（例如， “All Reduce”等），通讯开销大约为 “O(2
   #feature #bin)”

-  LightGBM中的数据并行

LightGBM 中采用以下方法较少数据并行中的通讯开销：

-  不同于“整合所有本地直方图以形成全局直方图”的方式，LightGBM
   使用分散规约(Reduce
   scatter)的方式对不同线程的不同特征（不重叠的）进行整合。

-  然后线程从本地整合直方图中寻找最佳划分并同步到全局的最佳划分中。

-  如上所述。\ **LightGBM 通过直方图做差法加速训练。
   基于此，我们可以进行单叶子的直方图通讯，并且在相邻直方图上使用做差法。**

-  通过上述方法，LightGBM 将数据并行中的通讯开销减少到 “O(0.5 #feature
   #bin)”。

投票并行
^^^^^^^^

投票并行未来将致力于将
**“数据并行”中的通讯开销减少至常数级别，其将会通过两阶段的投票过程较少特征直方图的通讯开销。**\ <

.. figure:: https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1553137736551-16480dfe-20e9-4539-a050-35b8602c19ca.jpeg#align=left&display=inline&height=509&originHeight=640&originWidth=938&size=0&status=done&width=746#align=left&display=inline&height=640&originHeight=640&originWidth=938&status=done&width=938
   :alt: 

.. figure:: https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1553137736645-950afbbc-744b-4261-8370-952e00a4e827.jpeg#align=left&display=inline&height=303&originHeight=438&originWidth=1080&size=0&status=done&width=746#align=left&display=inline&height=438&originHeight=438&originWidth=1080&status=done&width=1080
   :alt: 

.. figure:: https://cdn.nlark.com/yuque/0/2019/jpeg/200056/1553137736944-2d775098-484c-42f8-b478-374841cf7524.jpeg#align=left&display=inline&height=389&originHeight=563&originWidth=1080&size=0&status=done&width=746#align=left&display=inline&height=563&originHeight=563&originWidth=1080&status=done&width=1080
   :alt: 

.. _146-网络通信优化:

1.4.6 网络通信优化
~~~~~~~~~~~~~~~~~~

XGBoost由于采用pre-sorted算法，通信代价非常大，所以在并行的时候也是采用histogram算法；LightGBM采用的histogram算法通信代价小，通过使用集合通信算法，能够实现并行计算的线性加速。

.. _15-其他注意:

1.5 其他注意
------------

-  当生长相同的叶子时，Leaf-wise 比 level-wise 减少更多的损失。

-  高速，高效处理大数据，运行时需要更低的内存，支持 GPU

-  不要在少量数据上使用，会过拟合，建议 10,000+ 行记录时使用。

.. _2-lightgbm代码:

2 lightGBM代码
==============

.. _21-基础代码:

2.1 基础代码
------------

.. code:: python

   # 01. train set and test set 划分训练集和测试集
   train_data = lgb.Dataset(dtrain[predictors],label=dtrain[target],feature_name=list(dtrain[predictors].columns), categorical_feature=dummies)

   test_data = lgb.Dataset(dtest[predictors],label=dtest[target],feature_name=list(dtest[predictors].columns), categorical_feature=dummies)

   # 02. parameters 参数设置
   param = {
       'max_depth':6,
       'num_leaves':64,
       'learning_rate':0.03,
       'scale_pos_weight':1,
       'num_threads':40,
       'objective':'binary',
       'bagging_fraction':0.7,
       'bagging_freq':1,
       'min_sum_hessian_in_leaf':100
   }

   param['is_unbalance']='true'
   param['metric'] = 'auc'

   #03. cv and train 自定义cv函数和模型训练
   bst=lgb.cv(param,train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)

   estimators = lgb.train(param,train_data,num_boost_round=len(bst['auc-mean']))

   #04. test predict 测试集结果
   ypred = estimators.predict(dtest[predictors])

.. _22-模板代码:

2.2 模板代码
------------

.. _221-二分类:

2.2.1 二分类
~~~~~~~~~~~~

.. code:: python

   import lightgbm as lgb  
   import pandas as pd  
   import numpy as np  
   import pickle  
   from sklearn.metrics import roc_auc_score  
   from sklearn.model_selection import train_test_split  

   print("Loading Data ... ")  

   # 导入数据  
   train_x, train_y, test_x = load_data()  

   # 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置  
   X, val_X, y, val_y = train_test_split(  
       train_x,  
       train_y,  
       test_size=0.05,  
       random_state=1,  
       stratify=train_y # 这里保证分割后y的比例分布与原数据一致  
   )  

   X_train = X  
   y_train = y  
   X_test = val_X  
   y_test = val_y  

   # create dataset for lightgbm  
   lgb_train = lgb.Dataset(X_train, y_train)  
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
   # specify your configurations as a dict  
   params = {  
       'boosting_type': 'gbdt',  
       'objective': 'binary',  
       'metric': {'binary_logloss', 'auc'},  #二进制对数损失
       'num_leaves': 5,  
       'max_depth': 6,  
       'min_data_in_leaf': 450,  
       'learning_rate': 0.1,  
       'feature_fraction': 0.9,  
       'bagging_fraction': 0.95,  
       'bagging_freq': 5,  
       'lambda_l1': 1,    
       'lambda_l2': 0.001,  # 越小l2正则程度越高  
       'min_gain_to_split': 0.2,  
       'verbose': 5,  
       'is_unbalance': True  
   }  

   # train  
   print('Start training...')  
   gbm = lgb.train(params,  
                   lgb_train,  
                   num_boost_round=10000,  
                   valid_sets=lgb_eval,  
                   early_stopping_rounds=500)  

   print('Start predicting...')  

   preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

   # 导出结果  
   threshold = 0.5  
   for pred in preds:  
       result = 1 if pred > threshold else 0  

   # 导出特征重要性  
   importance = gbm.feature_importance()  
   names = gbm.feature_name()  
   with open('./feature_importance.txt', 'w+') as file:  
       for index, im in enumerate(importance):  
           string = names[index] + ', ' + str(im) + '\n'  
           file.write(string)

.. _222-多分类:

2.2.2 多分类
~~~~~~~~~~~~

.. code:: python

   import lightgbm as lgb  
   import pandas as pd  
   import numpy as np  
   import pickle  
   from sklearn.metrics import roc_auc_score  
   from sklearn.model_selection import train_test_split  

   print("Loading Data ... ")  

   # 导入数据  
   train_x, train_y, test_x = load_data()  

   # 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置  
   X, val_X, y, val_y = train_test_split(  
       train_x,  
       train_y,  
       test_size=0.05,  
       random_state=1,  
       stratify=train_y ## 这里保证分割后y的比例分布与原数据一致  
   )  

   X_train = X  
   y_train = y  
   X_test = val_X  
   y_test = val_y  


   # create dataset for lightgbm  
   lgb_train = lgb.Dataset(X_train, y_train)  
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
   # specify your configurations as a dict  
   params = {  
       'boosting_type': 'gbdt',  
       'objective': 'multiclass',  
       'num_class': 9,  
       'metric': 'multi_error',  
       'num_leaves': 300,  
       'min_data_in_leaf': 100,  
       'learning_rate': 0.01,  
       'feature_fraction': 0.8,  
       'bagging_fraction': 0.8,  
       'bagging_freq': 5,  
       'lambda_l1': 0.4,  
       'lambda_l2': 0.5,  
       'min_gain_to_split': 0.2,  
       'verbose': 5,  
       'is_unbalance': True  
   }  

   # train  
   print('Start training...')  
   gbm = lgb.train(params,  
                   lgb_train,  
                   num_boost_round=10000,  
                   valid_sets=lgb_eval,  
                   early_stopping_rounds=500)  

   print('Start predicting...')  

   preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

   # 导出结果  
   for pred in preds:  
       result = prediction = int(np.argmax(pred))  

   # 导出特征重要性  
   importance = gbm.feature_importance()  
   names = gbm.feature_name()  
   with open('./feature_importance.txt', 'w+') as file:  
       for index, im in enumerate(importance):  
           string = names[index] + ', ' + str(im) + '\n'  
           file.write(string)

.. _23-lightgbm-和-xgboost-的代码比较:

2.3 lightGBM 和 xgboost 的代码比较
----------------------------------

.. _231-划分训练集测试集:

2.3.1 划分训练集测试集
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   dtrain = xgb.DMatrix(x_train,label=y_train)
   dtest = xgb.DMatrix(x_test)


   # lightgbm
   train_data = lgb.Dataset(x_train,label=y_train)

.. _232-设置参数:

2.3.2 设置参数
~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   parameters = {
       'max_depth':7, 
       'eta':1, 
       'silent':1,
       'objective':'binary:logistic',
       'eval_metric':'auc',
       'learning_rate':.05}

   # lightgbm
   param = {
       'num_leaves':150, 
       'objective':'binary',
       'max_depth':7,
       'learning_rate':.05,
       'max_bin':200}
   param['metric'] = ['auc', 'binary_logloss']

.. _233-模型训练:

2.3.3 模型训练
~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   num_round = 50
   from datetime import datetime 
   start = datetime.now() 
   xg = xgb.train(parameters,dtrain,num_round) 
   stop = datetime.now()

   # lightgbm
   num_round = 50
   start = datetime.now()
   lgbm = lgb.train(param,train_data,num_round)
   stop = datetime.now()

.. _234-模型执行时间:

2.3.4 模型执行时间
~~~~~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   execution_time_xgb = stop - start 
   execution_time_xgb

   # lightgbm
   execution_time_lgbm = stop - start
   execution_time_lgbm

.. _235-模型测试:

2.3.5 模型测试
~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   ypred = xg.predict(dtest) 
   ypred

   # lightgbm
   ypred2 = lgbm.predict(x_test)
   ypred2[0:5]

.. _236-分类转换:

2.3.6 分类转换
~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   for i in range(0,9769): 
       if ypred[i] >= .5:       # setting threshold to .5 
          ypred[i] = 1 
       else: 
          ypred[i] = 0

   # lightgbm
   for i in range(0,9769):
       if ypred2[i] >= .5:       # setting threshold to .5
          ypred2[i] = 1
       else:  
          ypred2[i] = 0

.. _237-准确率计算:

2.3.7 准确率计算
~~~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   from sklearn.metrics import accuracy_score 
   accuracy_xgb = accuracy_score(y_test,ypred) 
   accuracy_xgb

   # lightgbm
   accuracy_lgbm = accuracy_score(ypred2,y_test)
   accuracy_lgbm
   y_test.value_counts()
   from sklearn.metrics import roc_auc_score

.. _238-rocaucscore计算:

2.3.8 roc_auc_score计算
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   #xgboost
   auc_xgb =  roc_auc_score(y_test,ypred)

   # lightgbm
   auc_lgbm = roc_auc_score(y_test,ypred2)

最后可以建立一个 dataframe 来比较 Lightgbm 和 xgb:

.. code:: python

   auc_lgbm comparison_dict = {
       'accuracy score':(accuracy_lgbm,accuracy_xgb),
       'auc score':(auc_lgbm,auc_xgb),
       'execution time':(execution_time_lgbm,execution_time_xgb)}

   comparison_df = DataFrame(comparison_dict) 
   comparison_df.index= ['LightGBM','xgboost'] 
   comparison_df

.. _3-lightgbm调参:

3 lightGBM调参
==============

LightGBM 垂直地生长树，即 leaf-wise，它会选择最大 delta loss
的叶子来增长。

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564710919791-ffb53669-ed67-4998-9572-8108fa135cc3.png#align=left&display=inline&height=253&originHeight=253&originWidth=640&size=0&status=done&width=640#align=left&display=inline&height=253&originHeight=253&originWidth=640&status=done&width=640#align=left&display=inline&height=253&originHeight=253&originWidth=640&status=done&width=640
   :alt: 

而以往其它基于树的算法是水平地生长，即 level-wise，

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564710919814-995b586c-bf4c-476b-8711-9bae8d2bee8e.png#align=left&display=inline&height=223&originHeight=223&originWidth=640&size=0&status=done&width=640#align=left&display=inline&height=223&originHeight=223&originWidth=640&status=done&width=640#align=left&display=inline&height=223&originHeight=223&originWidth=640&status=done&width=640
   :alt: 

当生长相同的叶子时，\ **Leaf-wise 比 level-wise
减少更多的损失。**\ 高速，高效处理大数据，运行时需要更低的内存，支持
GPU<**不要在少量数据上使用，会过拟合，建议 10,000+ 行记录时使用。**

.. _31-参数:

3.1 参数
--------

.. _331-控制参数:

3.3.1 控制参数
~~~~~~~~~~~~~~

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137914447-a17e7176-bf98-430b-b739-fcd3b3bea538.png#align=left&display=inline&height=423&name=image.png&originHeight=423&originWidth=962&size=68429&status=done&width=962#align=left&display=inline&height=423&originHeight=423&originWidth=962&status=done&width=962
   :alt: 

.. _332-核心参数:

3.3.2 核心参数
~~~~~~~~~~~~~~

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137930660-8de7333c-9d66-47ed-a9eb-f79a21f42223.png#align=left&display=inline&height=462&name=image.png&originHeight=462&originWidth=964&size=60718&status=done&width=964#align=left&display=inline&height=462&originHeight=462&originWidth=964&status=done&width=964
   :alt: 

.. _333-io参数:

3.3.3 IO参数
~~~~~~~~~~~~

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137958967-3d279653-22f6-4a32-826d-26fba04302da.png#align=left&display=inline&height=195&name=image.png&originHeight=195&originWidth=961&size=31698&status=done&width=961#align=left&display=inline&height=195&originHeight=195&originWidth=961&status=done&width=961
   :alt: 

.. _32-调参:

3.2 调参
--------

**num_leaves取值应 ``<= 2 ^（max_depth）``\ ， 超过此值会导致过拟合
一般不超过128（max_depth=7）**

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553138042545-5e2d4603-21c8-432d-8026-aebbf4c6efa8.png#align=left&display=inline&height=689&name=image.png&originHeight=689&originWidth=965&size=89429&status=done&width=965#align=left&display=inline&height=689&originHeight=689&originWidth=965&status=done&width=965
   :alt: 

.. _33 lightgbm参数优化:

3.3 LightGBM参数优化
--------------------

1 针对 Leaf-wise (最佳优先) 树的参数优化

LightGBM 使用 ``leaf-wise``\ (``leaf-wise-best-first-tree-growth``)
的树生长策略, 而很多其他流行的算法采用 ``depth-wise`` 的树生长策略. 与
``depth-wise`` 的树生长策略相较, ``leaf-wise`` 算法可以收敛的更快. 但是,
如果参数选择不当的话, ``leaf-wise`` 算法有可能导致过拟合.

想要在使用 ``leaf-wise`` 算法时得到好的结果,
这里有几个重要的参数值得注意:

1. ``num_leaves``. 这是控制树模型复杂度的主要参数. 理论上, 借鉴
   depth-wise 树, 我们可以设置 ``num_leaves = 2^(max_depth)`` 但是,
   这种简单的转化在实际应用中表现不佳. 这是因为, 当叶子数目相同时,
   leaf-wise 树要比 depth-wise 树深得多, 这就有可能导致过拟合. 因此,
   当我们试着调整 ``num_leaves`` 的取值时, 应该让其小于
   ``2^(max_depth)``. 举个例子, 当 ``max_depth=6``
   时(这里译者认为例子中, 树的最大深度应为7), **depth-wise
   树可以达到较高的准确率.但是如果设置 ``num_leaves``\ 为 ``127`` 时,
   有可能会导致过拟合, 而将其设置为 ``70`` 或 ``80`` 时可能**\ 会得到比
   depth-wise 树更高的准确率.*\* 其实, ``depth`` 的概念在 leaf-wise
   树中并没有多大作用, 因为并不存在一个从 ``leaves`` 到 ``depth``
   的合理映射.*\*

2. ``min_data_in_leaf``. 这是处理 leaf-wise
   树的过拟合问题中一个非常重要的参数.
   它的\ **值取决于训练数据的样本个树和 ``num_leaves``**.
   将其\ **设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合.
   实际应用中, 对于大数据集, 设置其为几百或几千就足够了.**

3. ``max_depth``. 你也可以利用 ``max_depth`` 来显式地限制树的深度.

2 针对更快的训练速度

-  通过设置 ``bagging_fraction`` 和 ``bagging_freq`` 参数来使用 bagging
   方法

-  通过设置 ``feature_fraction`` 参数来使用特征的子抽样

-  使用较小的 ``max_bin``

-  使用 ``save_binary`` 在未来的学习过程对数据加载进行加速

-  使用并行学习

3 针对更好的准确率

-  使用较大的 ``max_bin`` （学习速度可能变慢）

-  使用较小的 ``learning_rate`` 和较大的 ``num_iterations``

-  使用较大的 ``num_leaves`` （可能导致过拟合）

-  使用更大的训练数据

-  尝试 ``dart``

4 处理过拟合

-  使用较小的 ``max_bin``

-  使用较小的 ``num_leaves``

-  使用 ``min_data_in_leaf`` 和 ``min_sum_hessian_in_leaf``

-  通过设置 ``bagging_fraction`` 和 ``bagging_freq`` 来使用 bagging

-  通过设置 ``feature_fraction`` 来使用特征子抽样

-  使用更大的训练数据

-  使用 ``lambda_l1``, ``lambda_l2`` 和 ``min_gain_to_split`` 来使用正则

-  尝试 ``max_depth`` 来避免生成过深的树

.. _34-lightgbm并行化进阶:

3.4 LightGBM并行化进阶
----------------------

.. _1-选择合适的并行算法:

1 选择合适的并行算法
~~~~~~~~~~~~~~~~~~~~

LightGBM 现已提供了以下并行学习算法.

====================== ========================
**Parallel Algorithm** **How to Use**
====================== ========================
Data parallel          ``tree_learner=data``
Feature parallel       ``tree_learner=feature``
Voting parallel        ``tree_learner=voting``
====================== ========================

这些算法适用于不同场景,如下表所示:

===================== ================== ==================
\                     **#data is small** **#data is large**
===================== ================== ==================
**#feature is small** Feature Parallel   Data Parallel
**#feature is large** Feature Parallel   Voting Parallel
===================== ================== ==================

.. _2-缺失值的处理:

2 缺失值的处理
~~~~~~~~~~~~~~

-  LightGBM 通过默认的方式来处理缺失值，你可以通过设置
   ``use_missing=false`` 来使其无效。

-  LightGBM 通过默认的的方式用 NA (NaN) 去表示缺失值，你可以通过设置
   ``zero_as_missing=true`` 将其变为零。

-  当设置 ``zero_as_missing=false`` （默认）时，在稀疏矩阵里
   (和LightSVM) ，没有显示的值视为零。

-  当设置 ``zero_as_missing=true`` 时， NA 和 0
   （包括在稀疏矩阵里，没有显示的值）视为缺失。

.. _3-类别特征的支持:

3 类别特征的支持
~~~~~~~~~~~~~~~~

-  **当直接输入类别特征，LightGBM 能提供良好的精确度**\ 。不像简单的
   one-hot 编码，LightGBM 可以找到类别特征的最优分割。 相对于 one-hot
   编码结果，LightGBM 可以提供更加准确的最优分割。

-  用 ``categorical_feature`` 指定类别特征

-  需要转换为 int 类型，并且只支持非负数。 建议转换到连续的数字范围。

-  使用 ``min_data_per_group``, ``cat_smooth`` 去处理过拟合（当
   ``#data`` 比较小，或者 ``#category`` 比较大）

-  对于类别数量很大的类别特征(``#category`` 比较大),
   最好把它转化为数值特征。

.. _4-lambdarank:

4 LambdaRank
~~~~~~~~~~~~

-  标签应该是 int
   类型，较大的数字代表更高的相关性（例如：0：坏，1：公平，2：好，3：完美）。

-  使用 ``label_gain`` 设置每个标签对应的增益（gain）。

-  使用 ``max_position`` 设置 NDCG 优化位置。

.. _4-lightgbm案例:

4 lightGBM案例
==============

.. _40-lightgbm- 和-xgboost-的代码比较:

4.0 lightGBM 和 xgboost 的代码比较
----------------------------------

.. code:: python

   import lightgbm as lgb
   import xgboost as xgb
   import pandas as pd

   train_data = pd.read_csv('./binary.train',header=None,sep = '\t')
   test_data = pd.read_csv('./binary.test',header=None,sep = '\t')

   x_train = train_data.drop(0,axis = 1).values
   x_test = test_data.drop(0,axis = 1).values
   y_train = train_data[0].values
   y_test = test_data[0].values

   # xgboost
   xgb_train_data = xgb.DMatrix(data=x_train,label=y_train)
   xgb_test_data = xgb.DMatrix(data=x_test)

   # lightgbm
   lgb_train_data = lgb.Dataset(data=x_train,label=y_train)
   lgb_test_data = lgb.Dataset(data=x_test,label=y_test,reference=lgb_train_data)

   # set parameters
   xgb_params = {
       'max_depth':7,
       'eta':1,
       'silent':1,
       'objective':'binary:logistic',
       'eval_metric':'auc',
       'learning_rate':0.05
   }

   lgb_params = {
       'num_leaves':150,
       'objective':'binary',
       'max_depth':7,
       'learning_rate':0.05,
       'max_bin':200
   }
   lgb_params['metric'] = ['auc', 'binary_logloss']

.. code:: python

   %time
   num_round = 50
   xgb = xgb.train(
       xgb_params,
       dtrain = xgb_train_data,
       num_boost_round = num_round
   )

   CPU times: user 2 µs, sys: 0 ns, total: 2 µs
   Wall time: 21.5 µs
       
       
       
   %time
   num_round = 50
   lgb = lgb.train(
       lgb_params,
       train_set = lgb_train_data,
       num_boost_round = num_round
   )

   CPU times: user 3 µs, sys: 0 ns, total: 3 µs
   Wall time: 8.34 µs

.. code:: python

   # xgboost
   for i in range(len(y_pred_xgb)): 
       if y_pred_xgb[i] >= .5:        # setting threshold to .5 
          y_pred_xgb[i] = 1 
       else: 
          y_pred_xgb[i] = 0
   # lightgbm
   for i in range(len(y_pred_lgb)):    
       if y_pred_lgb[i] >= .5:       # setting threshold to .5
          y_pred_lgb[i] = 1
       else:  
          y_pred_lgb[i] = 0

.. code:: python

   # Converting probabilities into 1 or 0
   from sklearn.metrics import accuracy_score
   from sklearn.metrics import roc_auc_score

   # xgboost
   accuracy_xgb = accuracy_score(y_test,y_pred_xgb) 
   print(accuracy_xgb)

   # lightgbm
   accuracy_lgb = accuracy_score(y_test,y_pred_lgb)
   print(accuracy_lgb)

   # xgboost
   auc_xgb = roc_auc_score(y_test,y_pred_xgb)
   print(auc_xgb)

   # lightgbm
   auc_lgb = roc_auc_score(y_test,y_pred_lgb)
   print(auc_lgb)

   0.756
   0.744
   0.7572884416924665
   0.7455495356037152

.. code:: python

   # 建立一个 dataframe 来比较 Lightgbm 和 xgb
   comparison_dict = {
       'accuracy score':(accuracy_lgb,accuracy_xgb),
       'auc score':(auc_lgb,auc_xgb)
   }
   comparison_df = pd.DataFrame(comparison_dict) 
   comparison_df.index= ['lightgbm','xgboost'] 
   comparison_df

.. figure:: https://cdn.nlark.com/yuque/0/2019/png/200056/1564736209655-d2d215dc-b8d9-4671-bbde-00ae514e845e.png#align=left&display=inline&height=88&name=image.png&originHeight=88&originWidth=280&size=5740&status=done&width=280#align=left&display=inline&height=88&originHeight=88&originWidth=280&status=done&width=280
   :alt: 

.. _41-回归案例:

4.1 回归案例
------------

data来源：LightGBM包自带\ `data <https://github.com/Microsoft/LightGBM/tree/master/examples>`__

.. _411-代码:

4.1.1 代码
~~~~~~~~~~

.. code:: python

   import json
   import lightgbm as lgb
   import pandas as pd
   from sklearn.metrics import roc_auc_score
   path="D:/data/"
   print("load data")
   df_train=pd.read_csv(path+"regression.train.csv",header=None,sep='\t')
   df_test=pd.read_csv(path+"regression.train.csv",header=None,sep='\t')
   y_train = df_train[0].values
   y_test = df_test[0].values
   X_train = df_train.drop(0, axis=1).values
   X_test = df_test.drop(0, axis=1).values
   # create dataset for lightgbm
   lgb_train = lgb.Dataset(X_train, y_train)
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
   # specify your configurations as a dict
   params = {
           'task': 'train',
           'boosting_type': 'gbdt',
           'objective': 'binary',
           'metric': {'l2', 'auc'},
           'num_leaves': 31,
           'learning_rate': 0.05,
           'feature_fraction': 0.9,
           'bagging_fraction': 0.8,
           'bagging_freq': 5,
           'verbose': 0
           }
   print('Start training...')
   # train
   gbm = lgb.train(params,
                   lgb_train,
                   num_boost_round=20,
                   valid_sets=lgb_eval,
                   early_stopping_rounds=5)
   print('Save model...')
   # save model to file
   gbm.save_model(path+'lightgbm/model.txt')
   print('Start predicting...')
   # predict
   y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
   # eval
   print(y_pred)
   print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
   print('Dump model to JSON...')
   # dump model to json (and save to file)
   model_json = gbm.dump_model()
   with open(path+'lightgbm/model.json', 'w+') as f:
       json.dump(model_json, f, indent=4)
   print('Feature names:', gbm.feature_name())
   print('Calculate feature importances...')
   # feature importances
   print('Feature importances:', list(gbm.feature_importance()))

.. _412-运行结果:

4.1.2 运行结果
~~~~~~~~~~~~~~

.. code:: python

   load data
   Start training...
   [1] valid_0's auc: 0.76138  valid_0's l2: 0.243849
   Training until validation scores don't improve for 5 rounds.
   [2] valid_0's auc: 0.776568 valid_0's l2: 0.239689
   [3] valid_0's auc: 0.797394 valid_0's l2: 0.235903
   [4] valid_0's auc: 0.804646 valid_0's l2: 0.231545
   [5] valid_0's auc: 0.807803 valid_0's l2: 0.22744
   [6] valid_0's auc: 0.811241 valid_0's l2: 0.224042
   [7] valid_0's auc: 0.817447 valid_0's l2: 0.221105
   [8] valid_0's auc: 0.819344 valid_0's l2: 0.217747
   [9] valid_0's auc: 0.82034  valid_0's l2: 0.214645
   [10]    valid_0's auc: 0.821408 valid_0's l2: 0.211794
   [11]    valid_0's auc: 0.823175 valid_0's l2: 0.209131
   [12]    valid_0's auc: 0.824161 valid_0's l2: 0.206662
   [13]    valid_0's auc: 0.824834 valid_0's l2: 0.204433
   [14]    valid_0's auc: 0.825996 valid_0's l2: 0.20245
   [15]    valid_0's auc: 0.826775 valid_0's l2: 0.200595
   [16]    valid_0's auc: 0.827877 valid_0's l2: 0.198727
   [17]    valid_0's auc: 0.830383 valid_0's l2: 0.196703
   [18]    valid_0's auc: 0.833477 valid_0's l2: 0.195037
   [19]    valid_0's auc: 0.834914 valid_0's l2: 0.193249
   [20]    valid_0's auc: 0.836136 valid_0's l2: 0.191544
   Did not meet early stopping. Best iteration is:
   [20]    valid_0's auc: 0.836136 valid_0's l2: 0.191544
   Save model...
   Start predicting...
   [ 0.63918719  0.74876927  0.7446886  ...,  0.27801888  0.47378265
     0.49893381]
   The roc of prediction is: 0.836136144322
   Dump model to JSON...
   Feature names: ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Column_11', 'Column_12', 'Column_13', 'Column_14', 'Column_15', 'Column_16', 'Column_17', 'Column_18', 'Column_19', 'Column_20', 'Column_21', 'Column_22', 'Column_23', 'Column_24', 'Column_25', 'Column_26', 'Column_27']
   Calculate feature importances...
   Feature importances: [25, 4, 4, 41, 7, 56, 4, 1, 4, 29, 5, 4, 1, 20, 8, 10, 0, 7, 3, 10, 1, 21, 59, 7, 66, 77, 55, 71]

.. _42-icc竞赛-精品旅行服务成单预测:

4.2 [ICC竞赛] 精品旅行服务成单预测
----------------------------------

比赛说明：精品旅行服务成单预测

.. _421-业务需求:

4.2.1 业务需求
~~~~~~~~~~~~~~

提供了5万多名用户在旅游app中的浏览行为记录，其中有些用户在浏览之后完成了订单，且享受了精品旅游服务，而有些用户则没有下单。参赛者需要分析用户的个人信息和浏览行为，从而预测用户是否会在短期内购买精品旅游服务。预测用户是否会在短期内购买精品旅游服务。

.. _422-数据表格:

4.2.2 数据表格
~~~~~~~~~~~~~~

（1）数据整体描述：

数据包含5万多名用户的个人信息，以及他们上百万条的浏览记录和相应的历史订单记录，还包含有用户对历史订单的评论信息。

这些用户被随机分为2组，80%作为训练集，20%作为测试集。

两组数据的处理方式和内容类型是一致的，唯一不同的就是测试集中不提供需要预测的订单类型（即是否有购买精品旅游服务）。

（2）数据详细描述：

(a)用户个人信息：userProfile_.csv\ *\* （*\ 表示train或者test，下同）

数据共有四列，分别是用户id、性别、省份、年龄段。注：信息会有缺失。

.. code:: 

   例如： userid,gender,province,age

    100000000127,,上海,

    100000000231,男,北京,70后

(b)用户行为信息：action_*.csv*\*

数据共有三列，分别是用户id，行为类型，发生时间。

.. code:: 

   例如： userid,actionType,actionTime

    100000000111,1,1490971433

    100000000111,5,1490971446

    100000000111,6,1490971479

    100000000127,1,1490695669

    100000000127,5,1490695821

行为类型一共有9个，其中1是唤醒app；29则是有先后关系的，从填写表单到提交订单再到最后支付。

注意：数据存在一定的缺失！

(c)用户历史订单数据：orderHistory_*.csv*\*

该数据描述了用户的历史订单信息。数据共有7列，分别是用户id，订单id，订单时间，订单类型，旅游城市，国家，大陆。其中1表示购买了精品旅游服务，0表示普通旅游服务。

.. code:: 

   例如： userid,orderid,orderTime,orderType,city,country,continent

    100000000371, 1000709,1503443585,0,东京,日本,亚洲

    100000000393, 1000952,1499440296,0,巴黎,法国,欧洲

注意：一个用户可能会有多个订单，需要预测的是用户最近一次订单的类型；此文件给到的订单记录都是在“被预测订单”之前的记录信息！同一时刻可能有多个订单，属于父订单和子订单的关系。

(d)待预测订单的数据：orderFuture_*.csv*\*

对于train，有两列，分别是用户id和订单类型。供参赛者训练模型使用。其中1表示购买了精品旅游服务，0表示未购买精品旅游服务（包括普通旅游服务和未下订单）。

.. code:: 

   例如： userid,orderType

    102040050111,0

    103020010127,1

    100002030231,0

对于test，只有一列用户id，是待预测的用户列表。

(e)评论数据：userComment_*.csv*\*

共有5个字段，分别是用户id，订单id，评分，标签，评论内容。

其中受数据保密性约束，评论内容仅显示一些关键词。

.. code:: 

    userid,orderid,rating,tags,commentsKeyWords

    100000550471, 1001899,5.0,,

    10044000637, 1001930,5.0,主动热情|提前联系|景点介绍详尽|耐心等候,

    111333446057, 1001960,5.0,主动热情|耐心等候,[‘平稳’, ‘很好’]

.. _423-lightgbm模型:

4.2.3 lightGBM模型
~~~~~~~~~~~~~~~~~~

.. code:: python

   # -*- coding:utf-8 -*- 

   from __future__ import print_function
   from __future__ import division

   from data_helper import *

   import lightgbm as lgb
   from sklearn.model_selection import train_test_split
   import time
   import logging.handlers

   """Train the lightGBM model."""

   LOG_FILE = 'log/lgb_train.log'
   check_path(LOG_FILE)
   handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
   fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
   formatter = logging.Formatter(fmt)
   handler.setFormatter(formatter)
   logger = logging.getLogger('train')
   logger.addHandler(handler)
   logger.setLevel(logging.DEBUG)


   def lgb_fit(config, X_train, y_train):
       """模型（交叉验证）训练，并返回最优迭代次数和最优的结果。
       Args:
           config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
           X_train：array like, shape = n_sample * n_feature
           y_train:  shape = n_sample * 1
       Returns:
           best_model: 训练好的最优模型
           best_auc: float, 在测试集上面的 AUC 值。
           best_round: int, 最优迭代次数。
       """
       params = config.params
       max_round = config.max_round
       cv_folds = config.cv_folds
       early_stop_round = config.early_stop_round
       seed = config.seed
       # seed = np.random.randint(0, 10000)
       save_model_path = config.save_model_path
       if cv_folds is not None:
           dtrain = lgb.Dataset(X_train, label=y_train)
           cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                              metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
           # 最优模型，最优迭代次数
           best_round = len(cv_result['auc-mean'])
           best_auc = cv_result['auc-mean'][-1]  # 最好的 auc 值
           best_model = lgb.train(params, dtrain, best_round)
       else:
           X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
           dtrain = lgb.Dataset(X_train, label=y_train)
           dvalid = lgb.Dataset(X_valid, label=y_valid)
           watchlist = [dtrain, dvalid]
           best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
           best_round = best_model.best_iteration
           best_auc = best_model.best_score
           cv_result = None
       if save_model_path:
           check_path(save_model_path)
           best_model.save_model(save_model_path)
       return best_model, best_auc, best_round, cv_result


   def lgb_predict(model, X_test, save_result_path=None):
       y_pred_prob = model.predict(X_test)
       if save_result_path:
           df_result = df_future_test
           df_result['orderType'] = y_pred_prob
           df_result.to_csv(save_result_path, index=False)
           print('Save the result to {}'.format(save_result_path))
       return y_pred_prob


   class Config(object):
       def __init__(self):
           self.params = {
               'objective': 'binary',
               'metric': {'auc'},
               'learning_rate': 0.05,
               'num_leaves': 30,  # 叶子设置为 50 线下过拟合严重
               'min_sum_hessian_in_leaf': 0.1,
               'feature_fraction': 0.3,  # 相当于 colsample_bytree
               'bagging_fraction': 0.5,  # 相当于 subsample
               'lambda_l1': 0,
               'lambda_l2': 5,
               'num_thread': 6  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
           }
           self.max_round = 3000
           self.cv_folds = 5
           self.early_stop_round = 30
           self.seed = 3
           self.save_model_path = 'model/lgb.txt'


   def run_feat_search(X_train, X_test, y_train, feature_names):
       """根据特征重要度，逐个删除特征进行训练，获取最好的特征结果。
       同时，将每次迭代的结果求平均作为预测结果"""
       config = Config()
       # train model
       tic = time.time()
       y_pred_list = list()
       aucs = list()
       for i in range(1, 250, 3):
           drop_cols = feature_names[-i:]
           X_train_ = X_train.drop(drop_cols, axis=1)
           X_test_ = X_test.drop(drop_cols, axis=1)
           data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train_.shape, X_test_.shape)
           print(data_message)
           logger.info(data_message)
           lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train_, y_train)
           print('Time cost {}s'.format(time.time() - tic))
           result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
           logger.info(result_message)
           print(result_message)

           # predict
           # lgb_model = lgb.Booster(model_file=config.save_model_path)
           now = time.strftime("%m%d-%H%M%S")
           result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
           check_path(result_path)
           y_pred = lgb_predict(lgb_model, X_test_, result_path)
           y_pred_list.append(y_pred)
           aucs.append(best_auc)
           y_preds_path = 'stack_preds/lgb_feat_search_pred_{}.npz'.format(i)
           check_path(y_preds_path)
           np.savez(y_preds_path, y_pred_list=y_pred_list, aucs=aucs)
           message = 'Saved y_preds to {}. Best auc is {}'.format(y_preds_path, np.max(aucs))
           logger.info(message)
           print(message)


   def run_cv(X_train, X_test, y_train):
       config = Config()
       # train model
       tic = time.time()
       data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
       print(data_message)
       logger.info(data_message)
       lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train, y_train)
       print('Time cost {}s'.format(time.time() - tic))
       result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
       logger.info(result_message)
       print(result_message)
       # predict
       # lgb_model = lgb.Booster(model_file=config.save_model_path)
       now = time.strftime("%m%d-%H%M%S")
       result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
       check_path(result_path)
       lgb_predict(lgb_model, X_test, result_path)


   if __name__ == '__main__':
       # get feature
       feature_path = 'features/'
       train_data, test_data = load_feat(re_get=True, feature_path=feature_path)
       train_feats = train_data.columns.values
       test_feats = test_data.columns.values
       drop_columns = list(filter(lambda x: x not in test_feats, train_feats))
       X_train = train_data.drop(drop_columns, axis=1)
       y_train = train_data['label']
       X_test = test_data
       data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
       print(data_message)
       logger.info(data_message)

       # 根据特征搜索中最好的结果丢弃部分特征
       # n_drop_col = 141
       # drop_cols = feature_names[-n_drop_col:]
       # X_train = X_train.drop(drop_cols, axis=1)
       # X_test = X_test.drop(drop_cols, axis=1)
       # 直接训练
       run_cv(X_train, X_test, y_train)

       # 特征搜索
       # get feature scores
       # try:
       #     df_lgb_feat_score = pd.read_csv('features/lgb_features.csv')
       #     feature_names = df_lgb_feat_score.feature.values
       # except Exception as e:
       #     print('You should run the get_no_used_features.py first.')
       # run_feat_search(X_train, X_test, y_train, feature_names)

**注意**\ ：该案例还使用了XGboost和catBoost模型，以及其他特征提取方法，在此不详述。数据+模型见\ `github <https://github.com/yongyehuang/DC-hi_guides>`__

https://github.com/yongyehuang/DC-hi_guides

.. _5-lightgbm的坑:

5 lightGBM的坑
==============

.. _51-设置提前停止:

5.1 设置提前停止
----------------

如果在训练过程中启用了提前停止，可以用
bst.best_iteration从最佳迭代中获得预测结果：

``ypred = bst.predict(data,num_iteration = bst.best_iteration )``

.. _52-自动处理类别特征:

5.2 自动处理类别特征
--------------------

-  当使用本地分类特征，LightGBM能提供良好的精确度。不像简单的one-hot编码，lightGBM可以找到分类特征的最优分割。

-  用categorical_feature指定分类特征

-  首先需要转换为int类型，并且只支持非负数。转换为连续范围更好。

-  使用min_data_per_group，cat_smooth去处理过拟合（当#data比较小，或者#category比较大）

-  对于具有高基数的分类特征（#category比较大），最好转换为数字特征。

.. _53-自动处理缺失值:

5.3 自动处理缺失值
------------------

-  lightGBM通过默认方式处理缺失值，可以通过设置use_missing = false
   来使其无效。

-  lightGBM通过默认的方式用NA（NaN）去表示缺失值，可以通过设置zero_as_missing
   = true 将其变为0

-  当设置zero_as_missing =
   false（默认）时，在稀疏矩阵里（和lightSVM），没有显示的值视为0

-  当设置zero_as_missing =
   true时，NA和0（包括在稀疏矩阵里，没有显示的值）视为缺失。

.. |image1| image:: https://cdn.nlark.com/yuque/0/2019/png/200056/1553137735686-3bc562fb-0f9c-42b2-bd5b-aeb5cce84250.png#align=left&display=inline&height=232&originHeight=311&originWidth=1000&size=0&status=done&width=746#align=left&display=inline&height=311&originHeight=311&originWidth=1000&status=done&width=1000
