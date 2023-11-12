最近在对一个机器学习项目进行代码重构，其中遇到了很多坑，故记录总结下，以便于后面查看学习。
这是一个美妆类机器学习销量预测项目，需要针对近 30 个渠道，100
个品牌的美妆产品，预测商品类别包括护肤品、彩妆、护发、男士护肤、香氛等，为这些
SKU 提供未来 1 到 15 个月的月维度的需求预测。

我们从以下几个方面进行的重构。

1 架构设计
==========

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699599134464-c9ef5a65-e127-4d79-847c-933ce355f71c.png#averageHue=%232b2a2a&clientId=u43825ebc-bcf0-4&from=ui&id=ub31ae45e&originHeight=980&originWidth=1342&originalType=binary&ratio=2&rotation=0&showTitle=false&size=125227&status=done&style=none&taskId=ub795551b-1f14-4713-8568-f7c00728c1b&title=
   :alt: 图片1.png

   图片1.png

这是原来的架构设计，可以看到对于每个渠道品牌品类启动一个线程去启动 Spark
执行代码。 这种架构存在的潜在问题和风险：

-  缺乏合理使用Spark分布式架构，容易出现运行过程不稳定报错任务失败、容错能力差、资源分配不均衡等问题；
-  Driver节点压力过载，有限线程数大量重复加载数据，To
   Pandas直接写入中间过程，导致Driver节点严重内存不足；
-  预处理、特征工程、模块训练及预测、模型后处理均由单线程处理，且为完全耦合式代码，无法debug中间过程；
-  每个线程反复读写公共数据，且均和MySQL数据库交互，导致性能低下；

为此，我们设计了以下的架构：

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699599480984-c5e84c5a-25ec-4bbd-be1c-a9b6a817ca39.png#averageHue=%233f3f3e&clientId=u43825ebc-bcf0-4&from=ui&id=ucfc87945&originHeight=933&originWidth=1126&originalType=binary&ratio=2&rotation=0&showTitle=false&size=82365&status=done&style=none&taskId=uc30e5934-b42d-4ade-b562-5000256251f&title=
   :alt: 图片2.png

   图片2.png

全新架构的优点：

-  合理使用 Spark 分布式架构数据过程全部使用 Delta
   Lake，涉及到的小表读入内存并广播给各个节点；
-  预处理、特征工程、模块训练及预测、模型后处理完全解耦抽象，方便对接
   Azure Databricks 平台及debug 中间过程；
-  每个节点负载均衡，协同作业；
-  全部模块处理完后统一把数据落在Delta Lake；

2 数据访问层分离
================

数据访问层不分离的缺点
----------------------

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699599786822-a775407c-0ff9-4ccf-b294-78fda87fc9d1.png#averageHue=%23e2e2e2&clientId=u43825ebc-bcf0-4&from=paste&height=360&id=ub38d66e4&originHeight=720&originWidth=809&originalType=binary&ratio=2&rotation=0&showTitle=false&size=2334487&status=done&style=none&taskId=u3d82ded6-d8da-4716-81f1-6180124a302&title=&width=404.5
   :alt: image.png

   image.png

如图所示：

-  SQL混在代码中，导致代码冗长, 一个方法几百行, 无法理解主体逻辑;
-  数据读取混在代码中，无法统一修改读取方式;
-  无法整体掌握算法依赖的所有数据

数据访问层分离的优点
--------------------

-  简化代码，增加可读性（大量SQL混在代码中不易读）；
-  添加条件查询，可以读取全量数据，也可以按条件过滤；
-  方便分布式改造，分离数据层，可以整体切换Pandas和Spark；
-  小表统一广播11个，大表join 、map

具体举例如下：

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600001760-752a354c-29c9-4e1c-b9cb-82f4bcb308d8.png#averageHue=%23dedcdb&clientId=u43825ebc-bcf0-4&from=paste&height=560&id=u48c6ed41&originHeight=1120&originWidth=2136&originalType=binary&ratio=2&rotation=0&showTitle=false&size=634548&status=done&style=none&taskId=u76bb2fd6-c22a-428f-b9a6-e1d1cd2e0fc&title=&width=1068
   :alt: image.png

   image.png

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600036537-443969fc-7e60-4c04-9ee1-95854443d8f1.png#averageHue=%23dfdfdf&clientId=u43825ebc-bcf0-4&from=paste&height=583&id=u3362dfc3&originHeight=1166&originWidth=1480&originalType=binary&ratio=2&rotation=0&showTitle=false&size=521699&status=done&style=none&taskId=ud5e2f78a-cb0a-4ac2-889d-2bc6d06e7c8&title=&width=740
   :alt: image.png

   image.png

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600058460-3fdeb764-18fa-472b-814b-cbab91b02db2.png#averageHue=%23b5b5b5&clientId=u43825ebc-bcf0-4&from=paste&height=160&id=uc1d93bc4&originHeight=320&originWidth=2034&originalType=binary&ratio=2&rotation=0&showTitle=false&size=97941&status=done&style=none&taskId=u47240372-f393-4027-bc6c-a5dbad18624&title=&width=1017
   :alt: image.png

   image.png

3 全量读取模型配置表参数表
==========================

只读取一个渠道下模型
--------------------

-  循环遍历渠道下模型
-  获取一个模型对应的参数
-  启动Spark 执行一个品牌（可以一个，可能多个）

全量读取模型表、模型参数表
--------------------------

-  模型库、预处理、后处理、模型全量读取
-  按照不同的键关联在一起
-  模型参数表读取全量

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600538614-7558c4aa-eb11-4d63-abe5-e648668ec9a5.png#averageHue=%23dcdbdb&clientId=u43825ebc-bcf0-4&from=paste&height=507&id=u470a0558&originHeight=1014&originWidth=2388&originalType=binary&ratio=2&rotation=0&showTitle=false&size=860251&status=done&style=none&taskId=u5e3eb824-d75b-4d73-a596-90f47ca5162&title=&width=1194
   :alt: image.png

   image.png

4 全量数据预处理
================

-  模型参数表，分成group_category为空，group_category不空的
-  与master表关联

   -  为group_category 为 null的 根据channel+brand关联；
   -  为group_category 不为 null的 根据channel+brand+category关联；

-  通过主数据过滤月销量数据和全国销量数据
-  根据training_dimension 字段生成分组字段

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600865719-3f924a09-7696-462f-8d61-f21cb2b6c340.png#averageHue=%23d2d2d2&clientId=u43825ebc-bcf0-4&from=paste&height=298&id=u7cbcc87c&originHeight=595&originWidth=1280&originalType=binary&ratio=2&rotation=0&showTitle=false&size=3052003&status=done&style=none&taskId=u49f42a48-4b09-4c1c-b19c-093f5b5795b&title=&width=640
   :alt: image.png

   image.png

5 整体流程模块化
================

-  整体流程模块化

   -  读取数据，获取全量数据集
   -  预处理原始数据，处理模型，主数据，关联销量数据
   -  算法数据预处理
   -  算法特征工程
   -  算法模型训练与预测
   -  算法预测结果后处理
   -  销量转销售额
   -  结果字段填充

-  每一步可以单独保存结果

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600942110-eb6446ee-e516-4525-a9c3-a9a188d08944.png#averageHue=%23e2e2e2&clientId=u43825ebc-bcf0-4&from=paste&height=360&id=u38c2b4cc&originHeight=720&originWidth=835&originalType=binary&ratio=2&rotation=0&showTitle=false&size=2409485&status=done&style=none&taskId=u060b41ce-709b-4ca0-b210-d2b918bb909&title=&width=417.5
   :alt: image.png

   image.png

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699600948513-d90714fe-7688-47c8-b934-80df13c25084.png#averageHue=%23dddada&clientId=u43825ebc-bcf0-4&from=paste&height=360&id=uab2ac41d&originHeight=720&originWidth=1092&originalType=binary&ratio=2&rotation=0&showTitle=false&size=3150847&status=done&style=none&taskId=uf45dfcc2-3d9a-49e6-aa82-12b7fa56439&title=&width=546
   :alt: image.png

   image.png

6 代码优化具体策略
==================

-  使用向量化操作；
-  避免使用循环，多使用Pandas内置的函数和向量化操作来处理数据
-  使用适当的数据类型，减少内存使用并提高代码的执行速度
-  使用Pandas内置的函数
-  删除不必要的列，减少内存使用和提高代码的执行速度
-  多进程训练模型

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699601092617-165b5fae-d275-482b-a74f-6b06ace26714.png#averageHue=%239a9a9a&clientId=ue3c56663-733a-4&from=paste&height=63&id=udc79c6fe&originHeight=126&originWidth=1280&originalType=binary&ratio=2&rotation=0&showTitle=false&size=646361&status=done&style=none&taskId=u9d827b7b-4fc4-4160-8639-8580fc01500&title=&width=640
   :alt: image.png

   image.png

.. figure:: https://cdn.nlark.com/yuque/0/2023/png/200056/1699601118826-6a4ebe10-2c86-45f1-a0b8-68b0874b6308.png#averageHue=%23cdcdcd&clientId=ue3c56663-733a-4&from=ui&id=u447dceea&originHeight=1036&originWidth=1728&originalType=binary&ratio=2&rotation=0&showTitle=false&size=355944&status=done&style=none&taskId=ua1397b95-4986-4109-8d87-a7d24a9b5bb&title=
   :alt: 图片3.png

   图片3.png
