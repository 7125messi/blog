===================
LightGBM在Spark实战
===================

:Date:   2021-04-26T22:19:13+08:00

通常业务中对计算性能有要求时，通常不使用GPU跑tf，会使用xgboost/lightgbm
on Spark来解决，既保证速度，准确率也能接受。

LightGBM是使用基于树的学习算法的梯度增强框架。它被设计为分布式且高效的，具有以下优点：

根据官网的介绍

-  LigthGBM训练速度更快，效率更高。LightGBM比XGBoost快将近10倍。

-  降低内存使用率。内存占用率大约为XGBoost的1/6。

-  准确性有相应提升。

-  支持并行和GPU学习。

-  能够处理大规模数据。

大部分使用和分析LigthGBM的都是在python单机版本上。要在spark上使用LigthGBM，需要安装微软的MMLSpark包。

MMLSpark可以通过--packages安装。

spark

-  --packages参数

根据jar包的maven地址，使用该包，该参数不常用，因为公司内部的数据平台的集群不一定能联网。
如下示例：

.. code:: shell

   $ bin/spark-shell --packages  com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1 http://maven.aliyun.com/nexus/content/groups/public/

-  --repositories 为该包的maven地址，建议给定，不写则使用默认源。
   若依赖多个包，则中间以逗号分隔，类似--jars
   默认下载的包位于当前用户根目录下的.ivy/jars文件夹中
   应用场景：本地没有编译好的jar包，集群中服务需要该包的的时候，都是从给定的maven地址，直接下载

MMLSpark用法
============

1 .MMLSpark可以通--packages选项方便地安装在现有的Spark集群上，例如:

.. code:: shell

   spark-shell --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1
   pyspark --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1
   spark-submit --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1

这也可以在其他Spark
contexts中使用，例如，可以通过将MMLSpark添加到.aztk/spark-default.conf文件中来在AZTK中使用MMLSpark。

2 .要在Python(或Conda)安装上尝试MMLSpark，首先通过pip安装PySpark,
pip安装PySpark。接下来，使用--package或在运行时添加包来获取scala源代码

.. code:: python

   import pyspark
   spark = pyspark.sql.SparkSession.builder.appName("MyApp").\
   		config("spark.jars.packages","com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1").\
   		getOrCreate()

   import mmlspark

3.python建模使用

.. code:: python

   # 分类
   from mmlspark.lightgbm import LightGBMClassifier
   model = LightGBMClassifier(learningRate=0.3,
                              numIterations=100,
                              numLeaves=31).fit(train)
                              
   # 回归
   from mmlspark.lightgbm import LightGBMRegressor
   model = LightGBMRegressor(application='quantile',
                             alpha=0.3,
                             learningRate=0.3,
                             numIterations=100,
                             numLeaves=31).fit(train)

LightGBM on Spark项目应用
=========================

PySpark编写
-----------

.. code:: python

   from pyspark import SparkContext
   from pyspark.sql import SparkSession
   from pyspark.sql import functions as fn
   from pyspark.ml import feature as ft
   from mmlspark.lightgbm import LightGBMRegressor
   import psycopg2
   import uuid
   import datetime
   import sys

   from model_utils import categorical_features_encoding_transform, ENC_FEATURE_LIST, CATEGORICAL_FEATURES, FEATURES
   from spark_db_utils import read_dataset, psycopg_execute, fetch_gsc_data, write_dataset
   from spark_read_conf import get_spark_conf, get_conn_url, get_user_pwd, get_config, get_conn_info

   def train_lgb_model(categorical_features, vec_df, params_list, store_group, sku_group, str_objective, model_dict):
       """
       训练模型并且存储到HDFS上
       
       Parameters
       ----------
           categorical_features: list
               类别型变量集合
           vec_df: pyspark.sql.DataFrame
               通过VectorAssembler处理的特征向量(训练集)
           params_list: list
               超参集合（目前主要是alpha参数以及对应预测值列名）
           str_model_unit： str
               模型拼接的名称
       """
       # 遍历所有分位数类型
       model_row_list = []

       # 模型名称
       str_model_unit = f'''{store_group}_{sku_group}'''

       for param in params_list:
           alpha, col = param
           
           # lgb on spark
           objective = str_objective if alpha == 0.5 else 'quantile'
           lgb = LightGBMRegressor(objective=objective,
                                   alpha=alpha,
                                   metric='12',
                                   learningRate=0.06,
                                   baggingFraction=1.0,
                                   baggingFreq=6,
                                   featureFraction=1.0,
                                   maxDepth=5,
                                   numIterations=500,
                                   numLeaves=32,
                                   labelCol='sale_qty',
                                   categoricalSlotNames=categorical_features,
                                   predictionCol=col)
           # 模型训练
           model = lgb.fit(vec_df)

           # 当前时间
           str_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
           str_time_ms = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
           # 模型存放路径
           str_model_name = f'''hdfs:///user/xxx/forecast_model/{str_model_unit}_{col}_{str_time}'''
           # 模型存储
           model.write().overwrite().save(str_model_name)
           
           if model_dict.__contains__(f'{store_group}_{sku_group}_{col}'):
               previous_model_id = model_dict[f'{store_group}_{sku_group}_{col}']
           else:
               previous_model_id = '0'

           if alpha == 0.5:
               model_row = (str(uuid.uuid1()), previous_model_id,
                           'FORECAST_SKU', 'SKU_STORE_GROUP', f'''V{str_time_ms}''',
                            str_model_name, 'active', store_group, sku_group,
                            f'{store_group}_{sku_group}_{col}', f'{store_group}_{sku_group}_{col}')
           else:
               model_row = (str(uuid.uuid1()), previous_model_id,
                            'FORECAST_SKU', 'SKU_STORE_GROUP', f'''V{str_time_ms}''',
                            str_model_name, 'active_reference', store_group, sku_group,
                            f'{store_group}_{sku_group}_{col}', f'{store_group}_{sku_group}_{col}')

           model_row_list.append(model_row)

       return model_row_list

spark-submit提交任务跑模型
--------------------------

.. code:: shell

   # !/bin/bash

   current_dir=$(cd $(dirname $0); pwd)
   cd ${current_dir}

   source /etc/profile
   source ../global_batchid.sh
   source ../global_config.sh
   source ../tools/execute_gp_sql.sh

   bash /root/update_kerberos.sh

   function train_lgb_model()
   {
       cd /data/xxx/train_lgb_model
       
       /opt/spark-2.4.4/bin/spark-submit \
       --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1,com.microsoft.ml.lightgbm:lightgbmlib:2.3.100 \
       --driver-class-path ./greenplum-spark_2.11-1.6.2.jar \
       --jars ./greenplum-spark_2.11-1.6.2.jar train_food_category.py $END_DATE
       
       if [[ $? -ne 0 ]]; then
           echo "--> execute train_lgb_model failed!"
       exit 1
       fi
   }
   train_lgb_model
