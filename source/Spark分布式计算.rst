===============
Spark分布式计算
===============

:Date:   2022-01-14T10:49:07+08:00

[项目经验总结]

从事数据相关工作，最喜欢用的工具就是基于Pandas、Jupyter
Lab等工具，拿到样本数据，单机上快速迭代试验验证想法，这确实很方便，但是等到模型部署上线的时候，数据量很大，很难单机就搞定，目前主流的做法是用Spark分布式计算解决。

但是如果利用纯 PySpark API，就需要将Pandas
API重写成PySpark的API，虽然很多API很类似，但是多少有些不一样，而且有些逻辑用用Pandas生态很容易实现，而利用PySpark却很复杂，遇到PySpark没有的API，动辄就要写UDF函数了，所以实际生产部署的时候，如果采用此方式，改造成本会有点高。

有没有简单的方法？

我们知道通常Spark也是作为客户端，使用Hadoop的YARN作为集群的资源管理和调度器。Spark集群由Driver,
Cluster Manager（Standalone,Yarn 或 Mesos），以及Worker
Node组成。对于每个Spark应用程序，Worker
Node上存在一个Executor进程，Executor进程中包括多个Task线程。对于PySpark,为了不破坏Spark已有的运行时架构，Spark在外围包装一层Python
API。在Driver端，借助Py4j实现Python和Java的交互，进而实现通过Python编写Spark应用程序。在Executor端，则不需要借助Py4j，因为Executor端运行的Task逻辑是由Driver发过来的，那是序列化后的字节码。

Spark运行流程

-  Application首先被Driver构建DAG图并分解成Stage。

-  然后Driver向Cluster Manager申请资源。

-  Cluster Manager向某些Work Node发送征召信号。

-  被征召的Work Node启动Executor进程响应征召，并向Driver申请任务。

-  Driver分配Task给Work Node。

-  Executor以Stage为单位执行Task，期间Driver进行监控。

-  Driver收到Executor任务完成的信号后向Cluster Manager发送注销信号。

-  Cluster Manager向Work Node发送释放资源信号。

-  Work Node对应Executor停止运行。

所以简单的做法跑PySparkr任务时利用YARN的分发机制，将可以并行计算的任务同时分发到不同Work
Node计算，然后每个节点则利用由原来的Pandas API计算即可。

.. code:: python

   import sys
   import calendar
   from typing import Tuple,List
   import pandas as pd
   import numpy as np
   from sklearn import linear_model

   from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType
   from pyspark.sql import functions as F
   from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, broadcast
   from pyspark.sql import Row
   from pyspark.sql.dataframe import DataFrame
   from pyspark.sql.window import Window

   from foo.utils.conn_utils import SparkInit, spark_write_to_hive


   def sales_diff_mom_feature(df, unit):
       diff_list = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (0, 12), (8, 12), (11, 12)]
       for (i, j) in diff_list:
           df[f'{target}_{unit}_diff_{i}_{j}'] = df[f'{target}_{unit}_lag_{i}'] - df[f'{target}_{unit}_lag_{j}']
           df[f'{target}_{unit}_mom_{i}_{j}'] = (df[f'{target}_{unit}_lag_{i}'] + 1) / (df[f'{target}_{unit}_lag_{j}'] + 1)

       return df


   def sales_rolling_feature(df, window, unit):
       columns = [f'{target}_{unit}_lag_{i}' for i in range(window)]
       df[f'{target}_{unit}_lag_rolling_{window}_mean'] = df[columns].mean(axis=1).astype(np.float32)
       df[f'{target}_{unit}_lag_rolling_{window}_std'] = df[columns].std(axis=1).astype(np.float32)
       df[f'{target}_{unit}_lag_rolling_{window}_max'] = df[columns].max(axis=1).astype(np.float32)
       df[f'{target}_{unit}_lag_rolling_{window}_min'] = df[columns].min(axis=1).astype(np.float32)
       df[f'{target}_{unit}_lag_rolling_{window}_median'] = df[columns].median(axis=1).astype(np.float32)

       return df


   def value_sku_agg_rate_lag_feature(df, aggregation, lags):
       df[f'value_sku-{"_".join(aggregation)}_rate'] = df['value'] / df.groupby(aggregation)['value'].transform('sum')
       for lag in lags:
           df[f'value_sku-{"_".join(aggregation)}_rate_lag_{lag}'] = df.groupby([division, 'material_code'])[f'value_sku-{"_".join(aggregation)}_rate'].shift(lag).astype(np.float16)
       df = df.drop([f'value_sku-{"_".join(aggregation)}_rate'], axis=1)

       return df


   def get_days_of_month(days_dict, year_month):
       if year_month not in days_dict:
           year = int(str(year_month)[:4])
           month = int(str(year_month)[4:6])
           days = calendar.monthrange(year, month)[1]
           days_dict[year_month] = days

       return days_dict[year_month]


   def time_features(df, division):
       df['year'] = df['year_month'].apply(lambda x: int(str(x)[:4])).astype('category')
       df['month'] = df['year_month'].apply(lambda x: int(str(x)[4:6])).astype('category')
       df['on_market_month'] = (df['month_idx'] - df['min_month_idx'] + 1).astype(np.int8)
       df['on_market_day_first_month'] = df.apply(lambda x: get_days_of_month({}, x['year_month']) - int(str(x['mindate'])[6:8]) + 1, axis=1)
       df[f'on_market_month_{division}'] = (df['month_idx'] - df[f'min_month_idx_{division}_sku'] + 1).astype(np.int8)

       return df


   def sales_rolling_mean_feature_level(df, window, target, level):
       columns = [f'{target}_material_code_lag_{i}' for i in range(window)]
       df2 = pd.DataFrame(df.groupby([level, 'year_month'])[columns].sum()).reset_index()
       df2[f'{target}_lag_rolling_{window}_mean_{level}'] = df2[columns].mean(axis=1).astype(np.float32)
       df2 = df2.rename(columns={f'{target}_material_code_lag_{i}': f'{target}_{level}_lag_{i}' for i in range(window)})

       return df2[[level, 'year_month', f'{target}_lag_rolling_{window}_mean_{level}', f'{target}_{level}_lag_0']]


   def latest_sale_proportion_feature(df, window, target, level):
       df2 = sales_rolling_mean_feature_level(df, window, target, level)
       df2[f'latest_{window}_{target}_proportion_{level}'] = df2[f'{target}_{level}_lag_0'] / df2[f'{target}_lag_rolling_{window}_mean_{level}']
       df = pd.merge(df, df2[[level, 'year_month', f'latest_{window}_{target}_proportion_{level}']], on=[level, 'year_month'], how='left')
       return df


   def get_trend_by_lr(x, df_all, unit, window, target):
       current_month = x['month_idx']
       sku = x[unit]
       start_month = current_month - window
       condition = (df_all['month_idx'] >= start_month) & (df_all['month_idx'] <= current_month) & (df_all[unit] == sku)
       train_data = df_all.loc[condition, [target, 'month_idx']].drop_duplicates()

       coef = np.nan
       if len(train_data):
           model = linear_model.LinearRegression()
           model.fit(train_data['month_idx'].to_numpy().reshape(-1, 1), train_data[target])
           coef = model.coef_[0]
       return coef


   def add_trend_feature(df, unit, window, target):
       df_unit = df.loc[df['on_market_month'] > 3].groupby([unit, 'month_idx'])[target].sum().reset_index()
       df_unit = df_unit.dropna(axis=0, subset=[target])
       if df_unit.empty:
           return df
       try:
           df_unit[f'trend_slope_{unit}_{window}'] = df_unit.apply(lambda x: get_trend_by_lr(x, df_unit, unit, window, target), axis=1)
           df = pd.merge(df, df_unit[[unit, 'month_idx', f'trend_slope_{unit}_{window}']], on=[unit, 'month_idx'], how='left')
       except Exception as e:
           print(e)

       return df


   def feature_engineer(df, calendar, division, target):
       df = df.sort_values(['material_code', division, 'year_month'])
       unit_list = ['material_code']

       for unit in unit_list:
           for lag in range(13):
               df[f'{target}_{unit}_lag_{lag}'] = df.groupby([division, 'material_code'])[target].shift(lag)
           df = sales_diff_mom_feature(df, unit)
       print('sales_lag_feature finished!')
       print('sales_lag_feature diff & mon finished!')

       window_list = [2, 3, 6, 9, 12]
       for unit in unit_list:
           for window in window_list:
               df = sales_rolling_feature(df, window, unit)
       print('sales_rolling_feature finished!')

       agg_list = [['year_month', 'material_code']]
       for agg in agg_list:
           df = value_sku_agg_rate_lag_feature(df, agg, range(6))
       print('value_sku-aggregation_rate_feature finished!')

       df = time_features(df, division)
       print('time_features finished!')

       for level in ['material_code', 'sub_brand', 'category']:
           df = latest_sale_proportion_feature(df, 12, target, level)
       print("latest_sale_proportion_feature finished!")

       for unit in ['material_code', 'store']:
           for window in [3, 4, 5, 6, 9]:
               df = add_trend_feature(df, unit, window, target)
       print("trend feature finished!")

       cat_columns = ['month', 'material_code', 'brand', 'sub_brand', 'franchise', 'category', 'series', 'signature',
                      'area', 'axe', 'sub_axe', 'class', 'function_id', 'mstatus', division, 'sales_level',
                      'level3', 'level5', 'level6', 'level5_6', 'level3_5_6', 'oj1_brand', 'l2_label',
                      f'{division}_values_level', f'{division}_level', 'seasonality_flag', 'sku_type', 'Franchise', 'citycode', 'line_city', 'prvnname_ch', 'regionname_ch', 'area', 'nation']
       for column in cat_columns:
           if column in df.columns:
               df[column] = df[column].astype('category')
       print('category_feature finished!')

       for lag in range(1, 16):
           df[f'target_m_{lag}'] = df.groupby([division, 'material_code'])[target].shift(-lag).astype(np.float32)
       print('define Y finished!')

       for lag in range(1, 16):
           df[f'active_sku_filter_m_{lag}'] = df.groupby([division, 'material_code'])['filter'].shift(-lag).astype(np.float32)
           df[f'active_sku_filter_m_{lag}'] = df[f'active_sku_filter_m_{lag}'].fillna(0)
       df = df.drop('filter', axis=1)
       print('active_sku_filter finished!')

       column_list = df.columns.to_list()
       column_list.remove('brand')
       column_list.remove('category')
       df = df[column_list + ['brand', 'category']]
       
       return df


   def make_feature_engineer(rows, calendar_b, division, target):
       """
           groupbyKey -- category
       :param rows:
       :param calendar_b:
       :param division:
       :param target:
       :return:
       """
       row_list = list()
       for row in rows:
           row_list.append(row.asDict())
       df = pd.DataFrame(row_list)
       
       # 广播变量的值
       calendar = calendar_b.value

       df = feature_engineer(df, calendar, division, target)

       dfRow = Row(*df.columns)
       row_list = []
       for r in df.values:
           row_list.append(dfRow(*r))

       return row_list


   def spark_dis_com(spark, processor_data, calendar_b, division, target, repartition, parallel_column):
       #### distributed compute
       feature_data_rdd = processor_data.rdd. \
           map(lambda x: (x[parallel_column], x)). \
           groupByKey(). \
           flatMap(lambda x: make_feature_engineer(x[1], calendar_b, division, target))

       #### write table
       spark_write_to_hive(
           spark.createDataFrame(feature_data_rdd.repartition(repartition)),
           'ldlgtm_dpt.ld_feature_store_bh'
       )



   if __name__ == '__main__':
       ############################# offline test #############################
       ###################### Configuring
       # ...
       # processor_data = pd.read_csv('...')
       # calendar = pd.read_csv('...')
       # feature_data = feature_engineer(processor_data, calendar, division, target)
       # feature_data.to_pickle('...')
       
       ############################# online pre/prd #############################
       ###################### Init Spark
       spark =  SparkInit(f'sf-app-gtm-art-fcsting-POS-LD')

       ###################### Configuring
       ...

       # 加载数据
       processor_data = spark.sql(f""" ... """)
       calendar = get_calendar_data(spark)
       
       # 广播变量
       calendar_b = spark.sparkContext.broadcast(calendar.toPandas())
       
       spark_dis_com(spark, processor_data, calendar_b, division, target, 24, 'category')
       print("feature_data is successful")
       print("feature_data write table successful")
       
       spark.stop()

.. _1-sdfrddmaplambda-x-xparallelcolumn-xgroupbykeyflatmaplambda-x-funcx1-var1bvarnb-var1--varn:

1 sdf.rdd.map(lambda x: (x[parallel_column], x)).groupByKey().flatMap(lambda x: func(x[1], var1_b,...,varn_b, var1, ..., varn))
===============================================================================================================================

以上述代码举例说明：

-  offline test 是在线下测试的代码，如函数 feature_engineer
   即是普通的基于Pandas API 的纯Python代码；

-  online pre/prd 是线上开发和生产环境的代码，可以看到函数
   make_feature_engineer 和 spark_com_dis 的代码对于 feature_engineer
   稍加改动就变成了分布式计算的代码，主要有以下几点：

   -  利用spark.sql 读取 hive表里存储的预处理好的数据
      processor_data（pyspark.sql.dataframe.DataFrame），基于processor_data做特征工程计算；

   -  processor_data 可以根据\ **某个字段或某几个字段**\ 做map分组分发；

   -  flatMap API
      根据make_feature_engineer函数做分布式计算，并将最后结果合并；

   -  make_feature_engineer 函数先将每个分组内的pyspark.sql.Row准成
      Python dict，再转成 list, 继而生成一个Pandas
      DataFrame,然后继续使用 feature_engineer
      函数计算，最后还原成由pyspark.sql.Row组成的list；

   -  利用spark.createDataFrame API 创建 Spark DataFrame 写表；

   -  对于其他的辅助变量 例如calendar_b，需要广播到各个节点。

可以看到，这里分布式计算较为灵活，可以根据\ **某个字段或某几个字段**\ （需要根据自己的数据需求）做map分组分发，比如这里我是根据
category 分组分发计算（我这里的数据必须根据每个category
训练模型），非常实用。

.. _2-rddflatmaplambda-x-funcvar1bvarnb-var1--varn-x:

2 rdd.flatMap(lambda x: func(var1_b,...,varn_b, var1, ..., varn, x))
====================================================================

当然有的时候我们还可以根据下面的方式分组分发，这里我是根据每个门店 store
做分发：

-  利用 spark.sparkContext.parallelize(store_list, 24) 生成
   rdd（使用已经存在的迭代器或者集合通过调用spark驱动程序提供的parallelize函数来创建并行集合，并行集合被创建用来在分布式集群上并行计算，这里的24表示将RDD切分多少个分区）;

-  对于其他变量进行广播，每组RDD内的store数据直接使用 flatMap API计算；

.. code:: python

   def map_make_post_process(model_output_b, calendar_b, odf_df_b, event_df_b, sales_df_pro_b, sku_delisting_df_b, forecast_list, end_date, M0, M1_5_list, store):
       ######## calendar & odp data & event data & ld_month_actual_event_normal_ratio data broadcast
       model_output = model_output_b.value
       calendar = calendar_b.value
       odp_df = odf_df_b.value
       event_df = event_df_b.value
       sales_df_pro = sales_df_pro_b.value
       sku_delisting_df = sku_delisting_df_b.value

       output_formated = model_output.loc[model_output['store'] == store]
       
       ...
       
       output_formatedRow = Row(*output_formated.columns)
       row_list = []
       for r in output_formated.values:
           row_list.append(output_formatedRow(*r))

       return row_list
       
   if __name__ == '__main__':
       # ......
       store_list = model_output['store'].unique()
       store_rdd = spark.sparkContext.parallelize(store_list, 24)
       output_formated_data_row_list = store_rdd.flatMap(
           lambda x: map_make_post_process(model_output_b, calendar_b, odp_df_b, event_df_b, sales_df_pro_b, sku_delisting_df_b, forecast_list, end_date, M0, M1_5_list, x)
       )
       output_formated_data_sdf = output_formated_data_row_list.toDF().repartition(24).persist()
       ......

.. _3-rddgroupby:

3 rdd.groupBy()
===============

该方法稍微复杂点，需要对着下面示例好好体会

先看个简单的分组案例：

.. code:: python

   rdd = spark.sparkContext.parallelize(['18039', '47839'])
   result = rdd.groupBy(lambda x: int(x) % 50).collect()

   sorted([(x, sorted(y)) for (x, y) in result])   

   #############################################
   [(39, ['18039', '47839'])]

.. code:: python

   paralizm_num = 50

   # sale_sdf.rdd 根据 global_store_number 与 paralizm_num 取余进行门店分组号[0,1,2,3,...,49]
   # error_dis_rdd 根据 global_store_number 与 paralizm_num 取余进行门店分组号[0,1,2,3,...,49]

   sale_sdf_g_rdd = sale_sdf.rdd.\
               groupBy(lambda x: int(x['global_store_number']) % paralizm_num, numPartitions=paralizm_num)

   error_dis_r_rdd = error_dis_rdd.\
               groupBy(lambda x: int(x[1]) % paralizm_num, numPartitions=paralizm_num)

.. code:: python

   sorted([(x, sorted(y)) for (x, y) in sale_sdf_g_rdd.take(1)])

   #############################################################
   [(0,
     [Row(global_store_number='15000', notional_item_cd='NON10151', business_day='2022-05-25', pred_sale_qty=0.0, restore_sale_qty=0.0, secondary_category='Sandwich & Wrap', row_number=1),
      Row(global_store_number='15000', notional_item_cd='NON10151', business_day='2022-05-26', pred_sale_qty=0.0, restore_sale_qty=0.0, secondary_category='Sandwich & Wrap', row_number=1),
      Row(global_store_number='15000', notional_item_cd='NON10151', business_day='2022-05-27', pred_sale_qty=0.0, restore_sale_qty=0.0, secondary_category='Sandwich & Wrap', row_number=1),
      Row(global_store_number='15300', notional_item_cd='NON319', business_day='2022-06-22', pred_sale_qty=2.0, restore_sale_qty=0.0, secondary_category='Sandwich & Wrap', row_number=1),
      Row(global_store_number='15300', notional_item_cd='NON319', business_day='2022-06-23', pred_sale_qty=2.0, restore_sale_qty=2.0, secondary_category='Sandwich & Wrap', row_number=1),
      Row(global_store_number='15300', notional_item_cd='NON319', business_day='2022-06-24', pred_sale_qty=3.0, restore_sale_qty=3.0, secondary_category='Sandwich & Wrap', row_number=1),
      ...])]

.. code:: python

   sorted([(x, sorted(y)) for (x, y) in error_dis_r_rdd.take(1)])

   #############################################################
   [(0,
     [('Bakery', '15000', -6.0    0.022851
       -5.0    0.038620
       -4.0    0.059333
       -3.0    0.082858
       -2.0    0.105180
       -1.0    0.121365
       -0.0    0.127296
        1.0    0.121365
        2.0    0.105180
        3.0    0.082858
        4.0    0.059333
        5.0    0.038620
        6.0    0.022851
        7.0    0.012290
       dtype: float64, 1),
      ('Bakery',
       '15300',
       -3.0    0.053838
       -2.0    0.121795
       -1.0    0.198771
       -0.0    0.234024
        1.0    0.198771
        2.0    0.121795
        3.0    0.053838
        4.0    0.017169
       dtype: float64,
       1),
      ....

.. code:: python

   # merge_rdd 是由sale_sdf_g_rdd 和 error_dis_r_rdd 根据 门店分组号[0,1,2,3,...,49] 关联
   # 得到每个门店分组号下的若干个门店的sale_sdf_sub 和 error_dis_rdd_sub
   # 该方法适合多个df 同时分发
   merge_rdd = sale_sdf_g_rdd.join(error_dis_r_rdd).repartition(paralizm_num)
   [(x, y) for (x, y) in merge_rdd.take(10)]

   #############################################################
   [(0,
     (<pyspark.resultiterable.ResultIterable at 0x7f0f719fc8d0>,
      <pyspark.resultiterable.ResultIterable at 0x7f0f5abff128>)
   )
   ...
   ]

.. code:: python

   def map_thaw_sim(...,x,...):
       store_group = x[0]
       cate_sale_df = create_dataframe_new(x[1][0])
       error_dist_df = pd.DataFrame(data=x[1][1], columns=error_dist_columns)
       ...
       row_list = create_row_list(cate_sale_df)
       sku_row_list = create_row_list(error_dist_df)
       return row_list, sku_row_list

   simulation_rdd = merge_rdd.map(lambda x: map_thaw_sim(...,x,...)).persist()
   store_res_row_list = simulation_rdd.flatMap(lambda x: x[0]).repartition(paralizm_num)
   store_res_sku_row_list = simulation_rdd.flatMap(lambda x: x[1]).repartition(paralizm_num)

   """
       x is a Tuple
       
       x[0]: store_group, 即门店分组号[0,1,2,3,...,49]
       x[1]: (cate_sale_df, error_dist_df) 某个门店分组号下对应的两个df
       
       [(0,
         (
               <pyspark.resultiterable.ResultIterable at 0x7f0f719fc8d0>,
               <pyspark.resultiterable.ResultIterable at 0x7f0f5abff128>
          )
       )]
   """
