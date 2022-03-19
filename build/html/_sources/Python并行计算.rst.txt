==============
Python并行计算
==============

:Date:   2021-10-16T14:34:09+08:00

[参考文献]

-  《Effective Python：编写高质量Python代码的90个有效方法（原书第2版）》

并发（concurrency）指计算机似乎能在同一时刻做许多件不同的事情。例如，在只配有一个CPU核心的计算机上面，操作系统可以迅速切换这个处理器所运行的程序，因此尽管同一时刻最多只有一个程序在运行，但这些程序能够交替地使用这个核心，从而造成一种假象，让人觉得它们好像真的在同时运行。

并行（parallelism）与并发的区别在于，它强调计算机确实能够在同一时刻做许多件不同的事情。例如，若计算机配有多个CPU核心，那么它就真的可以同时执行多个程序。每个CPU核心执行的都是自己的那个程序之中的指令，这些程序能够同时向前推进。

在同一个程序之中，我们可以利用并发轻松地解决某些类型的问题。例如，并发可以让程序里面出现多条独立的执行路径，每条路径都可以处理它自己的I/O流，这就让我们觉得这些I/O任务好像真的是在各自的路径里面同时向前推进的。

并行与并发之间的区别，关键在于能不能提速（speedup）。如果程序把总任务量分给两条独立的执行路径去同时处理，而且这样做确实能让总时间下降到原来的一半，那么这就是并行，此时的总速度是原来的两倍。反过来说，假如无法实现加速，那即便程序里有一千条独立的执行路径，也只能叫作并发，因为这些路径虽然看起来是在同时推进，但实际上却没有产生相应的提速效果。

Python让我们很容易就能写出各种风格的并发程序。在并发量较小的场合可以使用线程（thread），如果要运行大量的并发函数，那么可以使用协程（coroutine）。

并行任务，可以通过系统调用、子进程与C语言扩展（Cextension）来实现，但要写出真正能够并行的Python代码，其实是很困难的。

●
即便计算机具备多核的CPU，Python线程也无法真正实现并行，因为它们会受全局解释器锁（GIL）牵制。

●
虽然Python的多线程机制受GIL影响，但还是非常有用的，因为我们很容易就能通过多线程模拟同时执行多项任务的效果。

●
多条Python线程可以并行地执行多个系统调用，这样就能让程序在执行阻塞式的I/O任务时，继续做其他运算。

● 虽然Python有全局解释器锁，但开发者还是得设法避免线程之间发生数据争用。

●
把未经互斥锁保护的数据开放给多个线程去同时修改，可能导致这份数据的结构遭到破坏。

●
可以利用threading内置模块之中的Lock类确保程序中的固定关系不会在多线程环境下受到干扰。

● 程序范围变大、需求变复杂之后，经常要用多条路径平行地处理任务。

● fan-out与fan-in是最常见的两种并发协调（concurrency
coordination）模式，前者用来生成一批新的并发单元，后者用来等待现有的并发单元全部完工。（分派--归集）

● Python提供了很多种实现fan-out与fan-in的方案。

但是： ●
每次都手工创建一批线程，是有很多缺点的，例如：创建并运行大量线程时的开销比较大，每条线程的内存占用量比较多，而且还必须采用Lock等机制来协调这些线程。

●
线程本身并不会把执行过程中遇到的异常抛给启动线程或者等待该线程完工的那个人，所以这种异常很难调试。

.. _1-通过线程池-threadpoolexecutor-用多线程做并发提升有限io密集型）:

1 通过线程池 ThreadPoolExecutor 用多线程做并发（提升有限，I/O密集型）
=====================================================================

**Python有个内置模块叫作concurrent.futures，它提供了ThreadPoolExecutor类。**
**这个类结合了线程（Thread）方案与队列（Queue）方案的优势，可以用来平行地处理
I/O密集型操作。**

ThreadPoolExecutor方案仍然有个很大的缺点，就是I/O并行能力不高，即便把max_workers设成100，也无法高效地应对那种有一万多个单元格，且每个单元格都要同时做I/O的情况。如果你面对的需求，没办法用异步方案解决，而是必须执行完才能往后走（例如文件I/O），那么ThreadPoolExecutor是个不错的选择。然而在许多情况下，其实还有并行能力更强的办法可以考虑。

利用ThreadPoolExecutor，我们只需要稍微调整一下代码，就能够并行地执行简单的I/O操作，这种方案省去了每次fan-out（分派）任务时启动线程的那些开销。

虽然ThreadPoolExecutor不像直接启动线程的方案那样，需要消耗大量内存，但它的I/O并行能力也是有限的。因为它能够使用的最大线程数需要提前通过max_workers参数指定。

.. _2-通过线程池-processpoolexecutor-用多进程做并发iocpu密集型）:

2 通过线程池 ProcessPoolExecutor 用多进程做并发（I/O、CPU密集型）
=================================================================

从开发者这边来看，这个过程似乎很简单，但实际上，multiprocessing模块与
ProcessPoolExecutor类要做大量的工作才能实现出这样的并行效果。同样的效果，假如改用其他语言来做，那基本上只需要用一把锁或一项原子操作就能很好地协调多个线程，从而实现并行。\ **但这在Python里面不行，所以我们才考虑通过ProcessPoolExecutor来实现。然而这样做的开销很大，因为它必须在上级进程与子进程之间做全套的序列化与反序列化处理。这个方案对那种孤立的而且数据利用度较高的任务来说，比较合适。**

**●
所谓孤立（isolated），这里指每一部分任务都不需要跟程序里的其他部分共用状态信息。**
**●
所谓数据利用度较高（high-leverage），这里指任务所使用的原始材料以及最终所给出的结果数据量都很小，因此上级进程与子进程之间只需要互传很少的信息就行，然而在把原始材料加工成最终产品的过程中，却需要做大量运算。刚才那个求最大公约数的任务就属于这样的例子，当然还有很多涉及其他数学算法的任务，也是如此。**

如果你面对的计算任务不具备刚才那两项特征，那么使用ProcessPoolExecutor所引发的开销可能就会盖过因为并行而带来的好处。在这种情况下，我们可以考虑直接使用multiprocessing所提供的一些其他高级功能，例如共享内存（shared
memory）、跨进程的锁（cross-process
lock）、队列（queue）以及代理（proxy）等。但是，这些功能都相当复杂，即便两个Python线程之间所要共享的进程只有一条，也是要花很大工夫才能在内存空间里面将这些工具安排到位。假如需要共享的进程有很多条，而且还涉及socket，那么这种代码理解起来会更加困难。

总之，不要刚一上来，就立刻使用跟multiprocessing这个内置模块有关的机制，而是可以先试着用ThreadPoolExecutor来运行这种孤立且数据利用度较高的任务。把这套方案实现出来之后，再考虑向ProcessPoolExecutor方案迁移。如果ProcessPoolExecutor方案也无法满足要求，而且其他办法也全都试遍了，那么最后可以考虑直接使用multiprocessing模块里的高级功能来编写代码。

●
把需要耗费大量CPU资源的计算任务改用C扩展模块来写，或许能够有效提高程序的运行速度，同时又让程序里的其他代码依然能够利用Python语言自身的特性。但是，这样做的开销比较大，而且容易引入bug。

●
Python自带的multiprocessing模块提供了许多强大的工具，让我们只需要耗费很少的精力，就可以把某些类型的任务平行地放在多个CPU核心上面处理。要想发挥出multiprocessing模块的优势，最好是通过concurrent.futures模块及其ProcessPoolExecutor类来编写代码，因为这样做比较简单。

●
只有在其他方案全都无效的情况下，才可以考虑直接使用multiprocessing里面的高级功能（那些功能用起来相当复杂）。

.. _3-使用joblib并行运行python代码实际工程中比较好用）:

3 **使用Joblib并行运行Python代码**\ （实际工程中比较好用）
==========================================================

对于大多数问题，并行计算确实可以提高计算速度。
随着PC计算能力的提高，我们可以通过在PC中运行并行代码来简单地提升计算速度。\ `Joblib <https://link.zhihu.com/?target=https%3A//joblib.readthedocs.io/en/latest/>`__\ 就是这样一个可以简单地将P\ **ython代码转换为并行计算模式的软件包，它可非常简单并行我们的程序，从而提高计算速度。**

`Joblib <https://link.zhihu.com/?target=https%3A//joblib.readthedocs.io/en/latest/>`__\ 是一组用于在Python中提供轻量级流水线的工具。
它具有以下功能：

-  透明的磁盘缓存功能和“懒惰”执行模式，简单的并行计算

-  Joblib对numpy大型数组进行了特定的优化，简单，快速。

除了并行计算功能外，Joblib还具有以下功能：

-  快速磁盘缓存：Python函数的memoize或make-like功能，适用于任意Python对象，包括大型numpy数组。

-  快速压缩：替代pickle，使用joblib.dump和joblib.load可以提高大数据的读取和存储效率。

以下我们使用一个简单的例子来说明如何利用Joblib实现并行计算。
我们使用单个参数\ ``i``\ 定义一个简单的函数\ ``my_fun()``\ 。
此函数将等待1秒，然后计算\ ``i**2``\ 的平方根，也就是返回\ ``i``\ 本身。

.. code:: python

   from joblib import Parallel, delayed
   import time, math

   def my_fun(i):
       """ We define a simple function here.
       """
       time.sleep(1)
       return math.sqrt(i**2)

这里我们将总迭代次数设置为10.我们使用\ ``time.time()``\ 函数来计算\ ``my_fun()``\ 的运行时间。
如果使用简单的for循环，计算时间约为10秒。

.. code:: python

   num = 10
   start = time.time()
   for i in range(num):
       my_fun(i)

   end = time.time()

   print('{:.4f} s'.format(end-start))

   # 10.0387 s

使用Joblib中的\ ``Parallel``\ 和\ ``delayed``\ 函数，我们可以简单地配置\ ``my_fun()``\ 函数的并行运行。
其中我们会用到几个参数，\ ``n_jobs``\ 是并行作业的数量，我们在这里将它设置为\ ``2``\ 。
``i``\ 是\ ``my_fun()``\ 函数的输入参数，依然是10次迭代。两个并行任务给节约了大约一半的for循环运行时间，结果并行大约需要5秒。

.. code:: python

   start = time.time()
   # n_jobs is the number of parallel jobs
   Parallel(n_jobs=2)(delayed(my_fun)(i) for i in range(num))
   end = time.time()
   print('{:.4f} s'.format(end-start))

   # 5.6560 s

就是这么简单！ 如果我们的函数中有多个参数怎么办？ 也很简单。
让我们用两个参数定义一个新函数\ ``my_fun_2p(i,j)``\ 。

.. code:: python

   def my_fun_2p(i, j):
       """ We define a simple function with two parameters.
       """
       time.sleep(1)
       return math.sqrt(i**j)

   j_num = 3
   num = 10
   start = time.time()
   for i in range(num):
       for j in range(j_num):
           my_fun_2p(i, j)

   end = time.time()
   print('{:.4f} s'.format(end-start))

   # 30.0778 s

   start = time.time()
   # n_jobs is the number of parallel jobs
   Parallel(n_jobs=2)(delayed(my_fun_2p)(i, j) for i in range(num) for j in range(j_num))
   end = time.time()
   print('{:.4f} s'.format(end-start))

   # 15.0622 s

.. _4-案例介绍:

4 案例介绍
==========

-  这里用了偏函数，执行主函数 data_preprocessor，生成 偏函数
   data_preprocessor_p；

-  apply_parallel, 就是用Joblib定义的并行计算函数，目前支持 pandas
   dataframe的func根据分组后数据并行计算再归并。

-  对于其他数据类型，可以参考 parmap函数

.. code:: python

   from functools import partial

   from model.month.offline_lgb_city_model import LgbCityModel

   from utils.multi_processor import apply_parallel

   ......

   lgbmodel = LgbCityModel()

   # 数据预处理并保存文件
   def data_preprocessor(sales_month, lgbmodel, master_data, division, upper_division):
       data = lgbmodel._preproces(sales_month, master_data, division, upper_division)
       return data

   data_preprocessor_p = partial(data_preprocessor,
                                 lgbmodel=lgbmodel,
                                 master_data=master_data,
                                 division=division,
                                 upper_division=upper_division)
   processor_data = apply_parallel(sales_lgb.groupby('category'), data_preprocessor_p)
   processor_data = lgbmodel._memory_process(processor_data)
   processor_data.to_pickle(f'data/preprocessed/preprocessed_lgb_{division}{suffix}_{run_time}.pkl')

.. code:: python

   import multiprocessing
   import pandas as pd
   from joblib import Parallel, delayed

   def apply_parallel(df_grouped, func, n_jobs=3):
       """
       与上边不同的是，他直接传递给子进程 分片数据，而不是分片索引
       注意：该函数不是通用函数，只针对返回 pandas dataframe的func
       Parameters
       ----------
       df_grouped：分片数据列表
       func：表调用的函数

       Returns: dataframe
       -------

       """
       results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df_grouped)
       # 过滤空的结果
       rs = filter(lambda x: len(x) > 0, results)
       # 否则index 有重复
       res = pd.concat(rs).reset_index(drop=True)
       return res

   def fun(f, q_in, q_out):
       """
       从blocking queue_in 中获取数据
       把结果保存到 blocking queue_out
       Parameters
       ----------
       f
       q_in
       q_out
       Returns
       -------

       """
       while True:
           i, x = q_in.get()
           if i is None:
               break
           q_out.put((i, f(x)))


   def parmap(f, X, nprocs=20):
       """
       1.主进程将数据按照索引分片推送如队列 q_in，最后将结束符号推入队列。blocking
       2.工作进程进程从q_in取索引 调用回调方法;如果去除的数据是结束符，则进程结束
       3.每个进行运算结果放入结果队列q_out
       4.获取每个"分片"的返回数据,按照传入的数据排序，然后返回
       注意：windows 下不能运行
       Parameters
       ----------
       f ：回调函数，每个子进程调用这个函数，传入索引分片
       X ：数组，每个元素是索引的数组
       nprocs：子进程个数

       Returns
       -------

       """
       q_in = multiprocessing.Queue(1)
       q_out = multiprocessing.Queue()
       # 创建工作进程
       proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
               for _ in range(nprocs)]
       # 每个工作进程当成守护进行，主线程结束，子线程跟着结束
       for p in proc:
           p.daemon = True
           p.start()
       sent = [q_in.put((i, x)) for i, x in enumerate(X)]
       # 结束标记
       [q_in.put((None, None)) for _ in range(nprocs)]
       res = [q_out.get() for _ in range(len(sent))]
       # 等待每个工作线程结束
       [p.join() for p in proc]
       return [x for i, x in sorted(res)]
