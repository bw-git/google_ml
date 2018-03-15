[TOC]

# 使用TensorFlow的基本步骤

## 1-Pandas简介

Pandas 是用于进行数据分析和建模的重要库，广泛应用于 TensorFlow 编码。

### 基本概念

以下行导入了 *pandas* API 并输出了相应的 API 版本：

```python
import pandas as pd
pd.__version__
```

*pandas* 中的主要数据结构被实现为以下两类：

- **DataFrame**，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
- **Series**，它是单一列。`DataFrame` 中包含一个或多个 `Series`，每个 `Series` 均有一个名称。

数据框架是用于数据操控的一种常用抽象实现形式。[Spark](https://spark.apache.org/) 和 [R](https://www.r-project.org/about.html) 中也有类似的实现。

创建 `Series` 的一种方法是构建 `Series` 对象。例如：

```python
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
#------ouput------
0    San Francisco
1         San Jose
2       Sacramento
dtype: object
```

您可以将映射 `string` 列名称的 `dict` 传递到它们各自的 `Series`，从而创建`DataFrame`对象。如果 `Series` 在长度上不一致，系统会用特殊的 [NA/NaN](http://pandas.pydata.org/pandas-docs/stable/missing_data.html) 值填充缺失的值。例如:

```python
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
pd.DataFrame({ 'City name': city_names, 'Population': population })
#------ouput------
	City name	Population
0	San Francisco	852469
1	San Jose	1015785
2	Sacramento	485199
```

但是在大多数情况下，您需要将整个文件加载到 `DataFrame` 中。下面的示例加载了一个包含加利福尼亚州住房数据的文件。请运行以下单元格以加载数据，并创建特征定义

```python
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
```

上面的示例使用 `DataFrame.describe` 来显示关于 `DataFrame` 的有趣统计信息。另一个实用函数是 `DataFrame.head`，它显示 `DataFrame` 的前几个记录：

```python
california_housing_dataframe.head()
```

*pandas* 的另一个强大功能是绘制图表。例如，借助 `DataFrame.hist`，您可以快速了解一个列中值的分布：

```python
california_housing_dataframe.hist('housing_median_age')
```

###访问数据

您可以使用熟悉的 Python dict/list 指令访问 `DataFrame` 数据：

```python
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print type(cities['City name'])
cities['City name']
#------ouput------
<class 'pandas.core.series.Series'>
0    San Francisco
1         San Jose
2       Sacramento
Name: City name, dtype: object
```

```python
print type(cities['City name'][1])
cities['City name'][1]
```

```python
print type(cities[0:2])
cities[0:2]
```

###操控数据

您可以向 `Series` 应用 Python 的基本运算指令。例如：

```python
population / 1000.
```

[NumPy](http://www.numpy.org/) 是一种用于进行科学计算的常用工具包。*pandas* `Series` 可用作大多数 NumPy 函数的参数：

```python
import numpy as np
np.log(population)
```

对于更复杂的单列转换，您可以使用 `Series.apply`。像 Python [映射函数](https://docs.python.org/2/library/functions.html#map)一样，`Series.apply` 将以参数形式接受 [lambda 函数](https://docs.python.org/2/tutorial/controlflow.html#lambda-expressions)，而该函数会应用于每个值。

下面的示例创建了一个指明 `population` 是否超过 100 万的新 `Series`：

```python
population.apply(lambda val: val > 1000000)
```

`DataFrames` 的修改方式也非常简单。例如，以下代码向现有 `DataFrame` 添加了两个 `Series`：

```python
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities
```

###练习1

通过添加一个新的布尔值列（当且仅当以下*两项*均为 True 时为 True）修改 `cities` 表格：

- 城市以圣人命名。
- 城市面积大于 50 平方英里。

**注意：**布尔值 `Series` 是使用“按位”而非传统布尔值“运算符”组合的。例如，执行*逻辑与*时，应使用 `&`，而不是 `and`。

**提示：**"San" 在西班牙语中意为 "saint"。

```python
cities['Is big and named by saint'] = cities['Area square miles']>50 & cities['City name'].apply(lambda name:name.startswith('San'))
```

```Python
#数组.apply(函数名称) 实现对数组中的元素逐个调用函数
def trans_format(x):
    try:
        return double(x)
    except:
        a,b=x.split(":")
        return (float(a)- float(b))/(float(a))
   
df.loc[:,'col_name'] = df['col_name'].apply(trans_format) #进行列处理后替换。
```



###索引

`Series` 和 `DataFrame` 对象也定义了 `index` 属性，该属性会向每个 `Series` 项或 `DataFrame` 行赋一个标识符值。

默认情况下，在构造时，*pandas* 会赋可反映源数据顺序的索引值。索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。

```python
city_names.index
#------ouput------
RangeIndex(start=0, stop=3, step=1)
```

调用 `DataFrame.reindex` 以手动重新排列各行的顺序。例如，以下方式与按城市名称排序具有相同的效果：

```python
cities.reindex([2, 0, 1])
#------ouput------
City name	Population	Area square miles	Population density	Is big and named with saint	Is big and named by saint
2	Sacramento	485199	97.92	4955.055147	True	True
0	San Francisco	852469	46.87	18187.945381	True	True
1	San Jose	1015785	176.53	5754.177760	True	True
```

**重建索引**是一种==随机排列== `DataFrame` 的绝佳方式。在下面的示例中，我们会取用类似数组的索引，然后将其传递至 NumPy 的 `random.permutation` 函数，该函数会随机排列其值的位置。如果使用此重新随机排列的数组调用 `reindex`，会导致 `DataFrame` 行以同样的方式随机排列。 尝试多次运行以下单元格！

```python
cities.reindex(np.random.permutation(cities.index))
#------ouput------
City name	Population	Area square miles	Population density	Is big and named with saint	Is big and named by saint
0	San Francisco	852469	46.87	18187.945381	True	True
2	Sacramento	485199	97.92	4955.055147	True	True
1	San Jose	1015785	176.53	5754.177760	True	True
```

###练习2

`reindex` 方法允许使用未包含在原始 `DataFrame` 索引值中的索引值。请试一下，看看如果使用此类值会发生什么！您认为允许此类值的原因是什么？

```python
cities.reindex([0,2,1,3])
#------ouput------
City name	Population	Area square miles	Population density	Is big and named with saint	Is big and named by saint
0	San Francisco	852469.0	46.87	18187.945381	True	True
2	Sacramento	485199.0	97.92	4955.055147	True	True
1	San Jose	1015785.0	176.53	5754.177760	True	True
3	NaN	NaN	NaN	NaN	NaN	NaN
'''
如果您的 reindex 输入数组包含原始 DataFrame 索引值中没有的值，reindex 会为此类“丢失的”索引添加新##行，并在所有对应列中填充 NaN 值.
'''
```

## 2-使用TensorFlow的起始步骤

- **张量**是TensorFlow的==数据模型==。

  在TensorFlow的程序中，所有的数据都通过张量形式来表示，可理解为多维数组。

  张量并没有真正保存数字，它保存的是如何得到这些数字的**计算过程**。

  ```python
  import tensorflow as tf
  a=tf.constant([1.0,2.0],name="a")
  b=tf.constant([3.0,4.0],name="b")
  result=a+b
  print(result)
  #输出结果：Tensor("add:0",shape=(2,),dtype=float32)
  #结果是一个张量结构，包含3个属性：标识符（表示张量是如何计算的）、维度、类型。
  ```

- **会话**是TensorFlow的==运行模型==。

  用来执行定义好的运算，当计算完成后需要关闭会话来帮助系统回收资源。

  ```python
  sess=tf.session()#创建会话
  sess.run(result)
  print (sess.run(result))
  sess.close

  ```

- 实际案例：线性回归建模，数据基于加利福尼亚州 1990 年的人口普查数据。

### 设置

首先加载必要的库。

```python
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
```

接下来，我们将加载数据集。

```python
#pd.read_csv()返回一个DataFrame对象。
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",") #sep参数用于指定分隔符。
```

我们将对数据进行**随机化处理**，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。此外，将 `median_house_value` 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据。

```python
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```

### 检查数据

建议您在使用数据之前，先对它有一个初步的了解。

我们会输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数。

```python
california_housing_dataframe.describe()
```

### 构建第一个模型

在本练习中，我们将尝试预测 `median_house_value`，它将是我们的标签（有时也称为目标）。我们将使用 `total_rooms` 作为输入特征。

**注意**：我们使用的是城市街区级别的数据，因此该特征表示相应街区的房间总数。

为了训练模型，我们将使用 TensorFlow [Estimator](https://www.tensorflow.org/get_started/estimator) API 提供的 [LinearRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor) 接口。此 API 负责处理大量低级别模型搭建工作，并会提供执行模型训练、评估和推理的便利方法。

#### 第 1 步：定义特征并配置特征列

**为了将我们的训练数据导入 TensorFlow，我们需要指定每个特征包含的数据类型**。在本练习及今后的练习中，我们主要会使用以下两类数据：

- **分类数据**：一种文字数据。在本练习中，我们的住房数据集不包含任何分类特征，但您可能会看到的示例包括家居风格以及房地产广告词。
- **数值数据**：一种数字（整数或浮点数）数据以及您希望视为数字的数据。有时您可能会希望将数值数据（例如邮政编码）视为分类数据（我们将在稍后的部分对此进行详细说明）。

在 TensorFlow 中，我们使用一种称为“**特征列**”的**结构**来表示**特征的数据类型**。

==特征列仅存储对特征数据的描述；不包含特征数据本身==。

一开始，我们只使用一个数值输入特征 `total_rooms`。以下代码会从 `california_housing_dataframe` 中提取 `total_rooms` 数据，并使用 `numeric_column` 定义特征列，这样会将其数据指定为数值：

```python
# 定义输入特征: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]] #my_feature是一个DateFrame。
#若是：my_feature = california_housing_dataframe[["total_rooms"]]，则my_feature是一个Series。

feature_columns = [tf.feature_column.numeric_column("total_rooms")] 
#给tf配置匹配这个输入的“特征列”--这是一个对“特征数据”的描述。
```

**注意**：`total_rooms` 数据的形状是一维数组（每个街区的房间总数列表）。这是 `numeric_column` 的默认形状，因此我们不必将其作为参数传递。

#### 第 2 步：定义目标

接下来，我们将定义目标，也就是 `median_house_value`。同样，我们可以从 `california_housing_dataframe` 中提取它：

```python
# Define the label.#是一个Series。
targets = california_housing_dataframe["median_house_value"]
```

#### 第 3 步：配置 LinearRegressor

接下来，我们将使用 LinearRegressor 配置线性回归模型，并使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型。learning_rate 参数可控制梯度步长的大小。

注意：为了安全起见，我们还会通过 clip_gradients_by_norm 将**梯度裁剪**应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。

> 梯度裁剪：在应用梯度值之前先设置其上限。有助于确保数值稳定性以及防止梯度爆炸。

```python
#首先配置“梯度下降优化器”，并应用“梯度裁剪”到优化器上，进行模型训练。
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

#然后，使用前面的特征列和优化器配置线性回归模型。
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)
```

#### 第 4 步：定义输入函数

要将加利福尼亚州住房数据导入 `LinearRegressor`，我们需要定义一个输入函数，让它告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。

首先，我们将 *Pandas* 特征数据转换成 NumPy 数组字典。然后，我们可以使用 TensorFlow [Dataset API](https://www.tensorflow.org/programmers_guide/datasets) 根据我们的数据构建 Dataset 对象，并将数据拆分成大小为 `batch_size` 的多批数据，以按照指定周期数 (num_epochs) 进行重复。

**注意**：如果将默认值 `num_epochs=None` 传递到 `repeat()`，输入数据会无限期重复。

然后，如果 `shuffle` 设置为 `True`，则我们会对数据进行随机处理，以便数据在训练期间以随机方式传递到模型。`buffer_size` 参数会指定 `shuffle` 将从中随机抽样的数据集的大小。

最后，输入函数会为该数据集构建一个迭代器，并向 LinearRegressor 返回下一批数据。

```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    #首先,将Pandas特征数据转换成NumPy数组字典。
    features = {key:np.array(value) for key,value in dict(features).items()}
    #my_features是一个DataFrame，dict(my_feature)是{key:index value}的字典，因此features是{key:np.array}的字典。
    
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets))
    #从一个tensor tuple创建一个包含“多个元素”的dataset。
    #tf.data.Dataset.from_tensors((features, labels))是从一个tensor tuple创建一个单元素的dataset
    ds = ds.batch(batch_size).repeat(num_epochs)  #先设定批大小和重复num_epochs次数信息。
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000) #buffer_size设置缓存的行数，然后在其中取一个batch。
    
    #从Dataset中实例化一个迭代器，迭代返回下一批数据。
    features, labels = ds.make_one_shot_iterator().get_next()#features仍是dict。
    return features, labels
```

**注意**：在后面的练习中，我们会继续使用此输入函数。

#### 第 5 步：训练模型

现在，我们可以在 `linear_regressor` 上调用 `train()` 来训练模型。我们会将 `my_input_fn` 封装在 `lambda` 中，以便可以将 `my_feature` 和 `target` 作为参数传入（有关详情，请参阅此 [TensorFlow 输入函数教程](https://www.tensorflow.org/get_started/input_fn#passing_input_fn_data_to_your_model)），首先，我们会训练 100 步。

```python
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
```

#### 第 6 步：评估模型

我们基于该训练数据做一次预测，看看我们的模型在训练期间与这些数据的拟合情况。

**注意**：训练误差可以衡量您的模型与训练数据的拟合情况，但并**不能**衡量模型**泛化到新数据**的效果。在后面的练习中，您将探索如何拆分数据以评估模型的泛化能力。

```python
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn) #predictions是生成器类型。

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])#得到数组类型预测值。

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error
```

这是出色的模型吗？您如何判断误差有多大？

由于均方误差 (MSE) 很难解读，因此我们经常查看的是均方根误差 (RMSE)。RMSE 的一个很好的特性是，它可以在与原目标相同的规模下解读。

我们来比较一下 RMSE 与目标最大值和最小值的差值：

```python
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print "Min. Median House Value: %0.3f" % min_house_value
print "Max. Median House Value: %0.3f" % max_house_value
print "Difference between Min. and Max.: %0.3f" % min_max_difference
print "Root Mean Squared Error: %0.3f" % root_mean_squared_error
```

我们的误差跨越目标值的近一半范围，可以进一步缩小误差吗？

这是每个模型开发者都会烦恼的问题。我们来制定一些基本策略，以降低模型误差。

首先，我们可以了解一下根据总体摘要统计信息，预测和目标的符合情况。

```pyhon
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()
```

好的，此信息也许有帮助。平均值与模型的 RMSE 相比情况如何？各种分位数呢？

我们还可以将数据和学到的线可视化。我们已经知道，单个特征的线性回归可绘制成一条将输入 *x* 映射到输出 *y* 的线。

首先，我们将获得均匀分布的随机数据样本，以便绘制可辨的散点图。

```python
sample = california_housing_dataframe.sample(n=300)
```

然后，我们根据模型的偏差项和特征权重绘制学到的线，并绘制散点图。该线会以红色显示。

```python
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
```

这条初始线看起来与目标相差很大。看看您能否回想起摘要统计信息，并看到其中蕴含的相同信息。

综上所述，这些初始健全性检查提示我们也许可以找到更好的线。

### 调整模型超参数

对于本练习，为方便起见，我们已将上述所有代码放入一个函数中。您可以使用不同的参数调用该函数，以了解相应效果。

我们会在 10 个等分的时间段内使用此函数，以便观察模型在每个时间段的改善情况。

对于每个时间段，我们都会计算训练损失并绘制相应图表。这可以帮助您判断模型收敛的时间，或者模型是否需要更多迭代。

此外，我们还会绘制模型随着时间的推移学习的特征权重和偏差项值的曲线图。您还可以通过这种方式查看模型的收敛效果。

```python
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
  # Create input functions
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print "Training model..."
  print "RMSE (on training data):"
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, root_mean_squared_error)
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print "Model training finished."

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print "Final RMSE (on training data): %0.2f" % root_mean_squared_error
```

### 任务1：使RMSE不超过180

调整模型超参数，以降低损失和更符合目标分布。 约 5 分钟后，如果您无法让 RMSE 低于 180，请查看解决方案，了解可能的组合。

```python
train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1
)
```

### 解决方案

####有适用于模型调整的标准启发法吗？

这是一个常见的问题。简短的答案是，不同超参数的效果取决于数据。因此，不存在必须遵循的规则，您需要对自己的数据进行测试。

即便如此，我们仍在下面列出了几条可为您提供指导的经验法则：

- 训练误差应该稳步减小，刚开始是急剧减小，最终应随着训练收敛达到平稳状态。
- 如果训练尚未收敛，尝试运行更长的时间。
- 如果训练误差减小速度过慢，则提高学习速率也许有助于加快其减小速度。
  - 但有时如果学习速率过高，训练误差的减小速度反而会变慢。
- 如果训练误差变化很大，尝试降低学习速率。
  - 较低的学习速率和较大的步数/较大的批量大小通常是不错的组合。
- 批量大小过小也会导致不稳定情况。不妨先尝试 100 或 1000 等较大的值，然后逐渐减小值的大小，直到出现性能降低的情况。

重申一下，切勿严格遵循这些经验法则，因为效果取决于数据。请始终进行试验和验证。

### 任务2：尝试其他特征

```
train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=5
    input_feature="population"
)
```

## 3-合成特征值

回顾下之前的“使用 TensorFlow 的基本步骤”练习中的模型。

首先，我们将加利福尼亚州住房数据导入 *Pandas* `DataFrame` 中：

### 设置

```python
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format

california_housing_dataframe=pd.read_csv("https://storage.googleapis.com/mledu-dataset/california_housing_train.csv",sep=',')
california_housing_dataframe=california_housing_dataframe.reindex(
	np.random_permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```

接下来，我们将设置输入函数，并针对模型训练来定义该函数：

```python
def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    features={key:np.array(value) for key,value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features,targests))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    features,labels = ds.make_one_shot_iterator().get_next()
    return featurea,labels
```

```python
def train_model(learning_rate,steps,batch_size,input_feature):
    periods = 10
    steps_per_period = steps/periods
    
    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label].astype('float32')
    
    training_input_fn = lambda:my_input_fn(my_feature_data,targets,batch_size=batch_size)
    predict_training_input_fn=lambda:my_input_fn(my_feature_data,targets,num_epochs=1,shuffle=False)
    
    feature_columns=[tf.features_column.numeric_column(my_feature)]
    
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    linear_regressor = tf.estimator.LinearRegressor(
    	feature_columns=feature_columns,
    	optimizer = my_optimizer)
    
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.title("Learned Line by Period")
    plt.ylable(my_label)
    plt.xlabel(my_feature)
    sample=california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature],sample[my_label])
    colors=[cm.coolwarn(x) for x in np.linespace(-1,1,periods)]
    
    print "Training model..."
    print "RMSE (on training data):"
    root_mean_squared_errors = []
    for period in range(0,periods):
        linear_regressor.train(
        	input_fn=training_input_fn,
        	steps = steps_per_period
        )
    predictions = linear_regresoor.predict(input_fn=predict_training_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    root_mean_squared_error = math.sqrt(
    	metrics.mean_squared_error(predictions,targets))
    
    print "period %02d: %0.2f" %(period, root_mean_squared_error)
    
    root_mean_squared_errors.append(root_mean_squared_error)
    
    y_extents = np.array([0,sample[my_label].max()])
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights'%input_features)[0]
  	bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    
    x_extents = (y_extents-bias) / weight
    x_extents = np.maximum(np.minimun(x_extents,sample[my_feature].max()),
                          sample[my_feature].min())
    y_extents = weight*x_extents + bias
    plt.plot(x_extents,y_extents,color=colors[period])
print "Model training finished"

plt.subplot(1,2,2)
plt.ylabel('RMSE')
plt.xlabel('Periods')
plt.tigle("Root Mean squared error vs. Periods")
plt.tight_layout()
plt.plot(root_mean_squared_errors)

calibration_data=pd.DataFrame()
calibration_data["perdictions"] = pd.Series(predictions)
calibration_data["targets"]=pd.Series(targets)
display.display(calibration_data.describe())
print "Final RMSE (on training data):%0.2f"%root_mean_squared_error
return calibration_data
```

### 任务1：尝试合成特征

`total_rooms` 和 `population` 特征都会统计指定街区的相关总计数据。

但是，如果一个街区比另一个街区的人口更密集，会怎么样？我们可以创建一个合成特征（即 `total_rooms` 与 `population` 的比例）来探索街区人口密度与房屋价值中位数之间的关系。

在以下单元格中，创建一个名为 `rooms_per_person` 的特征，并将其用作 `train_model()` 的 `input_feature`。

通过调整学习速率，您使用这一特征可以获得的最佳效果是什么？（效果越好，回归线与数据的拟合度就越高，最终 RMSE 也会越低。）

```PYTHON
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")
```

### 任务2：识别离群值

我们可以通过创建预测值与目标值的散点图来可视化模型效果。理想情况下，这些值将位于一条完全相关的对角线上。

使用您在任务 1 中训练过的人均房间数模型，并使用 Pyplot 的 `scatter()` 创建预测值与目标值的散点图。

您是否看到任何异常情况？通过查看 `rooms_per_person` 中值的分布情况，将这些异常情况追溯到源数据。

```PYTHON
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])
```

校准数据显示，大多数散点与一条线对齐。这条线几乎是垂直的，我们稍后再讲解。现在，我们重点关注偏离这条线的点。我们注意到这些点的数量相对较少。

如果我们绘制 `rooms_per_person` 的直方图，则会发现我们的输入数据中有少量离群值：

```PYTHON
plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()
```

### 任务3：截取离群值

看看您能否通过将 `rooms_per_person` 的离群值设置为相对合理的最小值或最大值来进一步改进模型拟合情况。

以下是一个如何将函数应用于 Pandas `Series` 的简单示例，供您参考：

```python
clipped_feature = my_dataframe["my_feature_name"].apply(lambda x: max(x, 0))
```

上述 `clipped_feature` 没有小于 `0` 的值。

```python
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

_ = california_housing_dataframe["rooms_per_person"].hist()
```

为了验证截取是否有效，我们再训练一次模型，并再次输出校准数据：

```python
calibration_data = train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person")

_ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
```

## 4-总结

### 本课程中的常用超参数

- steps：是指训练迭代的总次数。一步计算一批样本产生的损失，然后使用该值修改模型的权重一次。

- batch size：是指单步的样本数量（随机选择）。例如，SGD 的批量大小为 1。

  `总的训练样本数=batch size*steps`

### 方便变量

- periods：控制报告的粒度。

  例如，如果 `periods` 设为 7 且 `steps` 设为 70，则练习将每 10 步（或 7 次）输出一次损失值。与超参数不同，我们不希望您修改 `periods` 的值。请注意，修改 `periods` 不会更改您的模型所学习的内容。

  `每次训练的样本数=(batch size*steps)/periods`