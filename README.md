# HUST Machine Learning Lab
## 简介
华中科技大学2020年春季机器学习课程实验源代码，ACM班ZDY编写
## 文件说明
### 实验一：KNN识别手写数字  
代码文件为knn.py, 使用tensorflow2读取mnist数据集
### 实验二：朴素贝叶斯垃圾邮件分类
代码文件为naive_bayes.py（没有将数据进行矩阵化因此运行速度不高, 打开文件在类内部打开，这里可以继续改进）
### 结课实验：个人收入预测
1. 使用logistic回归作为分类器，代码文件为logistic_regression.py，使用梯度下降算法(SGD, Adagrad, RMSProp)更新参数
2. 使用非核化的SVM作为分类器, 代码文件为linear_svm.py， 同样使用与逻辑回归一样的梯度下降算法
3. 使用核化的SVM作为分类器, 代码文件为kernel_svm.py, 接受一个核函数类，使用凸优化包cvxopt进行训练与预测
4. 相关工具：  
   代码文件tools_logistic_and_svm.py：包括通过csv文件获得矩阵化的数据的函数，实验相关作图函数，tsne降维可视化函数   
   代码文件kernel.py：定义核函数类，包括linear kernel与gaussian kernel，每个类需要有类数据成员linear(表示是否为线性核)以及类函数
   成员calculate(计算两个变量对应的核函数值)
   
### 扩展：使用tensorflow2重写结课实验部分代码
1. 文件tf_logistic_regression.py：继承tf.keras.Model自定义模型实现LogisticRegression二分类器
2. 文件tf_linear_svm.py：继承tf.keras.Model自定义模型实现LinearSVM二分类器