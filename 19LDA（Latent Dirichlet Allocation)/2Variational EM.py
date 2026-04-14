import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    example = [["guide", "investing", "market", "stock"],
               ["dummies", "investing"],
               ["book", "investing", "market", "stock"],
               ["book", "investing", "value"],
               ["investing", "value"],
               ["dads", "guide", "investing", "rich", "rich"],
               ["estate", "investing", "real"],
               ["dummies", "investing", "stock"],
               ["dads", "estate", "investing", "real", "rich"]]

    # 将文档转换为词频向量：(文本集合中的第m个文本,单词集合中的第v个单词) = 第v个单词在第m个文本中的出现频数
    """
    实例化一个词频向量化器
    CountVectorizer 是一个实现**词袋模型（Bag of Words）**的工具类
    分词与清洗： 自动将句子按空格切分为单词，统一转换为小写，去掉无意义的标点符号。
构建词汇表： 扫描所有文本，提取出全部不重复的单词，给每个单词分配一个固定的列索引编号（这就是为什么你能通过 .get_feature_names_out() 拿到一个按照字母表排好序的单词列表）。
统计词频转化矩阵： 将每一段文本映射成一个长长的向量（长度等于词汇表的单词总数）。每一列对应一个单词词频，也就是统计这篇文本里某个单词出现了几次。
    """
    count_vector = CountVectorizer()
    # print("【单词列表】count_vector.get_feature_names_out() = 单词列表", count_vector)
    tf = count_vector.fit_transform([" ".join(doc) for doc in example])
    # print("【单词列表】count_vector.get_feature_names_out() = 单词列表", count_vector.get_feature_names_out())
    # print("【文本-单词计数矩阵】tf[i][j] = 第i个文本中第j个单词的出现频数", tf.toarray())


    """
    batch（批量变分推断）：模型在每次 EM 迭代更新参数时，都会把所有的训练文档完完整整地看一遍。优点是学习稳定准确，缺点是如果数据量极大，内存开销大且非常慢。由于这里的 example 样本集很小，使用 batch 是最合适的。
    online（在线变分推断）：每次迭代只抽取一小批数据（Mini-batch）来更新参数，速度更快，常用于海量数据的训练或是流式流数据
    """
    # 训练LDA主题模型：n_components = 话题数量
    lda = LatentDirichletAllocation(n_components=3,  # 话题个数K
                                    learning_method="batch",  # 学习方法：batch=变分推断EM算法(默认)；online=在线变分推断EM算法
                                    random_state=0)
    doc_topic_distr = lda.fit_transform(tf)

    print("【文本-话题计数矩阵】doc_topic_distr[i] = 第i个文本的话题分布")
    print(doc_topic_distr)
    # [[0.07 0.86 0.07]
    #  [0.12 0.76 0.12]
    #  [0.07 0.86 0.07]
    #  [0.82 0.09 0.09]
    #  [0.76 0.12 0.12]
    #  [0.88 0.06 0.06]
    #  [0.09 0.09 0.82]
    #  [0.09 0.83 0.09]
    #  [0.07 0.06 0.88]]

    print("【单词-话题非规范化概率矩阵】components_[i][j] = 第i个话题生成第j个单词的未规范化的概率")
    print(lda.components_)
    # [[1.33 1.33 0.33 0.33 1.34 3.33 0.33 0.33 2.35 0.33 2.33]
    #  [1.33 0.33 2.33 0.33 1.32 4.33 2.33 0.33 0.33 3.33 0.33]
    #  [0.34 1.34 0.33 2.33 0.34 2.34 0.33 2.33 1.32 0.33 0.33]]

    print("【单词-话题概率矩阵】components_[i][j] = 第i个话题生成第j个单词的概率")
    print(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis])
    # [[0.1  0.1  0.02 0.02 0.1  0.24 0.02 0.02 0.17 0.02 0.17]
    #  [0.08 0.02 0.14 0.02 0.08 0.26 0.14 0.02 0.02 0.2  0.02]
    #  [0.03 0.11 0.03 0.2  0.03 0.2  0.03 0.2  0.11 0.03 0.03]]