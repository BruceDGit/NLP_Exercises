"""
    分词
    pyhanlp获取hanlp中分词器的方式有两种
    第一种：直接调用封装好的hanlp类（目前有五种，如下）
        维特比 (viterbi)：效率和效果的最佳平衡。也是最短路分词，HanLP最短路求解采用Viterbi算法
        双数组trie树 (dat)：极速词典分词，千万字符每秒（可能无法获取词性，此处取决于你的词典）
        条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务
        感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习
        N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度
    第二种：使用JClass直接获取java类（除了以上5类分词器外，还可以获取的分词器有如下）
        NLP分词器（NLPTokenizer）：会执行词性标注和命名实体识别，由结构化感知机序列标注框架支撑。默认模型训练自9970万字的大型综合语料库，是已知范围内全世界最大的中文分词语料库。
        索引分词器（IndexTokenizer）：是面向搜索引擎的分词器，能够对长词全切分，另外通过term.offset可以获取单词在文本中的偏移量。
                                     （注：任何分词器都可以通过基类Segment的enableIndexMode方法激活索引模式。）
        快速字典分词器等

"""
from pyhanlp import *


with open("./data/1.txt", "r") as f:
    text = f.read()


# 维特比 (viterbi)
ViterbiNewSegment = HanLP.newSegment("viterbi")
print(ViterbiNewSegment.seg(text))

# 双数组trie树 (dat)

# 条件随机场 (crf)

# 感知机 (perceptron)

# N最短路 (nshort)
