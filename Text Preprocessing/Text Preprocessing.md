# 用Python进行文本预处理

原文链接：https://www.kaggle.com/shashanksai/text-preprocessing-using-python/notebook

Bruce 译

---



我们不能在机器学习中直接使用文本数据，所以我们需要将其转换为数值向量。本文章将为你介绍所有的文本转换技术。

 

> 在将文本数据提供给机器学习模型之前，需要将文本数据清洗并编码为数值，这种清洗和编码的过程称为**“文本预处理”**

 

在本篇文章中，我们将看到一些基本的文本清洗步骤和文本数据编码技术。 我们将会看到：

1. **了解数据** - 明白数据的全部含义。清理数据时应该考虑哪些事项（标点符号、停用词等）。

2. **基本清理** - 我们将看到清理数据时需要考虑哪些参数（例如标点符号，停用词等）及其代码。

3. 编码技术

    \- All the popular techniques that are used for encoding that I personally came across.

   - **Bag of Words**（词袋模型）
   - **Binary Bag of Words**（二进制词袋模型）
   - **Bigram, Ngram**
   - **TF-IDF**( **T**erm  **F**requency - **I**nverse **D**ocument **F**requency词频-逆文档频率)
   - **Word2Vec**
   - **Avg-Word2Vec**
   - **TF-IDF Word2Vec**

现在让我们开始这个有趣的旅程吧！

 

**导入库**

 

```python
import warnings
warnings.filterwarnings("ignore")         # 忽略不必要的警告

import numpy as np                        # 用于大型和多维数组
import pandas as pd                       # 用于数据处理和分析
import nltk                               # 自然语言处理工具包

from nltk.corpus import stopwords         # 停用词语料库
from nltk.stem import PorterStemmer       # 词干提取器

from sklearn.feature_extraction.text import CountVectorizer    #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer    #For TF-IDF
from gensim.models import Word2Vec                             #For Word2Vec
```

 

```python
data_path = "../input/Reviews.csv"
data = pd.read_csv(data_path)
data_sel = data.head(10000)                 # 只考虑前 10000 行
```

 

```python
# 查看数据的结构
print(data_sel.columns)
```

 输出结果:

```python
Index(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text'],
      dtype='object')
```

 

1. **了解数据**

从数据集来看，我们的主要目标是根据文本预测评论是**正面**还是**负面**。

我们看到Score列，它的的取值有1,2,3,4,5。其中1、2为负面评价，4、5为正面评价。对于得分=3，我们将其视为中性评价，让我们删除中性行，以便可以预测正或负。

HelfulnessNumerator表示有多少人发现评论有用，HelpfulnessDenominator是关于usefull 评论数 + not so usefull 评论数的。由此我们可以看出，HelfulnessNumerator 总是小于或等于 HelpfulnessDenominator。

 

```python
data_score_removed = data_sel[data_sel['Score']!=3]       # 去除中性评价
```

 

将分数值转换为类别标签Postiuve或Negative。

```python
def partition(x):
    if x < 3:
        return 'positive'
    return 'negative'

score_upd = data_score_removed['Score']
t = score_upd.map(partition)  # Series.map方法，接受一个函数或含有映射关系的字典型对象
data_score_removed['Score']=t
```

 

1. **基本清洗**

**重复数据删除**表示删除重复的行，必须删除重复的行才能获得稳定的结果。 根据UserId，ProfileName，Time，Text检查重复项。 如果所有这些值都相等，那么我们将删除这些记录。 （没有用户可以在相同的确切时间输入不同产品的评论。）

我们已经知道HelpfulnessNumerator应该始终小于或等于HelpfulnessDenominator，因此检查此条件并删除这些记录。



```python
final_data = data_score_removed.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})
```

 

```python
final = final_data[final_data['HelpfulnessNumerator'] <= final_data['HelpfulnessDenominator']]
```

 

```python
final_X = final['Text']
final_y = final['Score']
```

 

将所有单词转换为小写并删除标点符号和html标签（如果有）。

**词干提取**-将单词转换为基本词或词干（例如 - tastefully, tasty，这些单词将转换为词干“ tasti”）。 因为我们不考虑所有相似的词，所以这减小了向量维数。

**停用词** - 停用词是不必要的词，即使它们被删除了，句子的语义也不会发生变化。

例如 - This pasta is so tasty ==> pasta tasty    ( This , is, so 都是停用词，因此将它们删除)

参加以下代码单元，可以查看所有停用词。

 

```python
stop = set(stopwords.words('english')) 
print(stop)
```

输出结果：

```python
{'in', 'below', "aren't", 'them', 'be', 'needn', 'as', 'into', 'is', 'haven', 'o', "hadn't", 'few', 'until', 'she', 'for', 'his', 'do', 'what', 'again', 'mustn', "that'll", 'yourself', 're', 'most', 'y', "haven't", 'where', 'own', 'about', 'yourselves', 'before', "hasn't", "mustn't", 'an', 'been', "she's", 'hers', 'which', 'was', 'did', 'with', 'from', 'themselves', 'ourselves', 'we', "shouldn't", 'doing', 'should', 'between', 't', 'further', 'wasn', 'him', 'not', 'those', 'other', 'doesn', "weren't", 'your', 'don', 'my', 'that', 'd', 'you', 'there', 'any', 'very', 'only', 'who', 'through', 'i', 'up', 'same', 'after', 'the', 'why', 'ours', 'out', 'theirs', 'to', 'hadn', 'couldn', 'at', 'her', 'some', 'have', 'here', 'our', 'myself', 'once', "wouldn't", "you'll", "don't", 's', 'how', 'm', 'by', 'such', 'will', 'each', 'while', 'me', "doesn't", 'when', "you'd", 'these', 'it', 'no', "wasn't", 'just', 'than', 'or', 'having', 'itself', 'too', 'now', 'on', 'himself', 'won', 'down', 'so', "couldn't", 'but', 'hasn', "didn't", "it's", 'its', "isn't", 'weren', 'whom', 'shan', "won't", 'and', 'being', 'herself', 'ma', 'over', 'll', 'are', 've', 'off', 'has', 'ain', 'aren', 'both', 'nor', 'didn', 'does', "shan't", 'he', 'against', 'then', 'yours', 'all', 'during', 'under', 'mightn', 'isn', 'this', 'wouldn', 'above', "mightn't", 'their', 'am', "should've", 'a', 'more', 'of', 'had', 'were', 'because', "you've", 'they', "needn't", 'shouldn', "you're", 'if', 'can'}
```

 

```python
import re
temp =[]
snow = nltk.stem.SnowballStemmer('english')
for sentence in final_X:
    sentence = sentence.lower()                 # 转换为小写
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #将 HTML 标签替换为空格
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #将标点符号替换为空格
    
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # 删除停用词并提取词干
    temp.append(words)
    
final_X = temp    
```

 

```python
print(final_X[1])
```

 输出结果：

```python
['product', 'arriv', 'label', 'jumbo', 'salt', 'peanut', 'peanut', 'actual', 'small', 'size', 'unsalt', 'sure', 'error', 'vendor', 'intend', 'repres', 'product', 'jumbo']
```

 

```python
sent = []
for row in final_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

final_X = sent
print(final_X[1])
```

 输出结果：

```python
 product arriv label jumbo salt peanut peanut actual small size unsalt sure error vendor intend repres product jumbo
```

 

1. **编码技术**

   **词袋模型**

   在BoW中，我们构建了一个词典，其中包含来自文本评论数据集中的所有不重复单词的集合。在这里会计算每一个单词出现的频率。 如果字典中有 **d** 个不重复词，则对于每个句子或评论，其向量的长度将为 **d**，并将来自评论的词数存储在向量中的特定位置。 在这种情况下，向量将会非常稀疏。

   例如. pasta is tasty and pasta is good

   **[0]....[1]............[1]...........[2]..........[2]............[1]..........**            <== 它的向量表示（其余所有点将表示为零）

   **[a]..[and].....[good].......[is].......[pasta]....[tasty].......**            <== 这是词典

   使用scikit-learn的CountVectorizer，我们可以获取BoW并检查其包含的所有参数，其中之一是max_features = 5000，它表示仅考虑将前5000个最频繁重复的单词放置在字典中。 因此我们的字典长度或向量长度只有5000。

**二进制词袋模型**

在二进制BoW中，我们不计算单词出现的频率，如果单词出现在评论中，我们就只将对应位置设置为 **1**，否则设置为 **0**。 在CountVectorizer中，有一个参数 **binary = true**，它可以使我们的BoW变为二进制BoW。

 

```python
count_vect = CountVectorizer(max_features=5000)
bow_data = count_vect.fit_transform(final_X)
print(bow_data[1])
```

 输出结果：

```python
  (0, 3641)	1
  (0, 2326)	1
  (0, 4734)	1
  (0, 1539)	1
  (0, 4314)	1
  (0, 4676)	1
  (0, 3980)	1
  (0, 4013)	1
  (0, 162)	1
  (0, 3219)	2
  (0, 3770)	1
  (0, 2420)	2
  (0, 2493)	1
  (0, 332)	1
  (0, 3432)	2
```

 

**词袋/二进制词袋模型的缺点**

我们在进行这些文本到矢量编码时的主要目标是，意思相近的文本矢量应该彼此接近，但是在某些情况下，这对于Bow来说是不可能的

例如，如果我们考虑两个评论 “ **This pasta is very tasty**” 和 “**This pasta is not tasty** ”，则在停用词移除后，这两个句子都将转换为 “ **pasta tasty**”，因此两者的含义完全相同。

主要问题是这里我们没有考虑与每个单词相关的前后词，于是就有了Bigram和Ngram技术。

 

**BI-GRAM 词袋模型**

考虑到用于创建字典的单词对为Bi-Gram，因此Tri-Gram表示三个连续的单词，以此类推到NGram。

CountVectorizer 有一个参数 **ngram range** 如果分配给（1,2），表示为 Bi-Gram BoW。

但这极大地增加了我们的字典容量。

 

```python
final_B_X = final_X
```

 

```python
count_vect = CountVectorizer(ngram_range=(1,2))
Bigram_data = count_vect.fit_transform(final_B_X)
print(Bigram_data[1])
```

 输出结果：

```python
  (0, 143171)	1
  (0, 151696)	1
  (0, 95087)	1
  (0, 196648)	1
  (0, 60866)	1
  (0, 177168)	1
  (0, 193567)	1
  (0, 164722)	1
  (0, 165627)	1
  (0, 4021)	1
  (0, 133855)	1
  (0, 133898)	1
  (0, 155987)	1
  (0, 97865)	1
  (0, 100490)	1
  (0, 11861)	1
  (0, 142800)	1
  (0, 151689)	1
  (0, 95076)	1
  (0, 196632)	1
  (0, 60852)	1
  (0, 177092)	1
  (0, 193558)	1
  (0, 164485)	1
  (0, 165423)	1
  (0, 3831)	1
  (0, 133854)	2
  (0, 155850)	1
  (0, 97859)	2
  (0, 100430)	1
  (0, 11784)	1
  (0, 142748)	2
```

 

**词频 - 逆文档频率（TF-IDF）**

**词频 - 逆文档频率**，它可以确保对最频繁使用的单词给予较少的重视，同时也考虑低频使用的单词。

**词频**是在评论中出现 **特定单词（W）**的次数除以评论中的总单词数量 **（Wr）**。 词频值的范围是0到1。

**逆文档频率 **的计算为 **log（文档总数（N）/包含特定单词的文档数（n））**。 这里的文档称为 “评论”。

**词频 - 逆文档频率(TF-IDF)** 等于 $TF*IDF$ 即 $\dfrac{W*log{\frac Nn}}{Wr}$ 

使用scikit-learn的tfidfVectorizer，我们可以获得TF-IDF。

那么，即使在这里，我们为每个单词都获得了TF-IDF值，在某些情况下，删除停用词后，它还是可能会将不同含义的评论视为是相似的。 因此，我们可以使用BI-Gram或NGram模型。

 

```python
final_tf = final_X
tf_idf = TfidfVectorizer(max_features=5000)
tf_data = tf_idf.fit_transform(final_tf)
print(tf_data[1])
```

 

```Python
  (0, 3432)	0.1822092004981035
  (0, 332)	0.1574317775964303
  (0, 2493)	0.18769649750089953
  (0, 2420)	0.5671119742041831
  (0, 3770)	0.1536626385509959
  (0, 3219)	0.3726548417697838
  (0, 162)	0.14731616688674187
  (0, 4013)	0.14731616688674187
  (0, 3980)	0.14758995053747803
  (0, 4676)	0.2703170210936338
  (0, 4314)	0.14376924933112933
  (0, 1539)	0.2676489579732629
  (0, 4734)	0.22110622670603633
  (0, 2326)	0.25860104128863787
  (0, 3641)	0.27633136515735446
```

 

因此，要真正克服距离较近的语义评论的问题，我们需要使用Word2Vec。

**Word2Vec**

Word2Vec实际上采用了单词的语义含义以及它们与其他单词之间的关系。 它学习单词之间的所有内部关系，以密集的向量形式表示单词。

使用 **Gensim** 的库，我们就可以调用 Word2Vec，它接收的是像 **min_count = 5** 之类的参数，表示仅当单词在整个数据中重复超过5次时才会考虑。 **size = 50** 表示矢量长度为50，而 **workers** 表示的是运行此代码的核数。

**Average Word2Vec**

计算每个单词的Word2vec，将每个单词的向量相加，然后将向量除以句子的单词数，简单地求出所有单词的Word2vec的平均值。

 

```python
w2v_data = final_X
```

 

```python
splitted = []
for row in w2v_data: 
    splitted.append([word for word in row.split()])     #分词
```

 

```python
train_w2v = Word2Vec(splitted,min_count=5,size=50, workers=4)
```

 

```python
avg_data = []
for row in splitted:
    vec = np.zeros(50)
    count = 0
    for word in row:
        try:
            vec += train_w2v[word]
            count += 1
        except:
            pass
    avg_data.append(vec/count)
    
```

 

```python
print(avg_data[1])
```

 输出结果：

```python
[-0.01928419  0.24927433  0.25399788  0.36102516 -0.12853559  0.15150963
  0.17890403 -0.45340626  0.24381623  0.16463477  0.54286971  0.41620433
 -0.53215231  0.04350571  0.06012735 -0.15802164  0.26773642  0.3363306
 -0.12039118  0.37613642  0.16747488 -0.39811782 -0.27751294 -0.17169046
  0.29225921  0.12650274 -0.47982404  0.10921999  0.07278306  0.28619188
  0.39884462 -0.17332458  0.10595465 -0.4744667  -0.31394921 -0.37980286
  0.66549961 -0.47110406 -0.13237053  0.30876227  0.13172063  0.07284644
  0.25580531 -0.10714663 -0.18783146  0.06186486 -0.06675584 -0.31708457
  0.18070466 -0.09296898]
```

 

**TF-IDF WORD2VEC**

在TF-IDF Word2Vec中，每个单词的Word2Vec值乘以该单词的tfidf值，然后求和，然后除以句子的tfidf值之和。

就像是：

```python
                    V = ( t(W1)*w2v(W1) + t(W2)*w2v(W2) +.....+t(Wn)*w2v(Wn))/(t(W1)+t(W2)+....+t(Wn))
```

 

```python
tf_w_data = final_X
tf_idf = TfidfVectorizer(max_features=5000)
tf_idf_data = tf_idf.fit_transform(tf_w_data)
```

 

```python
tf_w_data = []
tf_idf_data = tf_idf_data.toarray()
i = 0
for row in splitted:
    vec = [0 for i in range(50)]
    
    temp_tfidf = []
    for val in tf_idf_data[i]:
        if val != 0:
            temp_tfidf.append(val)
    
    count = 0
    tf_idf_sum = 0
    for word in row:
        try:
            count += 1
            tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
            vec += (temp_tfidf[count-1] * train_w2v[word])
        except:
            pass
    vec = (float)(1/tf_idf_sum) * vec
    tf_w_data.append(vec)
    i = i + 1

print(tf_w_data[1])
    
```

 

```python
[ 0.04083192  0.02791251  0.15195388  0.60205388 -0.11878467  0.01459805
 -0.006749   -0.5096831   0.39797668  0.19843598  0.74515618  0.46565998
 -0.53260218  0.02270999 -0.09565694 -0.3178224   0.27076402  0.37911849
  0.14469637  0.33833481  0.16839808 -0.27573225 -0.37842466 -0.23809766
  0.10115526  0.1781537  -0.49534777  0.01055119  0.11080601  0.54118843
  0.62019201 -0.10166378  0.10285218 -0.58262819 -0.42879161 -0.44927694
  0.73491928 -0.6251714  -0.24852552  0.47421936  0.41891958  0.12787767
  0.27522988 -0.19559242 -0.29942479  0.1362636   0.08118395 -0.3148372
  0.13429163 -0.14443259]
```

 

**结论**

通过本文我们看到了将文本数据编码为数值向量的不同技术。 但是哪种技术适合我们的机器学习模型还要取决于数据的结构和模型的目标。