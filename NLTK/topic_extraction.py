import nltk.tokenize as tk
import nltk.corpus as nc
import nltk.stem.snowball as sb
import gensim.models.ldamodel as gm
import gensim.corpora as gc


# 读取文件
doc = []
with open('./data/topic.txt', 'r') as f:
    for line in f:
        doc.append(line[:-1])
print(doc)

# 预处理
tokenizer = tk.WordPunctTokenizer() # 分词器对象
stopwords = nc.stopwords.words('english') # 停用词
signs = [',', '.', '!'] # 无用符号
stemmer = sb.SnowballStemmer('english') # 词干提取器
lines_tokens = []
for line in doc:
    tokens = tokenizer.tokenize(line.lower())
    line_tokens = []
    for token in tokens:
        if token not in stopwords and token not in signs:
            token = stemmer.stem(token)
            line_tokens.append(token)
    lines_tokens.append(line_tokens)

# 存入gc的词典对象
dic = gc.Dictionary(lines_tokens)

# 通过字典构建词袋
bow = []
for line_tokens in lines_tokens:
    row = dic.doc2bow(line_tokens)
    bow.append(row)

# LDA模型
n_topics = 2
model = gm.LdaModel(bow, num_topics=n_topics, id2word=dic, passes=25)

# 输出主题词
topics = model.print_topics(num_topics=n_topics, num_words=4)
print(topics)
