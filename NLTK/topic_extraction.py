import nltk.tokenize as tk
import nltk.corpus as nc
import nltk.stem.snowball as sb
import gensim.models.ldamodel as gm
import gensim.corpora as gc


doc = []
with open('./data/topic.txt', 'r') as f:
    for line in f:
        doc.append(line[:-1])
tokenizer = tk.WordPunctTokenizer()
stopwords = nc.stopwords.words('english')
signs = [',', '.', '!']
stemmer = sb.SnowballStemmer('english')
lines_tokens = []
for line in doc:
    tokens = tokenizer.tokenize(line.lower())
