import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = 'The brown dog is running. ' \
      'The black dog is in the black room. ' \
      'Running in the room is forbidden.'
print(doc)
sentences = tk.sent_tokenize(doc)
print(sentences)
# 提取词袋矩阵
cv = ft.CountVectorizer()
bow = cv.fit_transform(sentences).toarray()
print(bow)
# 获取词袋模型中每个特征名
words = cv.get_feature_names()
print(words)
