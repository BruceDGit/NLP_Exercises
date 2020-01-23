import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft

doc = 'The brown dog is running. ' \
      'The black dog is in the black room. ' \
      'Running in the room is forbidden.'
print('doc:\n', doc)
sentences = tk.sent_tokenize(doc)
print(sentences)
cv = ft.CountVectorizer()
bow_temp = cv.fit_transform(sentences)
bow = bow_temp.toarray()
print('bow_temp data type:\n', type(bow_temp))
print('bow:\n', bow)
words = cv.get_feature_names()
print('words:\n', words)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow).toarray()
print('tfidf:\n', tfidf)
