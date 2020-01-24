import numpy as np
import sklearn.datasets as sd
import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as ft
import sklearn.model_selection as ms
import sklearn.metrics as sm

# reading
data = sd.load_files('./data/20news', encoding='latin1', shuffle=True, random_state=7)
print('-'*45)
# print(np.array(data.data)[0])
# print(np.array(data.target)[0])
# print(data.target_names)

# tfidf
cv = ft.CountVectorizer()
bow = cv.fit_transform(data.data)
tt = ft.TfidfTransformer()
tfidf = tt.fit_transform(bow)
# print(tfidf.shape)
train_x, test_x, train_y, test_y = ms.train_test_split(tfidf, data.target, test_size=0.1, random_state=7)
model = nb.MultinomialNB()
model.fit(train_x, train_y)

# predict
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))