import nltk.classify as cf
import nltk.classify.util as cu


train_data = [({'age': 15, 'score1': 95, 'score2': 95}, 'good'),
  ({'age': 15, 'score1': 45, 'score2': 55}, 'bad')]
test_data = [({'age': 16, 'score1': 85, 'score2': 80}, 'good'),
  ({'age': 14, 'score1': 86, 'score2': 58}, 'bad')]
test_x = [data[0] for data in test_data]

model = cf.NaiveBayesClassifier.train(train_data)
pred_res = model.classify_many(test_x)
print(pred_res)
ac = cu.accuracy(model, test_data)
print(ac)