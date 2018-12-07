
import gensim.downloader as api

model = api.load('glove-twitter-25')  # download the corpus and return it opened as an iterable
print(model.wv["car"])
print(model.most_similar(positive=[model.wv['computer']], topn=10))