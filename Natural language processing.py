import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

train_data = load_files("./aclImdb/train/", encoding="utf-8")
X_train, y_train = train_data.data, train_data.target

test_data = load_files("./aclImdb/test/", encoding="utf-8")
X_test, y_test = test_data.data, test_data.target

vectorizer_bow = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english', ngram_range=(1,2), max_features=5000)
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)
y_pred_bow = nb_bow.predict(X_test_bow)
print("BoW Accuracy:", accuracy_score(y_test, y_pred_bow))

vectorizer_tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english', ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
print("TF-IDF Accuracy:", accuracy_score(y_test, y_pred_tfidf))

sentences = [
    ['this', 'movie', 'is', 'very', 'good'],
    ['this', 'film', 'is', 'a', 'good'],
    ['very', 'bad', 'very', 'very', 'bad']
]
w2v_model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=1)
w2v_model.build_vocab(sentences, progress_per=1)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50)

for word in w2v_model.wv.index_to_key:
    print(f"{word} vector:\n{w2v_model.wv[word]}")

vocabs = list(w2v_model.wv.index_to_key)
vectors = np.array([w2v_model.wv[word] for word in vocabs])

tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=5000, random_state=23)
vectors_tsne = tsne_model.fit_transform(vectors)

plt.figure(figsize=(5,5))
plt.scatter(vectors_tsne[:,0], vectors_tsne[:,1])
for i, word in enumerate(vocabs):
    plt.annotate(word, xy=(vectors_tsne[i,0], vectors_tsne[i,1]))
plt.show()
