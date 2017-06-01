from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich',
    'What the fuck haha'
]

# 应用词频放大法
vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
vectors = vectorizer.fit_transform(corpus).todense()

print(vectors)
print(vectorizer.vocabulary_)

# 越小越相似
print(cosine_distances(vectors[0], vectors[1]))
print(cosine_distances(vectors[0], vectors[2]))
print(cosine_distances(vectors[1], vectors[2]))

