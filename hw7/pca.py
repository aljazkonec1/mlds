import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_json("rtvslo_keywords.json")
df['keywords_str'] = df['gpt_keywords'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(min_df=50)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['keywords_str'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(tfidf_df)

pca_df = pd.DataFrame(pca_transformed, columns=['PC1', 'PC2', 'PC3'])

most_important_features= np.argsort(np.linalg.norm(pca.components_.T, axis=1))
loadings = pca.components_.T[most_important_features][-30:]

# Create a 3D biplot
nr_points = 1000

pca_subdf = pca_df.sample(n=nr_points, random_state=42)
tfidf_subdf = tfidf_df.loc[pca_subdf.index]


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_subdf['PC1'], pca_subdf['PC2'], pca_subdf['PC3'], c='blue', marker='o')

for i, loading in enumerate(loadings):
    feature = tfidf_df.columns[most_important_features[i]]
    ax.quiver(0, 0, 0, loadings[i, 0], loadings[i, 1], loadings[i, 2], color='r')
    ax.text(loadings[i, 0], loadings[i, 1], loadings[i, 2], feature, color='r')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D Biplot of PCA-transformed Data')
plt.show()
