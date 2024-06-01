import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import vispy.scene
from vispy.scene import visuals


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import classla 
classla.download('sl')

nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=False)


# Download necessary NLTK data

# stemmer = SnowballStemmer("slovene")

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower(), language='slovene')
    # Remove punctuation and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    # Stem the tokens
    # stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('slovene'))


    stemmed_tokens = [token for token in tokens if token not in stop_words]

    stemmed_tokens = [token.words[0].lemma for token in nlp(text).sentences[0].tokens if token.text not in stop_words]

    return ' '.join(stemmed_tokens)


# Load data
df = pd.read_json("rtvslo_keywords.json")
# df['keywords_str'] = df['gpt_keywords'].apply(lambda x: ' '.join(word.replace(' ', '_') for word in x))
df['keywords_str'] = df['gpt_keywords'].apply(lambda x: ' '.join(x))

preprocessed_text = [preprocess_text(text) for text in df['keywords_str']]
print(preprocessed_text)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(min_df=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text)
print(tfidf_matrix)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Perform PCA
pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(tfidf_df)

pca_df = pd.DataFrame(pca_transformed, columns=['PC1', 'PC2', 'PC3'])

# Determine the most important features
most_important_features = np.argsort(np.linalg.norm(pca.components_.T, axis=1))
print(most_important_features )
loadings = pca.components_.T[most_important_features][-10:]

# Sample data points for plotting
nr_points = 20000
pca_subdf = pca_df.sample(n=nr_points, random_state=42)
tfidf_subdf = tfidf_df.loc[pca_subdf.index]

# Create vispy canvas and view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Plot the PCA points
scatter = visuals.Markers()
scatter.set_data(pca_subdf[['PC1', 'PC2', 'PC3']].values, face_color='blue', size=5)
view.add(scatter)

# Plot the arrows (loadings)
for i, loading in enumerate(loadings):
    feature = tfidf_df.columns[most_important_features[i]]
    arrow = visuals.Arrow(pos=[[0, 0, 0], loadings[i]], color='red', arrow_size=10, arrow_type='angle_60')
    view.add(arrow)
    text = visuals.Text(feature, color='red', pos=loadings[i], anchor_x='left', anchor_y='bottom')
    view.add(text)

# Add axis labels
x_axis = visuals.Axis(pos=[[0, 0, 0], [1, 0, 0]], tick_direction=(0, -1, 0), axis_label='PC1')
y_axis = visuals.Axis(pos=[[0, 0, 0], [0, 1, 0]], tick_direction=(-1, 0, 0), axis_label='PC2')
z_axis = visuals.Axis(pos=[[0, 0, 0], [0, 0, 1]], tick_direction=(0, 0, -1), axis_label='PC3')

view.add(x_axis)
view.add(y_axis)
view.add(z_axis)

# Run the application
if __name__ == '__main__':
    vispy.app.run()