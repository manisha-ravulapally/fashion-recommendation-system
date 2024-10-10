import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Initialize the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, tf.keras.layers.GlobalMaxPooling2D()])

# Fit the NearestNeighbors model using the training data
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

def convert_ids_to_paths(ids, base_path='images\\'):
    return [f'{base_path}{id}' for id in ids]
def process_saree_column(csv_file):
    df = pd.read_csv('relevant_images.csv')
    saree_ids = df['watches'].dropna().astype(int)
    saree_paths = convert_ids_to_paths(saree_ids.tolist())
    return saree_paths

saree_paths = process_saree_column('relevant_images.csv')
test_data=[('uploads/watch.jpg',saree_paths)]

def precision_at_k(relevant_items, recommended_items, k):
    relevant_at_k = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
    return sum(relevant_at_k) / k

def recall_at_k(relevant_items, recommended_items, k):
    relevant_in_recommendations = [1 if item in recommended_items[:k] else 0 for item in relevant_items]
    return sum(relevant_in_recommendations) / len(relevant_items)

def f1_score_at_k(relevant_items, recommended_items, k):
    precision = precision_at_k(relevant_items, recommended_items, k)
    recall = recall_at_k(relevant_items, recommended_items, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def accuracy_at_k(relevant_items, recommended_items, k):
    # Number of relevant items in the top k recommendations
    relevant_count = sum([1 if item in relevant_items else 0 for item in recommended_items[:k]])
    # Total number of items recommended
    total_recommended = len(recommended_items[:k])
    return relevant_count / total_recommended if total_recommended > 0 else 0

# Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# Evaluate model performance
for img_path, relevant_items in test_data:
    features = feature_extraction(img_path, model)
    distances, indices = neighbors.kneighbors([features], n_neighbors=5)
    recommended_items = [filenames[i].split('/')[-1].split('.')[0] for i in indices[0]]

    precision = precision_at_k(relevant_items, recommended_items, 5)
    recall = recall_at_k(relevant_items, recommended_items, 5)
    f1 = f1_score_at_k(relevant_items, recommended_items, 5)
    accuracy = accuracy_at_k(relevant_items, recommended_items, 5)

    print(f"Results for {img_path}:")
    print(f"Precision@5: {precision}")
    print(f"Recall@5: {recall}")
    print(f"F1-Score@5: {f1}")
    print(f"Accuracy@5: {accuracy}")
