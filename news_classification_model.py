import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Logistic Regression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump

# --------------------------------------

# 1. LOAD THE DATASET 

data = pd.read_csv('dataset.csv')

# inspect the dataset 

data.head()

# output size
len(data)
#2225

# output columns
data.columns
# Index(['Unnamed: 0', 'news', 'type'], dtype='object')

# rename column
data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# output labels 
print(data['type'].unique())
# ['business' 'entertainment' 'politics' 'sport' 'tech']

# verify the number of items per category 
print(data['type'].value_counts())
#type
#sport            511
#business         510
#politics         417
#tech             401
#entertainment    386

# the dataset is relatively well balanced 

# --------------------------------------

# 2. GENERATE TEXTUAL EMBEDDINGS

# will experiment with TF-IDF and pre-trained BERT embeddings

# 2. 1. TF-IDF 

vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(data['news']).toarray()

pca = PCA(n_components=2)
X_tfidf_2d = pca.fit_transform(X_tfidf)

# 2. 2. BERT

# load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts, batch_size=10):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)  
        with torch.no_grad():
            output = model(**encoded_input)
        embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy()) 

    # Concatenate batch embeddings
    return np.vstack(embeddings)
    
embeddings = get_bert_embeddings(data['news'].tolist())


# --------------------------------------

# 3. REPRESENT THE DATASET USING A 2D GRAPH 

# 3. 1. TF-IDF 

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tfidf_2d[:, 0], X_tfidf_2d[:, 1], c=data['type'].astype('category').cat.codes, cmap='viridis', label=data['type'])
plt.title('TF-IDF Embeddings Visualized in 2D')
plt.colorbar(scatter)
plt.show()

# 3. 2. BERT

# reduce embeddings' dimensions
pca_bert = PCA(n_components=2)
embeddings_2d = pca_bert.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data['type'].astype('category').cat.codes, cmap='viridis', label=data['type'])
plt.title('BERT Embeddings Visualized in 2D')
plt.colorbar(scatter)
plt.show()


# 3. 3. analysis 

# The plots are similar in terms of the separation of labels: one of the clusters does not overlap whilst the others overlap a little. There is slightly more overlap with the TF-IDF embeddings. 
# Therefore, BERT embeddings are opted for (if simplicity were of key importance, TF-IDF would be the more suitable option).

# --------------------------------------

# 4. TRAIN A CLASSIFIER 

# simple train/test division of 8:2 due to the small dataset size 
X_train, X_test, y_train, y_test = train_test_split(embeddings, data['type'], test_size=0.2, random_state=42)

# I will train a logistic regression and a SVM classifier and determine which one achieves better accuracy 
 
# 4.1 Logistic regression

log_reg = Logistic Regression(max_iter=1000)

log_reg.fit(X_train, y_train)

# make predictions and evaluate accuracy
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))
# save the report
with open('/logistic_regression_report.txt', 'w') as f:
    f.write(log_reg_report)

# Logistic Regression Accuracy: 0.9775


# 4.2 SVM 

svm = SVC()

svm.fit(X_train, y_train)

# make predictions and evaluate accuracy
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))

# SVM Accuracy: 0.9708

# 4.3 analysis 

# Logistic regression achieves higher accuracy. Also, its lowest F1-score is 0.97, whilst the SVM models' is 0.95. SVM has lower precision and recall for "business" and "tech" labels.
# Therefore, logistic regression is opted for. 

# 4.4 Grid Search for Logistic Regression 

log_reg = LogisticRegression(max_iter=1000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
    'penalty': ['l1', 'l2', 'elasticnet', 'none']  
}

grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Higher accuracy was not achieved, so the first trained model is retained.

# save the initial model 
dump(log_reg, 'logistic_regression_model.joblib')

# 5. FINAL EVALUATION RESULT

log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print("Model Accuracy:", log_reg_accuracy)







