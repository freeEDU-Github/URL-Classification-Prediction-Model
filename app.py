import pandas as pd
import streamlit as st
import numpy as np
import pickle as p

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#dataset
data = pd.read_csv("num.csv")
data_sample = pd.read_csv("importance.csv")

#Text
st.title('URL Classification Prediction Model')
st.write(
          'The main objective of this project is to thoroughly examine and understand the various techniques for detecting malicious URLs using machine learning. These techniques will be used to predict whether a given URL is benign, related to defacement, malware, phishing, or spam.')

st.write("You can test the sample data below")

st.dataframe(data_sample)

path_token_count = st.number_input("Path Token Count", step=1e-5, format="%.4f")
average_domain_token = st.number_input("Average Domain Token", step=1e-5, format="%.4f")
entropy_url = st.number_input("Entropy URL", step=1e-5, format="%.4f")
charcompvowels = st.number_input("Charcomp Vowels", step=1e-5, format="%.4f")
charcompace = st.number_input("Charcomp Ace", step=1e-5, format="%.4f")
path_url_ratio = st.number_input("Path Url Ratio", step=1e-5, format="%.4f")
domain_url_ratio = st.number_input("Domain URL Ratio", step=1e-5, format="%.4f")
symbol_count_url = st.number_input("Symbol Count URL", step=1e-5, format="%.4f")

features = ['path_token_count', 'average_domain_token', 'entropy_url', 'charcompvowels', 'charcompace', 'path_url_ratio', 'domain_url_ratio', 'symbol_count_url']

# Labeling X and y features
X = data[features]
y = data['url_type']
#X = np.nan_to_num(X)

# Training the data using decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
dtc_clf = DecisionTreeClassifier()
dtc_clf.fit(X_train,y_train)
predict_val = dtc_clf.predict([[path_token_count, average_domain_token, entropy_url, charcompvowels, charcompace, path_url_ratio, domain_url_ratio, symbol_count_url]])
predict_val = float(predict_val)

if predict_val == 1:
    st.info("URL Type: Benign")

elif predict_val == 0:
    st.info("URL Type: Defacement")

elif predict_val == 2:
    st.info("URL Type: Malware")

elif predict_val == 3:
    st.info("URL Type: Phishing")

else:
    st.info("URL Type: Spam")
