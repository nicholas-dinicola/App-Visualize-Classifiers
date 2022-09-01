import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("Streamlite example")

st.write("""
    # Explore Different Classifiers
    Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifer", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name: str): 
    if dataset_name == "Iris": 
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer": 
        data = datasets.load_breast_cancer()
    else: 
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name: str) -> dict: 
    params = dict()
    if clf_name == "KNN": 
        params["K"] = st.sidebar.slider("K", min_value=1, max_value=15)
    elif clf_name == "SVM": 
        params["C"] = st.sidebar.slider("C", min_value=0.01, max_value=10.0)
    else: 
        params["max_depth"] = st.sidebar.slider("max_depth", min_value=2, max_value=15)
        params["n_estimators"] = st.sidebar.slider("n_estimators", min_value=1, max_value=100)
    return params

params = add_parameter_ui(clf_name=classifier_name)

def get_classifier(clf_name: str, params: dict) -> object: 
    if clf_name == "KNN": 
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM": 
        clf = SVC(C=params["C"])
    else: 
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42, n_jobs=-1)
    return clf

clf = get_classifier(classifier_name, params)

# Classification 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {accuracy}")

# PLOT
# Reduce to two dimensions
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Dimension 1
x1 = X_projected[:, 0]
# Dimension 2
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()
st.pyplot(fig)