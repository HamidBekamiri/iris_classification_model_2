import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()

iris = iris[:130]

# Train a random forest classifier
rf = RandomForestClassifier()
rf.fit(iris.data, iris.target)

# Save the model as an artifact
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)
