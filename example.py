from sklearn.datasets import load_iris
from mpitree.tree import ParallelDecisionTreeClassifier

iris = load_iris()
X, y = iris.data[:, :2], iris.target

clf = ParallelDecisionTreeClassifier(max_depth=11).fit(X, y)

if not clf.WORLD_RANK:
    print(clf)
