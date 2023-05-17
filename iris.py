from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mpitree.decision_tree import ParallelDecisionTreeClassifier, WORLD_RANK

if __name__ == "__main__":
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.20, random_state=42
    )

    # Concurrently train a decision tree classifier of `max_depth` 2 among all processes
    clf = ParallelDecisionTreeClassifier(criterion={"max_depth": 2})
    clf.fit(X_train, y_train)

    # Evaluate the performance (e.g., accuracy) of the decision tree classifier
    train_score, test_score = clf.score(X_train, y_train), clf.score(X_test, y_test)

    if not WORLD_RANK:
        print(clf)
        print(f"Train/Test Accuracy: ({train_score:.2%}, {test_score:.2%})")
