#!/usr/bin/env python3

from mpi4py import MPI
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mpitree.decision_tree import ParallelDecisionTreeClassifier, WORLD_RANK

if __name__ == "__main__":
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.iloc[:, :2], iris.target, test_size=0.20, random_state=42
    )

    # Start the clock once all processes constructed their train-test sets
    start_time = MPI.Wtime()

    # Concurrently train a decision tree classifier of `max_depth` 2 among all processes
    tree = ParallelDecisionTreeClassifier(criterion={"max_depth": 2})
    tree.fit(X_train, y_train)

    # Evaluate the performance (e.g., accuracy) of the decision tree classifier
    train_score, test_score = tree.score(X_train, y_train), tree.score(X_test, y_test)

    # Stop the clock w.r.t each process
    end_time = MPI.Wtime()

    if not WORLD_RANK:
        print(tree)
        print(f"Train/Test Accuracy: ({train_score:.2%}, {test_score:.2%})")
        print(f"Parallel Execution Time: {(end_time - start_time) * 1000:.2f}ms")
