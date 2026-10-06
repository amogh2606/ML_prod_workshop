import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
