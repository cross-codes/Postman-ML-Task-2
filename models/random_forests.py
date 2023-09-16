import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from memory_profiler import profile


@profile
def main():
    start_time = time.time()

    df = pd.read_csv("../processed.csv")
    X = df.drop(columns=["DRK_YN", "sex"], axis=1)
    y = df.DRK_YN

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    model = RandomForestClassifier(n_estimators=100, random_state=69, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
