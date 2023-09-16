import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

start_time = time.time()

df = pd.read_csv("../processed.csv")
X = df.drop(columns=["DRK_YN", "sex"], axis=1)
y = df.DRK_YN

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=69
)

svmc = SVC(kernel="rbf", C=1.0)
svmc.fit(X_train, y_train)


y_pred = svmc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred.round())
report = classification_report(y_test, y_pred.round())

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.2f} seconds")
