from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_test, predictions))
