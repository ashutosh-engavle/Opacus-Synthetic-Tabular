import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from generate_synthetic_data import generate_synthetic_data_llms
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def generate():
    test_df = pd.read_csv('titanic_processed_test.csv')
    df_gen = generate_synthetic_data_llms(
        batch_size=51,
        experiment_name="Titanic_Epochs10",
        columns=list(test_df.columns),
        EPSILON=0.1,
    )
    df_gen["Survived"] = df_gen["Survived"].astype(int)
    print(df_gen)
    X_train = df_gen.drop("Survived", axis=1)
    y_train = df_gen["Survived"]

    test_df = test_df[df_gen.columns]
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    accuracy, fpr = train_and_evaluate_classifier(
        clf, X_train, y_train, X_test, y_test, "Titanic"
    )
    train_accuracy = get_train_accuracy(clf, X_train, y_train)
    print(
        f"Accuracy = {accuracy*100}%, FPR = {fpr*100}%, Train Accuracy = {train_accuracy*100}%"
    )

def get_train_accuracy(clf, X_train, y_train):
    y_pred = clf.predict_proba(X_train)[:, 1]
    y_pred_labels = (y_pred > 0.5).astype(int)
    y_train = y_train.astype(int)
    return accuracy_score(y_train, y_pred_labels)

def train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test, test_index):
    clf.fit(X_train, y_train)
    y_test = y_test.astype(int)

    # print(X_test)

    y_pred = clf.predict_proba(X_test)[:, 1]
    y_pred_labels = (y_pred > 0.5).astype(int)  # Actual

    cm = confusion_matrix(y_test, y_pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    accuracy = accuracy_score(y_test, y_pred_labels)

    print(
        f"Testset {test_index} : {type(clf).__name__}: Accuracy : {accuracy:.2f}, False Positive Rate : {fpr:.2f}"
    )

    return accuracy, fpr

if __name__ == '__main__':
    generate()