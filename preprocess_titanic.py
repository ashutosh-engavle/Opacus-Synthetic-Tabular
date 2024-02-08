import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_titanic():
    df = pd.read_csv("Titanic.csv", index_col="PassengerId")
    # Dropped because these columns seem irrelevant for analysis, there's no difference in keeping them
    df = df.drop(["Name", "Ticket", "Fare", "Cabin"], axis=1).reset_index(drop=True)

    object_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[object_cols] = df[object_cols].fillna("unknown")
    df = df.dropna(subset=numeric_cols)

    df = pd.get_dummies(df).astype(int)
    features = df.drop("Survived", axis=1)
    target = df["Survived"]
    combined_df = pd.concat([target, features], axis=1)
    combined_df.to_csv("titanic_processed.csv", index=False)
    print(combined_df)

    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    train_df.to_csv("titanic_processed_train.csv", index=False)
    test_df.to_csv("titanic_processed_test.csv", index=False)

if __name__ == '__main__':
    preprocess_titanic()