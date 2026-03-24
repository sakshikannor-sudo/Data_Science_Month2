import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    # Remove unnecessary columns
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical values
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})

    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    return df
