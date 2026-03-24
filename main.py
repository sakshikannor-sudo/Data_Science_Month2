from src.data_preprocessing import load_data
from src.data_preprocessing import preprocess_data

from src.train_model import train_model

from src.evaluate_model import evaluate_model


def main():

    print("Loading dataset...")

    df = load_data("data/titanic.csv")

    print("Preprocessing data...")

    df = preprocess_data(df)

    print("Training model...")

    model, X_test, y_test = train_model(df)

    print("Evaluating model...")

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":

    main()