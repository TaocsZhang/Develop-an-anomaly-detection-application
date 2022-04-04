import pandas as pd
import joblib
from pipeline import fraud_pipeline
from sklearn.model_selection import train_test_split
from config_core import config


def run_training() -> None:
    """Train the model."""

    # read training data
    data = pd.read_csv('data/dataset_TakeHome.csv')

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([config.target], axis=1),
        data[config.target],
        test_size=config.test_size,
        random_state=0,
    )

    # save test set
    X_test.to_csv('data/x_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    # fit model
    fraud_pipeline.fit(X_train, y_train)

    # save the trained model
    joblib.dump(fraud_pipeline, 'save/pipeline.joblib')


if __name__ == "__main__":
    run_training()