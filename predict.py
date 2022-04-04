import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


_fraud_pipeline = joblib.load('save/pipeline.joblib')


def make_prediction():
    '''Make predictions'''

    # read test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # predictions of test data
    y_test_pred = _fraud_pipeline.predict(X_test)

    print('{} Recall Score: {:.4f}'.format('RF', recall_score(y_test, y_test_pred)))
    print('{} Precision Score: {:.4f}'.format('RF', precision_score(y_test, y_test_pred)))
    print('{} F1 Score: {:.4f}'.format('RF', f1_score(y_test, y_test_pred)))
    print('{} Accuracy Score: {:.4f}'.format('RF', accuracy_score(y_test, y_test_pred)))
    return y_test_pred


if __name__ == "__main__":
    make_prediction()