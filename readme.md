# Take_Home_MLE (Finder)
## Requirements

To install requirements:
```
pip install -r requirements.txt
```

## Files

- config.py and config_core.py are used for configuration.

- scale_processing.py is used to scale features in the format of Pipeline.

- pipeline.py is used to construct the machine learning pipeline, including data processing and model training.

- train_pipeline.py is used to run pipeline.py and save the trained pipeline.

- predict.py is used to make predictions for the test data.

## Procedures

- Please put dataset_TakeHome.csv into data directory.

- Obtain the trained model:
```
python train_pipeline.py
```
- Obtain the test results:
```
python predict.py
```

## Others
- data: directory that contains data files.
- save: directory that contains the saved model files
 and other files, such as hyperparameters. 