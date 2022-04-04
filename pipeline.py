from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer
from sklearn.ensemble import RandomForestClassifier

from config_core import config
from scale_processing import scaling
import warnings
warnings.filterwarnings("ignore")


fraud_pipeline = Pipeline([

    # impute numerical variables with the mean
    ('mean_imputation', MeanMedianImputer(imputation_method='mean', variables=config.VARS_WITH_MISSING)),

    # define undersample strategy
    ('SMOTE', SMOTE(random_state=42)),
    #('undersampling', RandomUnderSampler(random_state=42)),

    # scale features
    ('scaling', scaling(config.SCALED_VARS)),

    # build RF classifier
    ('Random_Forest', RandomForestClassifier(max_features=1, max_samples=0.8, min_samples_leaf=1, min_samples_split=4,
                                             n_estimators=120)),
])