"""
======================
Tabular Classification
======================
"""
import numpy as np

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline


# Get the training data for tabular classification
# Move to Australian to showcase numerical vs categorical
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1
)

numerical = X.columns.to_list()
categorical = []
# numerical.remove('att214')

# Create a proof of concept pipeline!
dataset_properties = {
    'categorical_columns': categorical,
    'numerical_columns': numerical
}
pipeline = TabularClassificationPipeline(dataset_properties=dataset_properties)

# Configuration space
pipeline_cs = pipeline.get_hyperparameter_search_space()
print("Pipeline CS:\n", '_' * 40, f"\n{pipeline_cs}")
config = pipeline_cs.sample_configuration()
print("Pipeline Random Config:\n", '_' * 40, f"\n{config}")
pipeline.set_hyperparameters(config)

# Mock the categories
categorical_columns = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
categories = [np.unique(X[a]).tolist() for a in categorical_columns]

# Fit the pipeline
print("Fitting the pipeline...")
pipeline.fit(X={
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'num_features': len(X),
    'num_classes': len(np.unique(y)),
    'is_small_preprocess': True,
    'X_train': X_train,
    'categories': categories,
})

# Showcase some components of the pipeline
print(pipeline)
