TABULAR_CLASSIFICATION = 1
IMAGE_CLASSIFICATION = 2
TABULAR_REGRESSION = 3
IMAGE_REGRESSION = 4

REGRESSION_TASKS = [TABULAR_REGRESSION, IMAGE_REGRESSION]
CLASSIFICATION_TASKS = [TABULAR_CLASSIFICATION, IMAGE_CLASSIFICATION]

TABULAR_TASKS = [TABULAR_CLASSIFICATION, TABULAR_REGRESSION]
IMAGE_TASKS = [IMAGE_CLASSIFICATION, IMAGE_REGRESSION]
TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS

TASK_TYPES_TO_STRING = \
    {TABULAR_CLASSIFICATION: 'tabular_classification',
     IMAGE_CLASSIFICATION: 'image_classification',
     TABULAR_REGRESSION: 'tabular_regression',
     IMAGE_REGRESSION: 'image_regression'}

STRING_TO_TASK_TYPES = \
    {'tabular_classification': TABULAR_CLASSIFICATION,
     'image_classification': IMAGE_CLASSIFICATION,
     'tabular_regression': TABULAR_REGRESSION,
     'image_regression': IMAGE_REGRESSION}

# Output types have been defined as in scikit-learn type_of_target
# (https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html)
BINARY = 10
CONTINUOUSMULTIOUTPUT = 11
MULTICLASS = 12
CONTINUOUS = 13
MULTICLASSMULTIOUTPUT = 14

OUTPUT_TYPES = [BINARY, CONTINUOUSMULTIOUTPUT, MULTICLASS, CONTINUOUS]

OUTPUT_TYPES_TO_STRING = \
    {BINARY: 'binary',
     CONTINUOUSMULTIOUTPUT: 'continuous-multioutput',
     MULTICLASS: 'multi-class',
     CONTINUOUS: 'continuous',
     MULTICLASSMULTIOUTPUT: 'multiclass-multioutput'}

STRING_TO_OUTPUT_TYPES = \
    {'binary': BINARY,
     'continuous-multioutput': CONTINUOUSMULTIOUTPUT,
     'multi-class': MULTICLASS,
     'continuous': CONTINUOUS,
     'multiclass-multioutput': MULTICLASSMULTIOUTPUT}
