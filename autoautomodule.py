from collections import defaultdict
import json
import sys
from typing import Any, Dict, List, Optional, Union

import ConfigSpace
import numpy as np
import pandas as pd
import sklearn.ensemble
import openml


class OneVSOneSelector(object):
    def __init__(self, configuration, default_strategy_idx, rng):
        self.configuration = configuration
        self.default_strategy_idx = default_strategy_idx
        self.rng = rng
        self.models = None
        self.target_indices = None
        self.selectors_ = None
        self.weights_ = {}
        self.X_train = None

    def fit(self, X, y, methods, minima, maxima):
        self.X_train = X.copy()
        target_indices = np.array(list(range(y.shape[1])))
        models = dict()
        weights = dict()
        for i in range(len(target_indices)):
            models[i] = dict()
            weights[i] = dict()
            for j in range(i + 1, len(target_indices)):
                y_i_j = y[:, i] < y[:, j]
                min_i = np.array([minima[methods[i]][task_id] for task_id in X.index])
                max_i = np.array([maxima[methods[i]][task_id] for task_id in X.index])
                min_j = np.array([minima[methods[j]][task_id] for task_id in X.index])
                max_j = np.array([maxima[methods[j]][task_id] for task_id in X.index])

                minimum = np.minimum(min_i, min_j)
                maximum = np.maximum(max_i, max_j)
                diff = maximum - minimum
                diff[diff == 0] = 1
                normalized_y_i = (y[:, i].copy() - minimum) / diff
                normalized_y_j = (y[:, j].copy() - minimum) / diff

                weights_i_j = np.abs(normalized_y_i - normalized_y_j)
                """
                if np.all([target == y_i_j[0] for target in y_i_j]):
                    n_zeros = int(np.ceil(len(y_i_j) / 2))
                    n_ones = int(np.floor(len(y_i_j) / 2))
                    base_model = sklearn.dummy.DummyClassifier(strategy='constant', constant=y_i_j[0])
                    base_model.fit(
                        X.values,
                        np.array(([[0]] * n_zeros) + ([[1]] * n_ones)).flatten(),
                        sample_weight=weights_i_j,
                    )
                """
                if True:
                    base_model = sklearn.ensemble.RandomForestClassifier(
                        random_state=self.rng,
                        n_estimators=500,
                        oob_score=True,
                        bootstrap=True,
                        min_samples_split=self.configuration['min_samples_split'],
                        min_samples_leaf=self.configuration['min_samples_leaf'],
                        max_features=int(np.rint(X.shape[1] ** self.configuration['max_features'])),
                    )
                    base_model.fit(X.values, y_i_j, sample_weight=weights_i_j)
                models[i][j] = base_model
                weights[i][j] = weights_i_j
        self.models = models
        self.weights_ = weights
        self.target_indices = target_indices

    def predict(self, X):

        if self.default_strategy_idx is not None:
            use_prediction = False
            counter = 0
            te = X.copy().flatten()
            assert len(te) == 3
            for _, tr in self.X_train.iterrows():
                tr = tr.to_numpy()
                if tr[0] >= te[0] and tr[1] >= te[1] and tr[2] >= te[2]:
                    counter += 1

            if counter > 0:
                use_prediction = True

            if not use_prediction:
                print('Backup', counter)
                return np.array([1 if i == self.default_strategy_idx else 0 for i in self.target_indices])
            print('No backup', counter)

        X = X.reshape((1, -1))

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                raw_predictions[(i, j)] = self.models[i][j].predict(X)

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
                    #prediction = raw_predictions[(i, j)][x_idx]
                    #wins[i] += prediction[1]
                    #wins[j] += prediction[0]
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions

    def predict_oob(self, X):

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                rp = self.models[i][j].oob_decision_function_.copy()
                rp[np.isnan(rp)] = 0
                rp = np.nanargmax(rp, axis=1)
                raw_predictions[(i, j)] = rp

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
                    #prediction = raw_predictions[(i, j)][x_idx]
                    #wins[i] += prediction[1]
                    #wins[j] += prediction[0]
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions

def get_metafeatures(task_id, feat_list=None):
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)
    metafeatures = dataset.qualities

    if feat_list is not None:
        reduced_metafeatures = {}

        for feat_name in feat_list:
            reduced_metafeatures[feat_name] = metafeatures[feat_name]

        return reduced_metafeatures
    return metafeatures


def get_y_test_dummy(method, task_ids):
    # TODO write for real
    # So we want only one learning curve for each task_id, config combination
    metafeatures = [get_metafeatures(task_id, ['NumberOfInstances'])['NumberOfInstances'] for task_id in task_ids]
    LEARN_CURVE_LEN = 100
    num_configs = 1 # if multiple configs were used for one strategy
    perfs = np.array([mf > 1000 for mf in metafeatures],dtype=np.float)[:,None,None]
    print(perfs)
    if method == 'wei':
        return perfs
    else:
        return 1. - perfs


    # return np.random.rand(len(task_ids), num_configs, LEARN_CURVE_LEN)

def test():
    openml_task_ids = [232, 236, 241, 245, 253, 254, 256, 258, 260, 262, 267, 271, 273, 275, 279, 288, 336, 340, 2119, 2120, 2121, 2122,
                       2123, 2125, 2356, 3044, 3047, 3048, 3049, 3053, 3054, 3055, 75089, 75092, 75093, 75098, 75100, 75108, 75109, 75112,
                       75114, 75115, 75116, 75118, 75120, 75121, 75125, 75126, 75129, 75131, 75133, 75134, 75136, 75139, 75141, 75142,
                       75143, 75146, 75147, 75148, 75149, 75153, 75154, 75156, 75157, 75159, 75161, 75163, 75166, 75169, 75171, 75173,
                       75174, 75176, 75178, 75179, 75180, 75184, 75185, 75187, 75192, 75195, 75196, 75199, 75210, 75212, 75213, 75215,
                       75217, 75219, 75221, 75223, 75225, 75232, 75233, 75234, 75235, 75236, 75237, 75239, 75250, 126021, 126024, 126028,
                       126030, 126031, 146574, 146575, 146576, 146577, 146578, 146583, 146586, 146592, 146593, 146594, 146596, 146597,
                       146600, 146601, 146602, 146603, 146679, 166859, 166866, 166872, 166875, 166882, 166897, 166905, 166906, 166913,
                       166915, 166931, 166932, 166944, 166950, 166951, 166953, 166956, 166957, 166958, 166959, 166970, 166996, 167085,
                       167086, 167087, 167088, 167089, 167090, 167094, 167096, 167097, 167099, 167100, 167101, 167103, 167105, 167106,
                       167202, 167203, 167204, 167205, 168785, 168791, 189779, 189786, 189828, 189829, 189836, 189840, 189841, 189843,
                       189844, 189845, 189846, 189857, 189858, 189859, 189863, 189864, 189869, 189870, 189875, 189878, 189880, 189881,
                       189882, 189883, 189884, 189887, 189890, 189893, 189894, 189899, 189900, 189902, 190154, 190155, 190156, 190157,
                       190158, 190159, 211720, 211721, 211722, 211723, 211724]

    desired_metafeatures = ['NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances']

    metafeatures = {task_id : get_metafeatures(task_id, desired_metafeatures) for task_id in openml_task_ids}
    dummy_strategies = ['wei','dei','lei']
    build(dummy_strategies, pd.DataFrame(metafeatures).transpose(), np.random.RandomState(), openml_task_ids, ['wei'], get_y_test_dummy)

def build(
    strategies: List[str], # strategy ids (keys of configurations)
    metafeatures: pd.DataFrame, # the structure [str, np.array] is a mapping from task id to meta features of particular ds, order of task ids like in `task_ids`
    random_state: np.random.RandomState,
    task_ids: List[int], # all task ids to consider used
    default_strategies: List[str], # list of stragiy ids (like the ones in `strategies`) that will sequentially be searched for a feasible default strategy for the given set of `strategies`. See arouond line 227. Used for backup I think.
    get_y_test # a method like described in the default
):
    performance_matrix = pd.DataFrame(columns=strategies,index=task_ids)

    minima_for_methods = dict()
    minima_for_tasks = dict()
    maxima_for_methods = dict()
    maxima_for_tasks = dict()

    for method in strategies:
        y_test = get_y_test(method, task_ids)
        matrix = y_test.copy()
        minima = np.nanmin(np.nanmin(matrix, axis=2), axis=1)
        minima_as_dicts = {
            task_id: minima[i] for i, task_id in enumerate(task_ids)
        }
        maxima = np.nanmax(np.nanmax(matrix, axis=2), axis=1)
        maxima_as_dicts = {
            task_id: maxima[i] for i, task_id in enumerate(task_ids)
        }
        minima_for_methods[method] = minima_as_dicts
        maxima_for_methods[method] = maxima_as_dicts
        performance_matrix[method] = pd.Series(maxima_as_dicts) # define performance as max across configs of a task
        diff = maxima - minima
        diff[diff == 0] = 1
        del matrix

    for task_id in task_ids:
        min_for_task = 1.0
        for method in strategies:
            min_for_task = min(min_for_task, minima_for_methods[method][task_id])
        minima_for_tasks[task_id] = min_for_task
        max_for_task = 0.0
        for method in strategies:
            max_for_task = max(max_for_task, maxima_for_methods[method][task_id])
        maxima_for_tasks[task_id] = max_for_task

    # now we have min/max per task/strategy this is later used to normalize and to evaluate tasks!?

    # Classification approach - generate data
    y_values = []
    task_id_to_idx = {}
    for i, task_id in enumerate(task_ids):
        values = []
        task_id_to_idx[task_id] = len(y_values)
        for method in strategies:
            val = performance_matrix[method][task_id]
            values.append(val)
        y_values.append(values)
    y_values = np.array(y_values)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('min_samples_split', 2, 20, log=True,
                                                 default_value=2)
    )
    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('min_samples_leaf', 1, 20, log=True,
                                                 default_value=1)
    )
    cs.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter('max_features', 0, 1, default_value=0.5)
    )
    cs.seed(random_state.randint(0, 1000))
    meta_configurations = [cs.get_default_configuration()] + cs.sample_configuration(size=50)

    default_strategy = None
    for tmp in default_strategies:
        if tmp in strategies:
            default_strategy = tmp
            break
    if default_strategy is None:
        raise ValueError('Found no legal default strategy!')
    print('Using default strategy', default_strategy)

    best_loss = np.inf
    best_model = None
    best_sample_weight = None
    best_oob_predictions = None
    training_data = {}
    training_data['metafeatures'] = metafeatures.to_dict()
    training_data['y_values'] = [[float(_) for _ in row] for row in y_values]
    training_data['strategies'] = strategies
    training_data['minima_for_methods'] = minima_for_methods
    training_data['maxima_for_methods'] = maxima_for_methods

    for meta_configuration in meta_configurations:
        # Here for each meta_config we train a selector, so that given a task we can predict the performance of different strategies
        selector = OneVSOneSelector(
            configuration=meta_configuration,
            default_strategy_idx=strategies.index(default_strategy),
            rng=random_state,
        )
        selector.fit(
            X=metafeatures,
            y=y_values,
            methods=strategies,
            minima=minima_for_methods,
            maxima=maxima_for_methods,
        )

        predictions = selector.predict_oob(metafeatures)
        error = []
        for i in range(len(predictions)):
            error_i = y_values[i][np.argmax(predictions[i])] != np.min(y_values[i])
            error.append(error_i)
        error = np.array(error)

        sample_weight = []
        for sample_idx, task_id in enumerate(metafeatures.index):
            prediction_idx = np.argmax(predictions[sample_idx])
            y_true_idx = np.argmin(y_values[sample_idx])
            diff = maxima_for_tasks[task_id] - minima_for_tasks[task_id]
            diff = 1 if diff == 0 else diff
            normalized_predicted_sample = (y_values[sample_idx, prediction_idx] - minima_for_tasks[task_id]) / diff
            normalized_y_true = (y_values[sample_idx, y_true_idx] - minima_for_tasks[task_id]) / diff
            weight = np.abs(normalized_predicted_sample - normalized_y_true)
            sample_weight.append(weight)
        sample_weight = np.array(sample_weight)
        loss = np.sum(error.astype(int) * sample_weight)

        # print(np.sum(train_error), np.sum(error), train_loss, loss, best_loss)
        if loss < best_loss:
            best_loss = loss
            best_model = selector
            best_sample_weight = sample_weight
            best_oob_predictions = predictions

    training_data['configuration'] = best_model.configuration.get_dictionary()
    with open('/tmp/training_data.json', 'wt') as fh:
        json.dump(training_data, fh, indent=4)

    # print('Best predictor OOB score', best_loss)

    # for i in best_model.models:
    #     for j in best_model.models[i]:
    #         print(best_model.models[i][j].feature_importances_)

    regrets_rf = []
    regret_random = []
    regret_oracle = []
    base_method_regets = {method: [] for method in strategies}
    # Normalize each column given the minimum and maximum ever observed on these tasks
    normalized_regret = np.array(y_values, dtype=float)
    for task_id in performance_matrix.index:
        task_idx = task_id_to_idx[task_id]
        minima = np.inf
        maxima = -np.inf
        for method in strategies:
            minima = min(minima_for_methods[method][task_id], minima)
            maxima = max(maxima_for_methods[method][task_id], maxima)
        diff = maxima - minima
        if diff == 0:
            diff = 1
        normalized_regret[task_idx] = (normalized_regret[task_idx] - minima) / diff

        prediction = best_oob_predictions[task_idx]
        prediction_idx = np.argmax(prediction)
        regrets_rf.append(float(normalized_regret[task_idx][prediction_idx]))
        regret_random.append(
            [float(value) for value in np.random.choice(normalized_regret[task_idx], size=1000, replace=True)]
        )
        regret_oracle.append(float(np.min(normalized_regret[task_idx])))
        for method_idx, method in enumerate(strategies):
            base_method_regets[method].append(normalized_regret[task_idx][method_idx])

    normalized_regret_dataframe = pd.DataFrame(normalized_regret,
                                               columns=performance_matrix.columns)
    full_oracle_perf = normalized_regret_dataframe.min(axis=1).mean()
    print('Oracle performance', full_oracle_perf)
    for i in range(normalized_regret_dataframe.shape[1]):
        subset_oracle_perf = normalized_regret_dataframe.drop(normalized_regret_dataframe.columns[i], axis=1).min(axis=1).mean()
        print(normalized_regret_dataframe.columns[i], subset_oracle_perf - full_oracle_perf)

    print('Regret rf', np.mean(regrets_rf))
    print('Regret random', np.mean(regret_random))
    print('Regret oracle', np.mean(regret_oracle))

    return best_model

if __name__ == '__main__':
    print(test())