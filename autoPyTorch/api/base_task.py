import typing


class Task:

    @typing.no_type_check
    def __init__(
        self,
        **pipeline_kwargs,
    ):
        self._pipeline = pipeline_kwargs['pipeline']
        self._pipeline_config = pipeline_kwargs['pipeline_config']
        self._optimizer = pipeline_kwargs['optimizer']
        self._resource_scheduler = ['resource_scheduler']
        self._backend = ['backend']

    @typing.no_type_check
    def search(
        self,
        X,
        y,
        X_val,
        y_val,
        X_test,
        y_test,
        search_space,
    ):
        pass

    @typing.no_type_check
    def fit(
        self,
        X,
        y,
        X_test,
        y_test,
        model_config,
    ):
        pass

    @typing.no_type_check
    def predict(
        self,
        X_test,
        y_test,
    ):
        pass

    @typing.no_type_check
    def score(
        self,
        X_test,
        y_test,
    ):
        pass

    @typing.no_type_check
    def get_pipeline_config(
        self
    ):
        return self._pipeline_config

    @typing.no_type_check
    def set_pipeline_config(
        self,
        new_pipeline_config,
    ):
        self._pipeline_config = new_pipeline_config

    @typing.no_type_check
    def get_incumbent_results(
        self
    ):
        pass

    @typing.no_type_check
    def get_incumbent_config(
        self
    ):
        pass

    @typing.no_type_check
    def get_default_search_space(
        self
    ):
        pass
