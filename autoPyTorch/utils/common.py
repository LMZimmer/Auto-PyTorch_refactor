from typing import Any, Dict, Iterable, NamedTuple, Type


class FitRequirement(NamedTuple):
    """
    A class that holds inputs required to fit a pipeline. Also indicates wether
    requirements have to be user specified or are generated by the pipeline itself.

    Attributes:
    name: The name of the variable expected in the input dictionary
    supported_types: An iterable of all types that are supported
    user_defined: If false, this requirement does not have to be given to the pipeline
    """

    name: str
    supported_types: Iterable[Type]
    user_defined: bool
    dataset_property: bool

    def __str__(self) -> str:
        """
        String representation for the requirements
        """
        return "Name: %s | Supported types: %s | User defined: %s | Dataset property: %s" % (
            self.name, self.supported_types, self.user_defined, self.dataset_property)
      

def replace_prefix_in_config_dict(config: Dict[str, Any], prefix: str, replace: str = "") -> Dict[str, Any]:
    """
    Replace the prefix in all keys with the specified replacement string (the empty string by
    default to remove the prefix from the key). The functions makes sure that the prefix is a proper config
    prefix by checking if it ends with ":", if not it appends ":" to the prefix.

    :param config: config dictionary where the prefixed of the keys should be replaced
    :param prefix: prefix to be replaced in each key
    :param replace: the string to replace the prefix with
    :return: updated config dictionary
    """
    # make sure that prefix ends with the config separator ":"
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    # only replace first occurrence of the prefix
    return {k.replace(prefix, replace, 1): v
            for k, v in config.items() if
            k.startswith(prefix)}
