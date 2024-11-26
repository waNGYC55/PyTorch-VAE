import yaml
class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        """ this dict is given to init params and will be updated with config and args values
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_yaml(file_path):
    with open(file_path) as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)

    return loaded_config

def load_model_params(config, training=False):
    """load params for tacotron

    Args:
        args (_type_): input attribute_dict from train.py

    Raises:
        ValueError: speaker_number is required

    Returns:
        attribute_dict(_type_): attribute_dict for tacotron
    """

    loaded_config = load_yaml(config)
    attribute_dict = AttributeDict()
    attribute_dict.update(loaded_config)
    return attribute_dict