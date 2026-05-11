import yaml
from main.models.helper import ProbeModelWrapper
from main.utils.roots import setup_roots

setup_roots()

CONFIG_PATH = "main/configs/sst2-bert.yaml"

class SST2Bert:
    def __new__(cls, wrapped=False):
        yaml_args = yaml.load(open(CONFIG_PATH), Loader=yaml.Loader)
        list_dataset = yaml_args['dataset']
        probe_model = yaml_args['probe_com']
        probe_model.init(list_dataset.label_word_list)

        if not wrapped:
            return probe_model
        else:
            return ProbeModelWrapper(probe_model)
