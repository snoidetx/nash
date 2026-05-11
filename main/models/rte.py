from main.models.helper import ProbeModelWrapper
from main.utils.roots import setup_roots

setup_roots()

import yaml
from dataset import *
from ntk import *
from probe import *
from dvutils.Data_Shapley import *



class RTEBert:
    def __new__(cls, wrapped=False):
        CONFIG_PATH = "main/configs/rte-bert.yaml"
        yaml_args = yaml.load(open(CONFIG_PATH), Loader=yaml.Loader)
        list_dataset = yaml_args['dataset']
        probe_model = yaml_args['probe_com']
        probe_model.init(list_dataset.label_word_list)

        if not wrapped:
            return probe_model
        else:
            return ProbeModelWrapper(probe_model)

class RTELlama:
    def __new__(cls, wrapped=False):
        CONFIG_PATH = "main/configs/rte-llama.yaml"
        yaml_args = yaml.load(open(CONFIG_PATH), Loader=yaml.Loader)
        list_dataset = yaml_args['dataset']
        probe_model = yaml_args['probe_com']
        probe_model.init(list_dataset.label_word_list)

        if not wrapped:
            return probe_model
        else:
            return ProbeModelWrapper(probe_model)
