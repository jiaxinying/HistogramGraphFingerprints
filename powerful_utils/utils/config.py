'''
This file is largely adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
'''
import json
from easydict import EasyDict
import os
import datetime

NUM_LABELS = {'COLLAB':0, 'IMDB-BINARY':0, 'IMDB-MULTI':0, 'MUTAG':11, 'NCI1':37, 'PROTEINS':3, 'PTC_FM':22,'PTC_MR':22, 'QM9':0}
NUM_CLASSES = {'COLLAB':3, 'IMDB-BINARY':2, 'IMDB-MULTI':3, 'MUTAG':2, 'NCI1':2,  'PROTEINS':2, 'PTC_FM':2,'PTC_MR':2,'QM9':1}
LEARNING_RATES = {'COLLAB': 0.0001, 'IMDB-BINARY': 0.00005, 'IMDB-MULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC_FM':0.0001,'PTC_MR':0.0001,
                  'QM9':0.0001}
CHOSEN_EPOCH = {'COLLAB': 150, 'IMDB-BINARY': 100, 'IMDB-MULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC_FM':400,'PTC_MR':400}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file, dataset_name):
    config = get_config_from_json(json_file)
    if dataset_name != '':
        config.dataset_name = dataset_name
    config.num_classes = NUM_CLASSES[config.dataset_name]
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    config.distributed_fold = None  # specific for distrib 10fold - override to use as a flag
    return config


if __name__ == '__main__':
    config = process_config('../configs/10fold_config.json')
    print(config.values())
