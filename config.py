# @Time : 2020/12/17 14:37
# @Author : LiuBin
# @File : config.py
# @Description : 
# @Software: PyCharm

import yaml
import os

PROJECT_PATH = os.path.dirname(__file__)
CONF_PATH = os.path.join(PROJECT_PATH, 'conf')
YAML_CONFIG_PATH = os.path.join(CONF_PATH, 'config.yaml')


class ConfigManager(object):
    def __init__(self):
        self.PROJECT_PATH = PROJECT_PATH
        yaml_config_dict = yaml.load(open(YAML_CONFIG_PATH, encoding="utf-8"))
        data_path = yaml_config_dict.get('DATA_PATH')
        self.DATA_PATH = data_path if data_path.startswith('/') else os.path.join(PROJECT_PATH, data_path)

if __name__ == '__main__':
    conf = ConfigManager()
    print(conf.DATA_PATH)
