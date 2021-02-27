# @Time : 2021/2/26 15:51
# @Author : LiuBin
# @File : debug.py
# @Description : 
# @Software: PyCharm

import json
import os
from allennlp.commands.train import train_model_from_file
from config import PROJECT_PATH

import json
import shutil
import sys

from allennlp.commands import main

config_file = os.path.join(PROJECT_PATH,"conf/demo_classifier.jsonnet")

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "self_allennlp",
    "-o", overrides,
]

main()
#
# if __name__ == '__main__':
#     config_file = os.path.join(PROJECT_PATH, "conf/demo_classifier.jsonnet")
#     # Use overrides to train on CPU.
#     overrides = json.dumps({"trainer": {"cuda_device": -1}})
#     serialization_dir = "/tmp/debugger_train"
#     include_package = ["../self_allennlp"]
#     train_model_from_file(config_file, serialization_dir, overrides=overrides, include_package=include_package,
#                           force=True)
