import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, NewType, Any, Iterable, List, Union
import yaml
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

class YamlArgumentParser(HfArgumentParser):
    def parse_yaml_file(self, yaml_file: str):
        data = yaml.load(open(yaml_file, 'r').read(), Loader=yaml.FullLoader)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)


def _get_training_args(config_file):
    parser = YamlArgumentParser(TrainingArguments)

    if config_file.endswith(".json"):
        training_args = parser.parse_json_file(
            json_file=os.path.abspath(config_file))[0]
    elif config_file.endswith(".yml"):
        training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(config_file))[0]
    else:
        raise ValueError
    return training_args