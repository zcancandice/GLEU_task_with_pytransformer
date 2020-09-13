from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
from transformers import InputExample, InputFeatures
from typing import List, Optional, Union
import logging
logging.getLogger().setLevel(logging.INFO)


def convert_examples_to_features(examples, tokenizer, max_length, label_list):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    logging.info('>>> {} examples convert to features'.format(len(examples)))
    for ex_index, example in enumerate(examples):

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        input_ids, token_type_ids, attention_mask = inputs[
            "input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [0]*padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        assert len(input_ids) == max_length
        label = label_map[example.label]
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      label=label,
                                      ))
        if ex_index < 5:
            logging.info(">>> writing example %d" % (ex_index))
            logging.info('>>> text is {} '.format(example.text_a))
            logging.info('>>> input_ids is {}'.format(input_ids))
            logging.info('>>> label text is {} and label_ids is {}'.format(
                example.label, label))
    return features


class DataSet(Dataset):
    def __init__(self, processor, tokenizer, data_dir, max_length, mode):
        self.processor = processor
        self.label_list = processor.get_labels()
        if mode == "eval":
            examples = processor.get_eval_examples(data_dir)
        elif mode == "predict":
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)
        self.features = convert_examples_to_features(
            examples, tokenizer=tokenizer, max_length=max_length, label_list=self.label_list
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list