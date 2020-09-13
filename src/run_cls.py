import args_parser, build_metrics, dataset
from transformers import Trainer
from pathlib import Path
import os
import csv
import logging
import numpy as np
import argparse
from transformers import BertConfig
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from transformers import InputExample, InputFeatures
import modeling
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="/home/data_normal/focus/can/BertForUse/bert",
                    help="load transformers pretrained model name")

parser.add_argument("--config_file", default="./config.yml",
                    help="train config file")

parser.add_argument(
    "--data_dir", default="../data/traindata", help="input data dir")

parser.add_argument("--task", default="tnews", help="task name")

parser.add_argument("--max_length", default=40, help="max sequlence length")

args = parser.parse_args()

def build_classify_trainer(
    tokenizer,
    model,
    processor,
    train_config_file,
    data_dir,
    max_seq_length,
    return_dadaset=False
):
    training_args = args_parser._get_training_args(train_config_file)
    trainset = (dataset.DataSet(processor, tokenizer,
                                data_dir, max_seq_length, "train"))
    evalset = (dataset.DataSet(processor, tokenizer,
                               data_dir, max_seq_length, "eval"))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=evalset,
        compute_metrics=build_metrics.build_compute_classify_metrics_fn(),
    )

    if return_dadaset:
        return trainer, training_args, trainset, evalset
    else:
        return trainer, training_args


def do_classify_train(trainer, pretrain_model_path, tokenizer, training_args):
    trainer.train(model_path=pretrain_model_path)
    trainer.save_model()
    tokenizer.save_vocabulary(training_args.output_dir)


def do_classify_eval(trainer, dataset, training_args):
    logging.info('*** Evaluate ***')
    trainer.compute_metrics = build_metrics.build_compute_classify_metrics_fn()
    eval_result = trainer.evaluate(eval_dataset=dataset)
    output_dir = Path(training_args.output_dir)
    assert output_dir.exists()
    output_eval_file = os.path.join(
        training_args.output_dir, "eval_result.txt")
    with open(output_eval_file, 'w') as writer:
        for key, value in eval_result.items():
            logging.info(" %s = %s", key, value)
            writer.write(" %s = %s\n" % (key, value))


def do_classify_predict(trainer, dataset, training_args):
    logging.info('*** Test ***')
    predictions, label_ids, metrics = trainer.predict(test_dataset=dataset)

    output_test_file = os.path.join(
        training_args.output_dir,
        "test_results.txt"
    )
    with open(output_test_file, 'w') as writer:
        logging.info('*** Test results is in {}***'.format(output_test_file))
        writer.write("index\tprediction\n")
        for index, item in enumerate(label_ids):
            item = dataset.get_labels()[item]
            writer.write('%d\t%s\n' % (index, item))

class TnewsProcessor(object):
    def _read_csv(cls, input_file, delimiter="\t", quotechar=None):
        print('>>> input file is {}'.format(input_file))
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter,
                                quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "cls_train.csv")), "train")

    def get_eval_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "cls_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "cls_eval.csv")), "test")

    def get_labels(self):
        label_list = 'news_finance,news_entertainment,news_car,news_game,news_sports,news_world,news_tech,news_culture,news_house,news_travel,news_agriculture,news_military,news_edu,news_story,stock'.split(',')
        return label_list

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = '-1'
            else:
                label = line[0]
                text_a = line[1]
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

processors = {"tnews": TnewsProcessor()}
processor = processors[args.task]
model, tokenizer = modeling.bert_cls_loader(args.model_name, num_labels = len(processor.get_labels()))
trainer, train_args, trainset, evalset = build_classify_trainer(
    tokenizer,
    model,
    processor,
    args.config_file,
    args.data_dir,
    args.max_length,
    return_dadaset=True
)
if train_args.do_train:
    trainer.train(model_path=args.model_name)
    trainer.save_model()

if train_args.do_eval:
    do_classify_eval(
        trainer,
        evalset,
        train_args
    )
if train_args.do_predict:
    do_classify_predict(
        trainer,
        evalset,
        train_args
    )