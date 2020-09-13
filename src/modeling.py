from transformers import BertForSequenceClassification, BertTokenizer, BertConfig


def bert_cls_loader(model_name, num_labels, tokenizer_name=None, config_name=None):
    tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    config_name = config_name if config_name is not None else model_name
    config = BertConfig.from_pretrained(config_name)
    config.num_labels = num_labels
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    return model, tokenizer