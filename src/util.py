import os
import csv
from pathlib import Path

def util_tnews(path='../data/traindata'):
    rawpath=Path('../data/tnews')
    for files in rawpath.rglob('*'):
        data = [x.strip().split('_!_') for x in open(files).readlines()]
        label_list = []

        if 'train' in files.name:
            writename='cls_train.csv'
        elif 'test' in files.name:
            writename='cls_test.csv'
        else:
            writename='cls_dev.csv'
        writedir = os.path.join(path, writename)
        w = open(writedir, 'w', encoding="utf-8", newline="")
        writer = csv.writer(w, delimiter="\t")

        for row in data:
            label_list.append(row[2])
            writer.writerow(row[2:4])

        if 'train' in files.name:
            label_list_ = list(set(label_list))
            label_list_.sort(key=label_list.index)
            open(os.path.join(path, 'cls_label_list.txt'),
                'w').write('\n'.join(label_list_))
  
