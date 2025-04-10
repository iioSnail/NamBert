import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.common import BERT
from utils.dataset import CSCDataset
from utils.log_utils import log


class DefaultCollateFn(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        src, tgt = zip(*batch)
        src, tgt = list(src), list(tgt)

        src = BERT.get_bert_inputs(src, tokenizer=self.tokenizer, max_length=self.args.max_length)
        tgt = BERT.get_bert_inputs(tgt, tokenizer=self.tokenizer, max_length=self.args.max_length)

        return src, \
               tgt, \
               (src['input_ids'] != tgt['input_ids']).float(), \
               {}


def create_dataloader(args, collate_fn=None, tokenizer=None):
    dataset = None

    limit_size = args.limit_batches * args.batch_size

    if args.data is not None:
        dataset = CSCDataset(args.data, limit_size=limit_size)
    elif args.datas is not None:
        dataset = ConcatDataset([CSCDataset(data) for data in args.datas.split(",")])
    else:
        log.exception("Please specify data or datas.")
        exit()

    train_dataset = dataset
    if args.val_data is None:
        valid_size = int(len(dataset) * args.valid_ratio)
        train_size = len(dataset) - valid_size

        if valid_size > 0:
            train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        else:
            log.warning("No any valid data.")
            train_dataset = dataset
            valid_dataset = None
    else:
        valid_dataset = CSCDataset(args.val_data)

    if collate_fn is None:
        collate_fn = DefaultCollateFn(args, tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.workers)

    if valid_dataset is None:
        return train_loader, None

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=False,
                              num_workers=args.workers)

    return train_loader, valid_loader


def create_test_dataloader(args, data=None):
    if data is None:
        data = args.data
    dataset = CSCDataset(data)

    workers = 0
    if hasattr(args, "workers"):
        workers = args.workers

    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=workers)
