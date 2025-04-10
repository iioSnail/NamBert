from torch.utils.data import Dataset
from pathlib import Path

from utils.log_utils import log

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CSCDataset(Dataset):

    def __init__(self, data_file: str, filepath=None, **kwargs):
        super(CSCDataset, self).__init__()
        self.data_name = data_file.replace(".csv", "")
        if filepath is not None:
            filepath = filepath
        else:
            filepath = self.get_filepath(data_file)

        self.data, self.error_data = self.load_data_from_csv(filepath)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data_from_csv(self, filepath):
        log.info("Load dataset from %s", filepath)
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        error_data = []
        for line in lines[1:]:
            items = line.split(",")
            src = items[0].strip()
            tgt = items[1].strip()

            src = ''.join(src.replace(" ", "").replace(u"\u3000", ""))
            tgt = ''.join(tgt.replace(" ", "").replace(u"\u3000", ""))

            if len(src) == len(tgt):
                data.append((src, tgt))
            else:
                error_data.append((src, tgt))

        log.info("Load completed. Success num: %d, Failure num: %d.", len(data), len(error_data))

        return data, error_data

    def get_filepath(self, data_name):
        return ROOT / 'datasets' / data_name