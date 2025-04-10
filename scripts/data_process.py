import pickle
from pathlib import Path

data_dir = Path("./datasets")


def process(input_data, output_data):
    with open(data_dir / 'data' / input_data, mode='rb') as f:
        data_list = pickle.load(f)

    f = open(data_dir / output_data, mode='w', encoding='utf-8')
    f.write("src,tgt\n")
    for data_item in data_list:
        src = data_item['src']
        tgt = data_item['tgt']

        assert len(src) == len(tgt), "src and tgt are not equal in length."
        assert "," not in src
        assert "," not in tgt
        f.write(f"{src},{tgt}\n")

    f.close()

    print("Generate file: ", str(output_data))


if __name__ == '__main__':
    process("test.sighan13.pkl", "sighan_2013_test.csv")
    process("test.sighan14.pkl", "sighan_2014_test.csv")
    process("test.sighan15.pkl", "sighan_2015_test.csv")
    process("trainall.times2.pkl", "train.csv")
