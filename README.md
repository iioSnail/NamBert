# NamBert

Source code for the paper "Unveiling the Impact of Multimodal Features on Chinese Spelling Correction: From Analysis to Design".

## Environment

- Python >= 3.8
- pytorch >= 2.0
- pytorch lightning >= 2.0

```shell
conda create -n NamBert 
conda activate NamBert
git clone https://github.com/iioSnail/NamBert.git
cd NamBert
pip install -r requirements.txt
```

## Data

### Raw data

- SIGHAN Bake-off 2013: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html
- SIGHAN Bake-off 2014: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html
- SIGHAN Bake-off 2015: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html
- Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

You can download the cleaned data published by [ReaLiSe](https://drive.google.com/drive/folders/1dC09i57lobL91lEbpebDuUBS0fGz-LAk) and put them in the `datasets` directory.

The directory will be like:

```
datasets
└── data
    ├── test.sighan13.lbl.tsv
    ├── test.sighan13.pkl
    ├── test.sighan14.lbl.tsv
    ├── test.sighan14.pkl
    ├── test.sighan15.lbl.tsv
    ├── test.sighan15.pkl
    └── trainall.times2.pkl
```

Process data to fit this project:

```shell
python scripts/data_process.py
```

## Finetune

Recommend to directly download our pretrained model ([Google Drive](https://drive.google.com/file/d/1qnpus1ahcrj5DiBawLLjq9tfdMij3vbA/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1nQsk6cUogjDEA5RMabdjwQ?pwd=dqkx)) to finetune. 

Please put the pretrained model into the `ckpt` directory.

Run the command to finetune the model:

```
sh finetune.sh
```


## Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iioSnail/NamBert/blob/master/example.ipynb)

You can use our final model by [Hugging face](https://huggingface.co/iioSnail/NamBert-for-csc). For example:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("iioSnail/NamBert-for-csc", trust_remote_code=True)
model = AutoModel.from_pretrained("iioSnail/NamBert-for-csc", trust_remote_code=True)

inputs = tokenizer("我喜换吃平果，逆呢？", return_tensors='pt')
logits = model(**inputs).logits

target_ids = logits.argmax(-1)
target_ids = tokenizer.restore_ids(target_ids, inputs['input_ids'])

print(''.join(tokenizer.convert_ids_to_tokens(target_ids[0, 1:-1])))
```

If you would just like to use our model to predict, we recommend you use the predict method. For example:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("iioSnail/NamBert-for-csc", trust_remote_code=True)
model = AutoModel.from_pretrained("iioSnail/NamBert-for-csc", trust_remote_code=True)

model = model.to(device)
model = model.eval()
model.set_tokenizer(tokenizer)

model.predict("我是炼习时长两念半的个人练习生菜徐坤")
model.predict(["我是炼习时长两念半的个人练习生菜徐坤", "喜欢场跳rap篮球！！"])
```
