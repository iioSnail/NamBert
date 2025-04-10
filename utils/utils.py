import os
import random
from pathlib import Path

import numpy as np
import pypinyin
import torch
from PIL import ImageFont
from lightning_fabric import seed_everything

from utils.str_utils import is_chinese

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def setup_seed(seed):
    if seed < 0:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def render_color_for_text(text, indices, color='red', format='console'):
    color_indices = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33'
    }

    text = text.replace(" ", "")
    char_list = list(text)
    for i in range(len(indices)):
        if indices[i]:
            if format in ['console', "sh", "shell"]:
                char_list[i] = "\033[" + color_indices.get(color, '30') + "m" + text[i] + "\033[0m"
            elif format in ['markdown', "md"]:
                char_list[i] = ":%s[%s]" % (color, char_list[i])

    return ''.join(char_list)


def convert_ids_to_tokens(tokenizer, ids):
    shape = ids.shape
    ids = ids.reshape(-1)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return np.array(tokens).reshape(shape).tolist()


font = None

def convert_char_to_image(character, font_size=32):
    global font
    if font is None:
        font = ImageFont.truetype(str(ROOT / "assets" / "font" / "ms_yahei.ttf"), size=font_size)

    image = font.getmask(character)
    image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])

    image = image[:font_size, :font_size]

    if image.size != (font_size, font_size):
        back_image = np.zeros((font_size, font_size)).astype(np.float32)
        offset0 = (font_size - image.shape[0]) // 2
        offset1 = (font_size - image.shape[1]) // 2
        back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
        image = back_image

    return torch.tensor(image)


def convert_char_to_pinyin(character, size=-1, tone=False):
    if not is_chinese(character):
        return torch.LongTensor([0] * max(size, 1))

    if tone:
        pinyin = pypinyin.pinyin(character, style=pypinyin.TONE3)[0][0]
    else:
        pinyin = pypinyin.pinyin(character, style=pypinyin.NORMAL)[0][0]

    if not tone:
        embeddings = torch.tensor([ord(letter) - 96 for letter in pinyin])
    else:
        embeddings = []
        for letter in pinyin:
            if letter.isnumeric():
                embeddings.append(int(letter) + 27)
            else:
                embeddings.append(ord(letter) - 96)
        embeddings = torch.tensor(embeddings)

    if size > len(embeddings):
        padding = torch.zeros(size - len(embeddings))
        embeddings = torch.concat([embeddings, padding])

    return embeddings


def pred_token_process(src_tokens, pred_tokens, ignore_token: list = None):
    if len(src_tokens) != len(pred_tokens):
        print("[Error]unequal length:", ''.join(src_tokens))
        return pred_tokens

    for i in range(len(src_tokens)):
        if len(src_tokens[i]) != len(pred_tokens[i]):
            # print("[Warning]unequal token length: %s, token: (%s, %s)"
            #       % (''.join(src_tokens), src_tokens[i], pred_tokens[i]))
            pred_tokens[i] = src_tokens[i]
            continue

        if not is_chinese(src_tokens[i]):
            pred_tokens[i] = src_tokens[i]
            continue

        if ignore_token:
            if src_tokens[i] in ignore_token:
                pred_tokens[i] = src_tokens[i]
                continue

    return pred_tokens
