# -*- coding:utf-8 -*-
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

def is_chinese(uchar):
    return '\u4e00' <= uchar <= '\u9fa5'

def is_float(string):
    try:
        float(string)
        return True
    except:
        return False