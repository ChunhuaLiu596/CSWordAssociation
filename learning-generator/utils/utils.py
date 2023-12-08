import json
import os
import time
import argparse
import pathlib



def check_path(path):
    '''check whether a file path or a dir path existed or not'''
    d = os.path.dirname(path)
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
