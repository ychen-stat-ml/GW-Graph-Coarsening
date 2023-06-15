import sys
import os
import datetime
import time
import pickle

# 避免在代码中 hardcode 参数信息，所有参数写在单独的一块；
# 每次实验运行保存至少如下信息：log信息（如运行时间、Loss结果等），
#   以及该次实验的configuration，主文件名称路径，
# 给每次实验取一个独特的id （时间+文件名）， 所有实验结果保存在该id的目录下。
# 提前准备好要保存的数据于save变量中

def get_time_output(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        duration = e_time - s_time
        
        print('This function takes {} sec(s).'.format(e_time - s_time))
        return res, duration
        
    return inner

def retrieve_name_ex(var):
    frame = sys._getframe(2)    
    while(frame):        
        for item in frame.f_locals.items():            
            if (var is item[1]):
                return item[0]
        frame = frame.f_back    
    return "" 
    
def outputVar(var):    
    print("{} = {}".format(retrieve_name_ex(var),var))

    
def generateLog(config, result=None, save=None):
    # config 包括实验用的超参数 + 选用的模型名称，
    # result 包括loss结果 + runtime
    # save 包括需要存储的，更细一点的数据结果，比如每一轮的loss
    if not os.path.exists("log"):
        os.makedirs("log")
    
    # path = os.path.split(getFilePath())
    path = sys._getframe(1).f_code.co_filename
    # path = "Fuck"
    print(path)
    rawname = os.path.split(path)[-1]
    filename = "{}_{}".format(rawname, str(datetime.datetime.now())
        ).replace(' ', '_').replace(':', '.')
    
    with open(path, "r", encoding="utf-8") as source: text = source.read()
    
    with open(os.path.join("log", filename + ".txt"), "w", encoding="utf-8") as output:
        output.write("Configuration:\n")
        for var in config:
            output.write("{} = {}\n".format(retrieve_name_ex(var),var))
        
        output.write("\n\n")
        output.write("Experiments Result:\n")
        if not result is None:
            for var in result:
                output.write("{} = {}\n".format(retrieve_name_ex(var),var))
            
        output.write("\n\n")
        output.write("Saved Data:\n")
        if not save is None:
            for var in save:
                output.write("{}\n".format(retrieve_name_ex(var)))
        
        output.write("\n\n")
        output.write("Source Code:\n")
        output.write("{}\n".format(text))
        
    # print(filename)
    if not save is None:
        with open(os.path.join("log", filename + ".data"), "wb") as f: 
            pickle.dump(save, f)
        
        print(filename)

# import io, tokenize, re
# from https://stackoverflow.com/a/62074206

import io, tokenize

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out

def save_to(obj, filename):
    with open(filename, "wb") as f: 
        pickle.dump(obj, f)

def read_from(filename):
    with open(filename, "rb") as f: 
        out = pickle.load(f)
    return out

import random
import numpy as np
import torch

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # a = 10
    # outputVar(a)
    
    # print(getFilePath())
    
    with open('test_call.py', 'r', encoding='UTF-8') as f:
        print(remove_comments_and_docstrings(f.read()))
