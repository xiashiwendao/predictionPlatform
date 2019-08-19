# -*- coding:utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import os, sys

srctype = "csv"
filepath = "dataset\\banner_group.csv"

basepath= os.path.abspath(filepath)
# filePath = os.path.join(basepath, "train_modified.csv")
#filePath = os.path.join(basepath, "banner_group.csv")
train = pd.read_csv(basepath)

print(srctype)
print(datasource)

filepath = os.path.abspath(datasource)
pd = pd.read_csv(filepath)
pd.head()

if srctype=="csv":
    filepath = os.path.abspath(datasource)
    pd = pd.read_csv(filepath)
    pd.head()
print("OK")

# if __name__ =='__main__':
#     parse = argparse.ArgumentParser()
#     parse.add_argument('--srctype')
#     parse.add_argument("--datasource")
#     args = vars(parse.parse_args())
#     srctype = args["srctype"]
#     datasource = args["datasource"]
    
print("OK")