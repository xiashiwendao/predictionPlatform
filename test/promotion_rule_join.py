import pandas as pd
import numpy as np
import os, sys

basepath="C:\\Users\\wenyang.zhang\\Documents\\MySpace\\Workspace\\FullChannel\\GetDataForIBeauty\\Data"
data_uat = pd.read_csv(os.path.join(basepath, "PROMOTION_RULE_UAT_v3.csv"), encoding="utf_8_sig")
# data_uat.head(3)
data_pro = pd.read_csv(os.path.join(basepath, "PROMOTION_RULE_PRO_v3.csv"), encoding="utf_8_sig")
data_pro.head(300)
data_merge = data_uat.merge(data_pro, on=["promotion","rule"], how="outer")
data_merge.head(100)
data_merge.to_csv(os.path.join(basepath, "PROMOTION_RULE_MERGE_v3.csv"), encoding="utf_8_sig")