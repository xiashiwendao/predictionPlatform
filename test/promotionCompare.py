import pandas as pd
import os, sys

datasetpath = os.path.abspath(".\\dataset")
print(datasetpath)
df_uat = pd.read_csv(os.path.join(datasetpath, "BF_ONLINE_PROMOTION_UAT.csv"))
df_uat.head()
df_pro = pd.read_csv(os.path.join(datasetpath, "BF_ONLINE_PROMOTION_PRO_OVER.csv"))
df_pro.head()
# df_uat = df_uat[df_uat.ENTITY_TYPE=='Action']
len(df_uat)
# df_pro = df_pro[df_pro.ENTITY_TYPE=='Action']
len(df_pro)
df_uat = df_uat[['promotion_id','promotion','rule_id','rule_Name','action_Id','Crite_Name']]
df_uat.head()
df_pro = df_pro[['promotion_id','promotion','rule_id','rule_Name','action_Id','Crite_Name']]
df_pro.head()

df_merged = df_uat.merge(df_pro, on=['promotion', 'rule_Name', 'Crite_Name'], how='inner')
len(df_merged)
df_merged.to_csv(os.path.join(datasetpath, "BF_ONLINE_PROMOTION_MERGED.csv"))