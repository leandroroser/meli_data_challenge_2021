#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import os
from ray import init, shutdown
from meli_functions import run
import time
import json
import random
import pickle
import xgboost as xgb

seed = 42
random.seed(42)

def final_run():
    datapath = "./DATA"


    train = pd.read_parquet(os.path.join(datapath, "train_active.parquet"))
    validation = pd.read_parquet(os.path.join(datapath,"validation_active.parquet"))
    test = pd.read_parquet(os.path.join(datapath,  "test.parquet"))
    maxval = pd.read_parquet(os.path.join(datapath,  "maxval.parquet"))
    test.sku = test.sku.astype("int64")

    train_predictor = "rolling_cumsum_1"
    train.rename({train_predictor:"target_stock"}, axis = 1, inplace = True)
    validation.rename({train_predictor:"target_stock"}, axis = 1, inplace = True)

    train.loc[train["target_stock"] != -1 , "target_stock"] = np.log(train.loc[train["target_stock"] != -1, "target_stock"] + 0.001)
    validation.loc[validation["target_stock"] != -1 ,"target_stock"] = np.log(validation.loc[validation["target_stock"] != -1, "target_stock"]  + 0.001)

    train["target_stock"].replace(-1, np.NaN, inplace = True)
    validation["target_stock"].replace(-1, np.NaN, inplace = True)
    train.loc[train.duplicated(["sku", "target_stock"]), "target_stock"] = np.NaN
    validation.loc[validation.duplicated(["sku", "target_stock"]), "target_stock"] = np.NaN

    train = train[["sku", "id", "target_stock", 'item_domain_id']]
    validation = validation[["sku", "id", "target_stock", 'item_domain_id']]

    train = train[train.sku.isin(test.sku)]
    validation = validation[validation.sku.isin(test.sku)]

    train.sku = train.sku.astype("int32")
    validation.sku = validation.sku.astype("int32")
    train.id = train.id.astype("int8")
    validation.id = validation.id.astype("int8")

    gc.collect()


    le = LabelEncoder()
    le.fit(pd.concat([train["item_domain_id"],  validation["item_domain_id"], 
                    test["item_domain_id"]]).unique())
    train["item_domain_id"] = le.transform(train["item_domain_id"])
    validation["item_domain_id"] = le.transform(validation["item_domain_id"])
    test["item_domain_id"] = le.transform(test["item_domain_id"])
    pickle.dump(le, open("./DATA/labelencoder_item_domain_id", "wb"))
    

    train.loc[:, "target_stock"] = train["target_stock"].astype("float32")
    validation.loc[:,  "target_stock"] = validation["target_stock"].astype("float32")
    train.loc[:, "item_domain_id"] = train["item_domain_id"].astype("int32")
    validation.loc[:, "item_domain_id"] = validation["item_domain_id"].astype("int32")
    test.loc[:, "target_stock"] = test["target_stock"].astype("float32")
    test.loc[:, "item_domain_id"] = test["item_domain_id"].astype("int32")

    y_train = train.id
    x_train = train.drop(["id"], axis = 1)
    y_train = y_train-1

    y_val = validation.id
    x_val = validation.drop(["id"], axis = 1)
    y_val = y_val-1


    with open('./DATA/best_config.json', 'r') as f:
        best_config = json.load(f)


    model = run(best_config, x_train, x_val, y_val, y_val, maxval, seed)

    pickle.dump(model, open('./DATA/xgbmodel.pkl', 'wb'))
    
    test = xgb.DMatrix(test[x_train.columns])
    print("predicting")
    out = model.predict(test)
    out = pd.DataFrame(out)
    out.to_csv("./DATA/out.csv", header = False, index = False, sep = ",", float_format='%.4f')
