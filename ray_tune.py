#!/usr/bin/env python
# coding: utf-8


seed = 42
num_cpus = 8
num_gpus = 1

import random
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import os
from ray import init, shutdown
from meli_functions import tune_xgboost
import time
import json
import pickle
from ray import tune
random.seed(42)


def optimize():
    datapath = "./DATA"

    train = pd.read_parquet(os.path.join(datapath, "train_active.parquet"))
    validation = pd.read_parquet(os.path.join(datapath,"validation_active.parquet"))
    test = pd.read_parquet(os.path.join(datapath,  "test.parquet"))
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
    pickle.dump(le, open("./labelencoder_item_domain_id", "wb"))


    maxval = train.groupby("sku", dropna=True)["target_stock"].max()
    qq = np.quantile(maxval[maxval.notna()], 0.25)
    #sns.histplot(maxval, bins = 200)


    samples = np.random.choice(train.sku.unique(), round(0.2 * len(train.sku.unique())))
    index_train = train.sku.isin(samples)
    index_val = validation.sku.isin(samples)
    x_train = train[index_train]
    x_val = validation[index_val]


    x_train.loc[:, "target_stock"] = x_train["target_stock"].astype("float32")
    x_val.loc[:,  "target_stock"] = x_val["target_stock"].astype("float32")
    x_train.loc[:, "item_domain_id"] = x_train["item_domain_id"].astype("int32")
    x_val.loc[:, "item_domain_id"] = x_val["item_domain_id"].astype("int32")

    y_train = x_train.id
    x_train = x_train.drop(["id"], axis = 1)
    y_train = y_train-1

    y_val = x_val.id
    x_val = x_val.drop(["id"], axis = 1)
    y_val = y_val-1

    del train
    del validation
    gc.collect()


    search_space = {'learning_rate': tune.uniform(1e-5, 3e-1), 
                    "min_child_weight" : tune.randint(100,5000),
                    'reg_alpha' : tune.uniform(0, 5),
                    'reg_lambda' : tune.uniform(0, 5),
                    'max_depth': tune.randint(2, 7),
                    "subsample": tune.uniform(0.4, 1),
                    "colsample_bytree": tune.uniform(0.4, 1),
                    "colsample_bylevel" : tune.uniform(0.4, 1),
                    "null_weight" : tune.uniform(0, 1),
                    "outlier_weight": tune.uniform(0, 1),
                    "delta_inf": tune.uniform(0,6),
                    "qq" : qq,
                    "maxval" : maxval}
                    
    shutdown()
    init(num_cpus=8, num_gpus=1)                      
    start = time.time()
    analysis = tune_xgboost(seed = seed, data = {"x_train": x_train, "x_val":x_val, 
                            "y_train": y_train, "y_val": y_val}, 
                            search_space = search_space, name = "xgboost_experiment")
    print('It takes %s minutes' % ((time.time() - start)/60))

    df = analysis.results_df
    df.to_csv("./DATA/results.csv")

    with open('./DATA/best_config.json', 'w') as f:
        json.dump(analysis.best_config, f)
