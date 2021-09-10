#!/usr/bin/env python
# coding: utf-8


from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import udf,pandas_udf
from pyspark.sql import Window
import pyspark.sql.types as T
import pandas as pd
import numpy as np
import xgboost as xgb
import random
import gc
import os
import pickle
import warnings
from sklearn.preprocessing import OneHotEncoder
from ray import tune, init, shutdown
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from datetime import datetime, timedelta

def create_window(partitionby, orderby = None, rangebetween = None):
    """Create a window for Pyspark
    """

    out = f"Window.partitionBy('{partitionby}')"
    if orderby is not None:
        out = out + f".orderBy('{orderby}')"
    if rangebetween is not None:
        out = out + f".rangeBetween({rangebetween[0]}, {rangebetween[1]})"
    return eval(out)
    
    
def plot_frames_train_val(train, validation, frames, main, x = "id", y1 = "cumsum",
 y2 = "cumsum", ax1_lab = "train", ax2_lab = "Validation", rows = 7, cols = 20):
    
    """ Plot train-validation data
    """

    fig,axs =  plt.subplots(rows, cols, figsize = (20,10))
    k = 0
    for i in range(rows):
        for j in range(cols):
            try:
                train_data  = train[train.sku.isin([frames[k]])]
                val_data  = validation[validation.sku.isin([frames[k]])]
                l1 = axs[i][j].scatter(train_data[x], train_data[y1], c = "r", label = ax1_lab, alpha = 0.5)
                l2 = axs[i][j].scatter(val_data[x], val_data[y2], c = "b", label = ax2_lab, alpha = 0.5)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                axs[i][j].title.set_text(frames[k])
            except:
                pass
            k += 1
    fig.legend([l1, l2], labels = [ax1_lab.title(), ax2_lab.title()])
    plt.subplots_adjust(right=0.9)
    fig.suptitle(main.title())
    plt.show()
     
        
@udf(T.IntegerType())

def count_zeros(x):
    """
    Count # leading zeros in a Python list
    """
    counter = 0
    for i in x:
        if i == 0.0:
            counter += 1
        else:
            break
    return counter
    
    
@udf(T.IntegerType()) 
def are_consecutive_dates(x):
    """Check if dates are consecutive in time
    """
    x = sorted([datetime.strptime(i, "%Y-%m-%d") for i in x])
    res = True
    for idx in range(1, len(x)):
        if (x[idx] - x[idx - 1]).days != 1:
            res = False
            break
    return res


def forward_fill(window, data, column):
    """ Forward fill a spark Column
    """
    return data.withColumn(column, F.last(column, True).over(window)) 
    
def write_data(data, name):
    """ write pyspark DataFrame in coalesced format
    """
    data.coalesce(1).write.format("parquet").mode("overwrite").save(name)
    
    
def proportion_transform(x, var, drop_first = False):
    """Split a categorical variable into the proportions of each category for the
       given SKU across all the dates in the training set
    """
    out = x.groupBy(["sku", var]).count()
    out = out.groupBy("sku").pivot(var).sum("count").na.fill(0)
    distinct_values = [row[0] for row in x.select(var).distinct().collect()]
    for i in distinct_values:
        denominator = set(distinct_values) - set(i)
        denominator = [f"F.col('{j}')" for j in denominator]
        denominator = "+".join(denominator) 
        expression = f"out.withColumn('{i}_prop', F.col('{i}')/({denominator}))" 
        out = eval(expression)
    out = out.drop(*distinct_values)
    if drop_first:
        out = out.drop(f"{distinct_values[0]}_prop")
    return out.na.fill(0)


def create_proportion_columns(x):
    """Split multiple categorical variables into the proportions of each category for the
       given SKU across all the dates in the training set
    """
    listing_type = proportion_transform(x, "listing_type")
    shipping_payment = proportion_transform(x, "shipping_payment")
    shipping_logistic_type = proportion_transform(x, "shipping_logistic_type")
    minutes_active = x.groupBy("sku").avg("minutes_active").withColumnRenamed("avg(minutes_active)", "minutes_active_avg")
    selling_rate = x.groupBy("sku").avg("selling_rate").withColumnRenamed("avg(selling_rate)", "selling_rate_avg")
    price = x.groupBy("sku").avg("current_price").withColumnRenamed("avg(current_price)", "price_avg")
    features = (listing_type.join(shipping_payment, "sku")
            .join(shipping_logistic_type, "sku")
            .join(minutes_active, "sku")
            .join(selling_rate, "sku")
            .join(price, "sku")) 
    return features


def get_dof(X, X_active):
    """ Get counts for each day of the week for a given SKU
    """
    X = X.withColumn("dayofweek", F.dayofweek("date"))
    X_active = X_active.withColumn("dayofweek", F.dayofweek("date"))
    dof_X = X.groupBy("sku").pivot("dayofweek").sum("sold_quantity").na.fill(0)
    dof_X_active = X.groupBy("sku").pivot("dayofweek").sum("sold_quantity").na.fill(0)
    dof_X = dof_X.withColumn('sum',sum([F.col(c) for c in dof_X.columns]))
    dof_X_active = dof_X_active.withColumn('sum',sum([F.col(c) for c in dof_X_active.columns]))
    dof_X = dof_X.select(F.col("sku"), *[F.col(x)/F.col("sum") for x in dof_X.columns[1:-1]]).drop("sum")
    dof_X_active = dof_X_active.select(F.col("sku"), *[F.col(x)/F.col("sum") for x in dof_X_active.columns[1:-1]]).drop("sum")
    for i,j in zip(range(7), dof_X.columns[1:]):
        dof_X = dof_X.withColumnRenamed(j, f"day_{i}")
        dof_X_active = dof_X.withColumnRenamed(j, f"day_{i}")
    return dof_X,dof_X_active


def get_days_active(X):
    """Get the number of days active for a given SKU
    """
    days_X_active = X.groupBy("sku").agg({"is_active":"sum"})
    total_days = X.groupBy("sku").agg({"sku":"count"})
    days_X_active = days_X_active.join(total_days, "sku")
    days_X_active = days_X_active.withColumn("proportion_active", F.col("sum(is_active)")/F.col("count(sku)"))
    return days_X_active.select("sku", "proportion_active")


def crps_loss(y_pred, dtrain, is_higher_better=False):
    """ Compute crps loss for XGboost. Not implemented in the code.
    """
    labels = dtrain.get_label()
    y_true = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1)).cumsum(axis=1)
    y_pred = y_pred.reshape(-1, 30)
    y_pred = np.clip(y_pred.cumsum(axis=1), 0, 1)
    out =  np.power(y_true - y_pred, 2)
    out = float(np.sqrt(np.sum(out)/(len(labels) - 1)))
    del labels
    del y_true
    del y_pred
    gc.collect()
    return 'crps_loss', out


def crps_obj(y_pred, dtrain):
    """ Compute crps objective function for XGboost. Not implemented in the code.
    """
    labels = dtrain.get_label()
    y_pred =  np.asarray(y_pred)
    y_true = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1)).cumsum(axis=1)
    y_pred = y_pred.reshape(-1, 30)
    y_pred = np.clip(y_pred.cumsum(axis=1), 0, 1)
    grad = 2 * (y_pred - y_true)
    hess = 2.0 + 0.0 * grad
    del labels
    del y_true
    del y_pred
    gc.collect()
    return grad.flatten(), hess.flatten()


def fitness(config, checkpoint_dir=None, data = None, seed = None):
    """ Function to optimize by Tune
    """
    x_train = data["x_train"]
    x_val = data["x_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    maxval = data["maxval"]
    bounds  = config["qq"] - config["delta_inf"]
    cuts = pd.DataFrame(maxval.reset_index().values, columns = ["sku", "cut"]).reset_index(drop = True)
    cuts['cut'] = cuts['cut'].astype("float")
    x_train = x_train.merge(cuts, on = "sku", how = "left")
    train_weights = x_train["cut"] > bounds
    train_weights = train_weights.astype(int)
    x_train = x_train.drop("cut", axis = 1)
    train_weights[train_weights == 0] = config['outlier_weight']
    train_weights[x_train.target_stock.isna()] = config['null_weight']  

    x_val = x_val.merge(cuts, on = "sku",how = "left")
    validation_weights = x_val["cut"] > bounds
    validation_weights = validation_weights.astype(int)
    x_val = x_val.drop("cut", axis = 1)
    validation_weights[validation_weights == 0] = config['outlier_weight']
    validation_weights[x_val.target_stock.isna()] = config['null_weight']  

    dtrain= xgb.DMatrix(x_train, y_train, weight = train_weights)
    dvalid = xgb.DMatrix(x_val, y_val, weight = validation_weights)
        
    params  = dict(learning_rate = config["learning_rate"],
                  min_child_weight = config["min_child_weight"],
                  reg_alpha = config["reg_alpha"],
                  reg_lambda = config["reg_lambda"],
                  subsample = min(config["subsample"], 1),
                  colsample_bytree = min(config["colsample_bytree"], 1),
                  colsample_bylevel = min(config["colsample_bylevel"], 1),
                  tree_method = "gpu_hist",
                  eval_metric = "mlogloss", 
                  objective = "multi:softmax",
                  booster = "gbtree",
                  num_class = 30,
                  disable_default_eval_metric = 0,
                  seed = seed)

    model =   xgb.train(params, dtrain=dtrain,
                    num_boost_round=200,
                    evals=[(dvalid, 'dvalid')],
                    callbacks=[TuneReportCheckpointCallback(filename="model_ray.xgb")],
                    verbose_eval= False)
                
    del dtrain  
    del dvalid     
    del model
    gc.collect()



def tune_xgboost(seed, data, name, search_space, resume = False, num_samples = 20, cpus = 4, gpus = 0.5):
    """ Tune process to optimize hyperparameters
    """

    scheduler = AsyncHyperBandScheduler(
        max_t=200,  
        grace_period=1  ,
        reduction_factor=2)

    search_alg = HyperOptSearch(random_state_seed = seed)

    analysis = tune.run(
        tune.with_parameters(fitness, data = data, seed = seed),
        name = name,
        metric="dvalid-mlogloss",
        mode="min",
        search_alg=search_alg,
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        resume = resume)
    
    search_alg.save("./DATA/hyperopt-checkpoint.pkl")

    return analysis





def run(config, x_train, x_val, y_train, y_val, maxval, seed):
    """ Run hyperparameter optimization for the best configuration
    """

    bounds  = config["qq"] - config["delta_inf"]
    cuts = pd.DataFrame(maxval.reset_index().values, columns = ["sku", "cut"]).reset_index(drop = True)
    cuts['cut'] = cuts['cut'].astype("float")
    x_train = x_train.merge(cuts, on = "sku", how = "left")
    train_weights = x_train["cut"] > bounds
    train_weights = train_weights.astype(int)
    x_train = x_train.drop("cut", axis = 1)
    train_weights[train_weights == 0] = config['outlier_weight']
    train_weights[x_train.target_stock.isna()] = config['null_weight']  

    x_val = x_val.merge(cuts, on = "sku",how = "left")
    validation_weights = x_val["cut"] > bounds
    validation_weights = validation_weights.astype(int)
    x_val = x_val.drop("cut", axis = 1)
    validation_weights[validation_weights == 0] = config['outlier_weight']
    validation_weights[x_val.target_stock.isna()] = config['null_weight']  

    dtrain= xgb.DMatrix(x_train, y_train, weight = train_weights)
    dvalid = xgb.DMatrix(x_val, y_val, weight = validation_weights)
        
    params  = dict(learning_rate = config["learning_rate"],
                  min_child_weight = config["min_child_weight"],
                  reg_alpha = config["reg_alpha"],
                  reg_lambda = config["reg_lambda"],
                  subsample = min(config["subsample"], 1),
                  colsample_bytree = min(config["colsample_bytree"], 1),
                  colsample_bylevel = min(config["colsample_bylevel"], 1),
                  tree_method = "hist",
                  eval_metric = "mlogloss", 
                  booster = "gbtree",
                  num_class = 30,
                  objective = "multi:softprob",
                  disable_default_eval_metric = 1,
                  seed = seed)

    early_stop = xgb.callback.EarlyStopping(rounds=30,
                                metric_name='mlogloss',
                                data_name='dvalid',
                                save_best=True)

    model =   xgb.train(params, dtrain=dtrain,
                    num_boost_round=300,
                    evals=[(dvalid, 'dvalid')],
                    callbacks=[early_stop],
                    verbose_eval= True)
                
    return model