#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from meli_functions import (create_window,
                           create_proportion_columns,
                           forward_fill,write_data,
                           are_consecutive_dates,
                           get_days_active)
import numpy as np


def processing():
    spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "15g").appName('meli-app')     .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


    data_path = "./DATA"

    train = spark.read.parquet(f"{data_path}/train_data.parquet")
    meta = spark.read.json(f"{data_path}/items_static_metadata_full.jl")
    test = spark.read.csv(f"{data_path}/test_data.csv", header = True)
    train0 = train # keep track of original data

    meta = meta.withColumn("item_domain_id", F.regexp_replace(F.col("item_domain_id"), "^.*-", ""))
    meta.filter(F.col("sku") == 454273).show() # some with nulls
    meta = meta.na.fill("")

    # Add metadata, items sold/domain id a
    test = test.withColumn("monid", F.monotonically_increasing_id())
    test = test.join(meta[["sku", "item_domain_id", "site_id"]], "sku", how = "left")
    test = test.orderBy("monid")

    # Check if dates are consecutive for each SKU
    not_consecutives = train.groupBy("sku").agg(are_consecutive_dates(F.collect_list(F.col("date"))).alias("consecutive"))
    not_consecutives.filter(F.col("consecutive") == False).show()


    # check if ts start at different dates
    mindate = train.groupBy("sku").agg(F.min("date"))
    mindate.select("min(date)").distinct().show()

    # check if ts start at different dates
    maxdate = train.groupBy("sku").agg(F.max("date"))
    maxdate.select("max(date)").distinct().show()


    train = train.withColumn('selling_rate', F.when(F.col("minutes_active") > 0, F.col("sold_quantity")/F.col("minutes_active")).otherwise(F.lit(0)))
    train = train.withColumn('is_active', F.when(F.col("minutes_active") > 0, F.lit(1)).otherwise(F.lit(0)))
    train_active = train.filter(F.col('is_active') == 1)

    window = create_window('sku', 'date')
    train = train.withColumn('id',  F.row_number().over(window))
    train = train.withColumn("is_train", F.when(F.col("id") <= 30, F.lit(1)).otherwise(F.lit(0)))
    train = train.withColumn("sku_split", F.concat(F.col("sku"), F.lit("_"), F.col("is_train").cast(T.StringType()))).drop("id")

    train_active = train_active.withColumn('id',  F.row_number().over(window))
    train_active = train_active.withColumn("is_train", F.when(F.col("id") <= 30, F.lit(1)).otherwise(F.lit(0)))
    train_active = train_active.withColumn("sku_split", F.concat(F.col("sku"), F.lit("_"), F.col("is_train").cast(T.StringType()))).drop("id")
    # Compute cumsum for items sold quantities


    window2 = create_window('sku_split', 'date')
    windowval = create_window("sku_split", 'date', [Window.unboundedPreceding, 0])

    train = train.withColumn('cumsum', F.sum('sold_quantity').over(windowval))
    train_active = train_active.withColumn('cumsum', F.sum('sold_quantity').over(windowval))

    train = train.withColumn("max", F.max("cumsum").over(windowval) + 1)
    train_active = train_active.withColumn("max", F.max("cumsum").over(windowval) + 1)
    train = train.withColumn('cumsum_pc', F.col("cumsum") / F.col("max"))
    train_active = train_active.withColumn('cumsum_pc', F.col("cumsum") / F.col("max"))

    train = train.withColumn('id',  F.row_number().over(window2))
    train_active = train_active.withColumn('id',  F.row_number().over(window2))


    validation = train.filter(F.col("is_train") == 0)
    train = train.filter(F.col("is_train") == 1)
    validation_active = train_active.filter(F.col("is_train") == 0)
    train_active = train_active.filter(F.col("is_train") == 1)

    days_activity_train = get_days_active(train)
    days_activity_validation = get_days_active(validation)

    dof_train,dof_train_active = get_dof(train,train_active)
    dof_validation,dof_validation_active = get_dof(validation,validation_active)


    rolling_windows = [1, 2, 3, 4, 5]

    for cumtype in ["cumsum", "cumsum_pc"]:
        for interval in rolling_windows:
            train = train.withColumn(f"rolling_{cumtype}_{interval}", F.avg(cumtype).over(create_window('sku', 'id', [-interval, 0])))
            train_active = train_active.withColumn(f"rolling_{cumtype}_{interval}", F.avg(cumtype).over(create_window('sku', 'id', [-interval,0])))
            validation = validation.withColumn(f"rolling_{cumtype}_{interval}", F.avg(cumtype).over(create_window('sku', 'id', [-interval, 0])))
            validation_active = validation_active.withColumn(f"rolling_{cumtype}_{interval}", F.avg(cumtype).over(create_window('sku', 'id', [-interval, 0])))


    skus = train.select("sku").distinct().rdd.flatMap(lambda x: x).collect()
    skus = np.random.choice(skus, 200).tolist()


    q0 = train.filter(train.sku.isin(skus)).toPandas()
    q1 = validation.filter(train.sku.isin(skus)).toPandas()

    frames = q0.sku.unique()
    plot_frames_train_val(train = q0, validation = q1, frames = frames, main = "Train vs validation")
    plot_frames_train_val(train = q0[q0.minutes_active > 0], validation = q1[q1.minutes_active > 0], frames = frames, main = "Train vs validation for both minutes active > 0")
    plot_frames_train_val(train = q0, validation = q0[q0.minutes_active > 0], frames = frames, ax1_lab = "train", ax2_lab= "Train minutes active > 0", 
                        main = "Train vs train minutes active > 0") 
    plot_frames_train_val(train = q1, validation = q1[q1.minutes_active > 0], frames = frames, ax1_lab = "validation", ax2_lab= "Validation minutes active > 0",
                        main = "Validation vs validation minutes active > 0") 


    q0 = train_active.filter(train_active.sku.isin(skus)).toPandas()
    q1 = validation_active.filter(train_active.sku.isin(skus)).toPandas()

    plot_frames_train_val(train = q0, validation = q1, frames = frames, main = "Train active vs validation")


    features_train = create_proportion_columns(train)
    features_train_active = create_proportion_columns(train_active)
    features_validation = create_proportion_columns(validation)
    features_validation_active = create_proportion_columns(validation)


    # Generate time series for 30 days
    id = spark.range(1,31)
    sku = train.select(F.col("sku")).distinct()
    dates_sku = id.crossJoin(sku)

    train = dates_sku.join(train, ["sku", "id"], how="left")
    train_active = dates_sku.join(train_active, ["sku", "id"], how="left")

    validation = dates_sku.join(validation, ["sku", "id"], how="left")
    validation_active = dates_sku.join(validation_active, ["sku", "id"], how="left")


    # Add metadata
    train = train.join(meta[["sku", "item_domain_id", "site_id"]], "sku")
    train_active = train_active.join(meta[["sku", "item_domain_id", "site_id"]], "sku")
    validation = validation.join(meta[["sku", "item_domain_id", "site_id"]], "sku")
    validation_active = validation_active.join(meta[["sku", "item_domain_id", "site_id"]], "sku")


    to_drop = ["date", "is_train", "sku_split"]
    train = train.drop(*to_drop)
    validation = validation.drop(*to_drop)
    train_active = train_active.drop(*to_drop)
    validation_active = validation_active.drop(*to_drop)


    windowval = create_window('sku', 'id', [Window.unboundedPreceding, 0])

    for item in ["item_domain_id", "site_id", "currency", "max"]:
        train = forward_fill(windowval, train, item) # fill with last non null value
        train_active = forward_fill(windowval, train_active, item) # fill with last non null value
        validation = forward_fill(windowval, validation, item) # fill with last non null value
        validation_active = forward_fill(windowval, validation_active, item) # fill with last non null value


    to_fill =  ["sold_quantity", "minutes_active", "is_active", 
                "current_price", "cumsum", "selling_rate"] + [f"rolling_cumsum_{interval}" for interval in rolling_windows] + [f"rolling_cumsum_pc_{interval}" for interval in rolling_windows] 


    to_fill = dict(zip(to_fill, [-1 for i in range(len(to_fill))]))
    train = train.na.fill(to_fill)
    train_active = train_active.na.fill(to_fill).drop("is_active")
    validation = validation.na.fill(to_fill)
    validation_active = validation_active.na.fill(to_fill).drop("is_active")


    train_active_min = train_active.select("sku", "id", "rolling_cumsum_1", 'item_domain_id', "site_id")
    validation_active_min = validation_active.select("sku", "id", "rolling_cumsum_1", 'item_domain_id', "site_id")

    validation.printSchema()

    test.printSchema()


    write_data(train,"train.parquet")
    write_data(validation,"validation.parquet")
    write_data(train_active,"train_active.parquet")
    write_data(validation_active,"validation_active.parquet")
    write_data(test,"test.parquet")

    write_data(train_active_min, "train_active_min.parquet")
    write_data(validation_active_min, "validation_active_min.parquet")

    write_data(features_train,"features_train.parquet")
    write_data(features_train_active,"features_train_active.parquet")
    write_data(features_validation,"features_validation.parquet")
    write_data(features_validation_active,"features_validation_active.parquet")

    write_data(dof_train,"dof_train.parquet")
    write_data(dof_train_active,"dof_train_active.parquet")
    write_data(dof_validation,"dof_validation.parquet")
    write_data(dof_validation_active,"dof_validation_active.parquet")

    write_data(days_activity_train, "days_activity_train.parquet")
    write_data(days_activity_validation, "days_activity_validation.parquet")
