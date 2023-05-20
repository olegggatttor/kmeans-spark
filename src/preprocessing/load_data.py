def load(spark, path):
    return spark.read.option("sep", "\t").option("header", True).csv(path)
