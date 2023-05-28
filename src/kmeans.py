import argparse
from pyspark.sql import SparkSession

from preprocessing.load_data import load
from model import PySparkKMeans


def main(spark, train_data_path: str, save_path: str, params: dict):
    df = load(spark, train_data_path)

    trainer = PySparkKMeans(df, params)
    trainer.train()
    trainer.save_model(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", default="../data/en.openfoodfacts.org.products.csv", required=True)
    parser.add_argument("save_path", required=True)
    parser.add_argument("k", default=2)
    parser.add_argument("max_iter", default=5)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.driver.cores", "2") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "10g") \
        .appName("openfood-trainer").getOrCreate()

    main(spark, args.train_path, args.save_path, {"k": args.k, "max_iter": args.max_iter})
