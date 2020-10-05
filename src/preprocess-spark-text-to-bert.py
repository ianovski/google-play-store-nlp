# Parts of this code taken from data-science-on-aws: https://github.com/data-science-on-aws/workshop

from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv
import collections
import subprocess
import sys
import zipfile
import wget
import pandas as pd
from pandas import read_csv
import json
import re
import io
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pip', '--upgrade'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wrapt', '--upgrade', '--ignore-installed'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==1.11.0', '--ignore-installed'])
import tensorflow as tf
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==2.8.0'])
from transformers import DistilBertTokenizer

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameWriter
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import split
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.context import SparkContext
from pyspark import SparkConf


from visualize import Visualize

if(not os.path.isfile('src/modeling.py')):
    wget.download('https://raw.githubusercontent.com/google-research/bert/master/modeling.py',out='src/modeling.py')
if(not os.path.isfile('src/extract_features.py')):
    wget.download('https://raw.githubusercontent.com/google-research/bert/master/extract_features.py',out='src/extract_features.py')
if(not os.path.isfile('src/tokenization.py')):
    wget.download('https://raw.githubusercontent.com/google-research/bert/master/tokenization.py',out='src/tokenization.py')
    
import modeling
import extract_features
import tokenization

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# We set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 64
DATA_COLUMN = 'review_body'
LABEL_COLUMN = 'star_rating'
LABEL_VALUES = ['login','security','cameras','automation','network','android','other']
COL_NUM = 5 # Number of columns in review csv

label_map = {}
for (i, label) in enumerate(LABEL_VALUES):
    label_map[label] = i


class InputFeatures(object):
  """BERT feature vectors."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class Input(object):
  """A single training/test input for sequence classification."""

  def __init__(self, text, label=None):
    """Constructs an Input.
    Args:
      text: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.text = text
    self.label = label
    

def convert_input(label, text):
    # First, we need to preprocess our data so that it matches the data BERT was trained on:
    #
    # 1. Lowercase our text (if we're using a BERT lowercase model)
    # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
    # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
    #
    # Fortunately, the Transformers tokenizer does this for us!
    #
#    tokens = tokenizer.tokenize(text_input.text)

    # Next, we need to do the following:
    #
    # 4. Map our words to indexes using a vocab file that BERT provides
    # 5. Add special "CLS" and "SEP" tokens (see the [readme](https://github.com/google-research/bert))
    # 6. Append "index" and "segment" tokens to each input (see the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))
    #
    # Again, the Transformers tokenizer does this for us!
    #
    encode_plus_tokens = tokenizer.encode_plus(text,
                                               pad_to_max_length=True,
                                               max_length=MAX_SEQ_LENGTH)

    # Convert the text-based tokens to ids from the pre-trained BERT vocabulary
    input_ids = encode_plus_tokens['input_ids']
    # Specifies which tokens BERT should pay attention to (0 or 1)
    input_mask = encode_plus_tokens['attention_mask']
    # Segment Ids are always 0 for single-sequence tasks (or 1 if two-sequence tasks)
    segment_ids = [0] * MAX_SEQ_LENGTH

    # Label for our training data (star_rating 1 through 5)
    label_id = label_map[label]

    return {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'label_ids': [label_id]}


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(',')


def parse_args():
    # Unlike SageMaker training jobs (which have `SM_HOSTS` and `SM_CURRENT_HOST` env vars), processing jobs to need to parse the resource config file directly
    resconfig = {}
    try:
        with open('/opt/ml/config/resourceconfig.json', 'r') as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print('/opt/ml/config/resourceconfig.json not found.  current_host is unknown.')
        pass # Ignore

    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--hosts', type=list_arg,
        default=resconfig.get('hosts', ['unknown']),
        help='Comma-separated list of host names running the job'
    )
    parser.add_argument('--current-host', type=str,
        default=resconfig.get('current_host', 'unknown'),
        help='Name of this host running the job'
    )
    parser.add_argument('--input-data', type=str,
        default='/opt/ml/processing/input/data',
    )
    parser.add_argument('--output-data', type=str,
        default='/opt/ml/processing/output',
    )
    return parser.parse_args()

class Training():

    def __init__(self, filename):
        self.filename = filename
        self.sc = SparkContext("local", "first app")
        jar_paths = ["/home/alex/.m2/repository/org/tensorflow/tensorflow-hadoop/1.11.0/tensorflow-hadoop-1.11.0.jar",
                    "/home/alex/.m2/repository/org/tensorflow/spark-tensorflow-connector_2.12/1.11.0/spark-tensorflow-connector_2.12-1.11.0.jar"]
        self.spark = SparkSession.builder\
            .config('spark.jars',jar_paths)\
            .appName('GoogleReviewsSparkProcessor')\
            .getOrCreate()
        self.reviews = None


    def read_reviews_csv(self):
        self.reviews = self.remove_new_lines(self.filename).toDF().select('_5')

    def remove_new_lines(self, filename):
        rdd = self.sc.wholeTextFiles(filename)\
            .map(lambda x: re.sub(r'(?!(([^"]*"){2})*[^"]*$),', ' ', x[1].replace("\r\n", ",").replace(",,", ",")).split(","))\
            .flatMap(lambda x: [x[k:k + COL_NUM] for k in range(0, len(x), COL_NUM)])
        return rdd 

    def unzip_file(self, filename):
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall()
            download_file_command = "rm " + filename
            os.system(download_file_command)

    def get_bert(self):
        """Download the 2018 uncased pretrained BERT model"""
        directory = 'uncased_L-24_H-1024_A-16'
        filename = directory + '.zip'
        if(not os.path.exists(directory)):
            print("[debug] Downloading BERT weights and config file......")
            url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
            wget.download(url)
            self.unzip_file(filename)
    
    def train_bert(self):
        """Finetune BERT using customer reviews"""
        df = self.reviews.toPandas()
        df.to_csv("input.txt", index=False, header=False)
        out = os.system("python3 extract_features.py \
            --input_file=input.txt \
            --output_file=output.json \
            --vocab_file=uncased_L-24_H-1024_A-16/vocab.txt \
            --bert_config_file=uncased_L-24_H-1024_A-16/bert_config.json \
            --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
            --layers=-1,-5 \
            --max_seq_length=256 \
            --batch_size=8")
        bert_output = pd.read_json("output.json", lines = True)
        return(bert_output)
    
    def remove_new_line_chars(self):
        print("[debug] reviews.shape = {}".format(self.reviews.shape))
        self.reviews = self.reviews.replace('\n', '', regex=True)

    def read_model(self, filepath):
        model = pd.read_json(filepath, lines = True)
        return (model)
        
    def extract_labels(self,vis,filename):
        vis = Visualize()
        vis.read_reviews_labelled_csv(filename)
        labels = vis.get_reviews_labelled()['category']
        return labels

    # def transform(self, s3_input_data, s3_output_train_data, s3_output_validation_data, s3_output_test_data): 
    def transform(self,output_train_data,output_validation_data,output_test_data):
        
        # Define Spark DF schema
        schema = StructType([
            StructField('userName', StringType(), True),
            StructField('date', StringType(), True),
            StructField('score', IntegerType(), True),
            StructField('scoreText', IntegerType(), True),
            StructField('text', StringType(), True),
            StructField('category', StringType(), True)
        ])

        # Import as pandas dataframe to maintain schema
        pdDF = pd.read_csv(self.filename)

        # Convert to Spark dataframe
        df_csv = self.spark.createDataFrame(pdDF,schema = schema)
        # df_csv = self.remove_new_lines(self.filename).toDF()

        # # This dataset should already be clean, but always good to double-check
        # print('Showing null review_body rows...')
        # df_csv.where(col('review_body').isNull()).show()

        # print('Showing cleaned csv')
        # df_csv_dropped = df_csv.na.drop(subset=['review_body'])
        # df_csv_dropped.show()

        # # TODO:  Balance
        features_df = df_csv.select(['category', 'text'])

        tfrecord_schema = StructType([
        StructField("input_ids", ArrayType(IntegerType(), False)),
        StructField("input_mask", ArrayType(IntegerType(), False)),
        StructField("segment_ids", ArrayType(IntegerType(), False)),
        StructField("label_ids", ArrayType(IntegerType(), False))
        ])

        bert_transformer = udf(lambda text, label: convert_input(text, label), tfrecord_schema)

        self.spark.udf.register('bert_transformer', bert_transformer)

        transformed_df = features_df.select(bert_transformer('category', 'text').alias('tfrecords'))
        transformed_df.show(truncate=False)

        flattened_df = transformed_df.select('tfrecords.*')
        flattened_df.show()

        # # Split 90-5-5%
        train_df, validation_df, test_df = flattened_df.randomSplit([0.9, 0.05, 0.05])
        
        train_df.write.format('json').save(path=output_train_data)
        print('Wrote to output file:  {}'.format(output_train_data))
    
        validation_df.write.format('json').save(path=output_validation_data)
        print('Wrote to output file:  {}'.format(output_validation_data))

        test_df.write.format('json').save(path=output_test_data)    
        print('Wrote to output file:  {}'.format(output_test_data))

        restored_test_df = self.spark.read.format('json').load(path=output_test_data)
        restored_test_df.show()

def main():
    training = Training('labelled_reviews_small.csv')
    training.transform('output_train_data.json','output_validation_data.json','output_test_data.json')
    exit()
    # training.read_reviews_csv()
    # training.get_bert()
    # model = training.train_bert()
    print("Training Complete")

    model = training.read_model('output.json')
    # print(model)
    
    vis = Visualize()
    labels = training.extract_labels(vis, 'labelled_reviews.csv')
    vis.plot_tsne(model,labels)

    # spark = SparkSession.builder.appName('AmazonReviewsSparkProcessor').getOrCreate()

    # # Convert command line args into a map of args
    # args_iter = iter(sys.argv[1:])
    # args = dict(zip(args_iter, args_iter))

    # # Retrieve the args and replace 's3://' with 's3a://' (used by Spark)
    # s3_input_data = args['s3_input_data'].replace('s3://', 's3a://')
    # print(s3_input_data)

    # s3_output_train_data = args['s3_output_train_data'].replace('s3://', 's3a://')
    # print(s3_output_train_data)

    # s3_output_validation_data = args['s3_output_validation_data'].replace('s3://', 's3a://')
    # print(s3_output_validation_data)

    # s3_output_test_data = args['s3_output_test_data'].replace('s3://', 's3a://')
    # print(s3_output_test_data)

    # transform(spark, 
    #           s3_input_data, 
    #           '/opt/ml/processing/output/bert/train', 
    #           '/opt/ml/processing/output/bert/validation', 
    #           '/opt/ml/processing/output/bert/test',
    #     # s3_output_train_data, s3_output_validation_data, s3_output_test_data
    # )


if __name__ == "__main__":
    main()
