import time
import random
import pandas as pd
from glob import glob
import argparse
import json
import subprocess
import sys
import os
import tensorflow as tf
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers.configuration_distilbert import DistilBertConfig
MAX_SEQ_LENGTH = 128

class Train():
    def __init__(self):
        self.epochs=3
        self.learning_rate = 0.00001
        self.epsilon = 0.00000001
        self.train_batch_size=128
        self.validation_batch_size=128
        self.test_batch_size=128
        self.train_steps_per_epoch=100
        self.validation_steps=100
        self.test_steps=100
        self.train_instance_count=1
        self.train_instance_type='ml.c5.9xlarge'
        self.train_volume_size=1024
        self.use_xla=True
        self.use_amp=True
        self.freeze_bert_layer=False
        self.enable_sagemaker_debugger=True
        self.enable_checkpointing=False
        self.enable_tensorboard=False
        self.input_mode='Pipe'
        self.run_validation=True
        self.run_test=True
        self.run_sample_predictions = True
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.model = None
        self.callbacks = []

    def select_data_and_label_from_record(self,record):
        x = {
            'input_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'segment_ids': record['segment_ids']
        }
        y = record['label_ids']

        return (x, y)

    def file_based_input_dataset_builder(self,
                                    channel,
                                     input_filenames,
                                     pipe_mode,
                                     is_training,
                                     drop_remainder):

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.

        if pipe_mode:
            print('***** Using pipe_mode with channel {}'.format(channel))
            from sagemaker_tensorflow import PipeModeDataset
            dataset = PipeModeDataset(channel=channel,
                                      record_format='TFRecord')
        else:
            print('***** Using input_filenames {}'.format(input_filenames))
            dataset = tf.data.TFRecordDataset(input_filenames)

        dataset = dataset.repeat(100)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
            "input_mask": tf.io.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([MAX_SEQ_LENGTH], tf.int64),
            "label_ids": tf.io.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            return tf.io.parse_single_example(record, name_to_features)
        
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                # batch_size=8,
                batch_size=8,
                drop_remainder=drop_remainder,
                num_parallel_calls=tf.data.experimental.AUTOTUNE))

        dataset.cache()

        if is_training:
            dataset = dataset.shuffle(seed=42,
                                      buffer_size=10,
                                      reshuffle_each_iteration=True)
        return dataset

    def read_training_data(self, filepath):
        train_data = filepath
        train_data_filenames = glob('{}/*.tfrecord'.format(train_data))
        print('train_data_filenames {}'.format(train_data_filenames))

        self.train_dataset = self.file_based_input_dataset_builder(
            channel='train',
            input_filenames=train_data_filenames,
            pipe_mode=False,
            is_training=True,
            drop_remainder=False).map(self.select_data_and_label_from_record)
    
    def read_validation_data(self, filepath):
        validation_data = filepath
        validation_data_filenames = glob('{}/*.tfrecord'.format(validation_data))
        print('validation_data_filenames {}'.format(validation_data_filenames))

        self.validation_dataset = self.file_based_input_dataset_builder(
            channel='validation',
            input_filenames=validation_data_filenames,
            pipe_mode=False,
            is_training=False,
            drop_remainder=False).map(self.select_data_and_label_from_record)
    
    def read_test_data(self, filepath):
        test_data = filepath
        test_data_filenames = glob('{}/*.tfrecord'.format(test_data))
        print(test_data_filenames)
        self.test_dataset = self.file_based_input_dataset_builder(
            channel='test',
            input_filenames=test_data_filenames,
            pipe_mode=False,
            is_training=False,
            drop_remainder=False).map(self.select_data_and_label_from_record)
    
    def load_pretrained_bert_model(self):
        CLASSES = [1, 2, 3, 4, 5, 6]
        
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
                                          num_labels=len(CLASSES))
        
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', 
                                            config=config)

    def setup_custom_classifier_model(self):
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric=tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model.layers[0].trainable=not self.freeze_bert_layer
        self.model.summary()

        log_dir = './tmp/tensorboard/'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        
        self.callbacks.append(tensorboard_callback)

        history = self.model.fit(self.train_dataset,
                    shuffle=True,
                    epochs=self.epochs,
                    steps_per_epoch=self.train_steps_per_epoch,
                    validation_data=self.validation_dataset,
                    validation_steps=self.validation_steps,
                    callbacks=self.callbacks)

    def evaluate_model(self):
        test_history = self.model.evaluate(self.test_dataset,
                        steps=self.test_steps,                            
                        callbacks=self.callbacks)
        
    def save_model(self, filename):
        model_dir = './tmp/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filepath = model_dir + filename
        model.save_pretrained(model_dir)
    
def main():
    train = Train()
    # with(tf.device("/CPU:0")):
    train.read_training_data('output_train_data')
    train.read_validation_data('output_validation_data')
    train.read_test_data('output_test_data')

    train.load_pretrained_bert_model()
    train.setup_custom_classifier_model()
    train.evaluate_model()
    train.save_model()



if __name__ == "__main__":
    main()