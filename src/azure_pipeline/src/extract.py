""" Extract reviews from their source - the Google Play Store"""

from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import json
import sys
import os.path
import azureml.core
from azureml.core import Workspace, Datastore, Dataset
from azureml.pipeline.core import PipelineData

class Extract():
    def __init__(self):
        self.ws = Workspace.from_config(".azureml/config.json")
    
    def get_all_reviews(self):
        result = reviews_all(
            sys.argv[1],
            sleep_milliseconds=10, # defaults to 0
            lang='en', # defaults to 'en'
            country='us', # defaults to 'us'
            sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
        )
        return result

    def get_reviews(self,app,count,filter_score=None):
        result, continuation_token = reviews(
        app,
        lang='en', # defaults to 'en'
        country='us', # defaults to 'us'
        sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
        count=count, # defaults to 100
        filter_score_with=filter_score # defaults to None(means all score)
        )
        return result

    def save_result(self, result, filename, src_dir, targ_path):
        # Get file storage associated with the workspace
        def_file_store = Datastore(self.ws, "workspacefilestore")
        datastore = self.ws.get_default_datastore()

        # Prepare dataframe for saving to csv
        data = pd.DataFrame.from_dict(result)
        data.index.name = 'index'

        # save dataframe to local directory as csv
        filepath = os.path.join(src_dir,filename)
        data.to_csv(filepath)

        # upload the local folder from source directory to taget directory
        datastore.upload(src_dir=src_dir,target_path=targ_path)

        # save dataset in cloud location
        target_path = os.path.join(targ_path,filename)
        dataset = Dataset.Tabular.from_delimited_files(path = [(datastore,(target_path))])

        return dataset
    
    """ Register datasets with your workspace in order to share them with others and reuse them across experiments in your workspace """
    def register_dataset(self, dataset, name, description):
        ds = dataset.register(workspace=self.ws,
                                 name=name,
                                 description=description)

def run():
    extract = Extract()
    apk = 'com.fantome.penguinisle'
    print("Extracting reviews")
    result = extract.get_reviews(apk,3)
    print("Saving Azure dataset")
    dataset = extract.save_result(result, 'reviews.csv', 'data', 'data')
    print("Registering Azure dataset")
    extract.register_dataset(dataset, "reviews", "raw review data")

# if __name__== "__main__":
#     run()