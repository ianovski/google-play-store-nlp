""" Extract reviews from their source - the Google Play Store"""

from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import json
import sys
import azureml.core
from azureml.core import Workspace, Datastore, Dataset
from azureml.pipeline.core import PipelineData

class Extract():
    def __init__(self):
        self.ws = Workspace.from_config("resources/config.json")
    
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

    def save_result(self, result):
        # Get file storage associated with the workspace
        def_file_store = Datastore(self.ws, "workspacefilestore")
        datastore = self.ws.get_default_datastore()
        data = pd.DataFrame.from_dict(result)
        data.index.name = 'index'
        filepath = "data/test.csv"
        output = data.to_csv(filepath)
        datastore.upload(src_dir='data',target_path='data')
        dataset = Dataset.Tabular.from_delimited_files(path = [(datastore,("data/test.csv"))])

def main():
    extract = Extract()
    result = extract.get_reviews('com.fantome.penguinisle',3)
    extract.save_result(result)

if __name__== "__main__":
    main()