import boto3
import json
from pandas import read_csv
import pandas as pd

class Comprehend():
    def __init__(self,access_key_id,secret_access_key):
        self.client = boto3.client('comprehend',
		region_name="us-east-2",
		aws_access_key_id = access_key_id,
		aws_secret_access_key = secret_access_key)
        
    def get_key_phrases(self,document):
        """Runs AWS API and returns key phrases for given sentence"""
        response = self.client.detect_key_phrases(
            Text=document,
            LanguageCode='en'
        )
        return(response)
    
    def get_all_key_phrases(self, input_file, output_file):
        key_phrases = pd.DataFrame()
        series = read_csv(input_file,parse_dates=True,squeeze=True)
        col_length = len(series.columns)
        index = 1
        print("Writing to file ............")
        for text in series["text"]:
            key_phrase = self.get_key_phrases(text)
            key_phrase["ResponseMetadata"]["OrigPhrase"] = text
            key_phrase["ResponseMetadata"]["PhraseIndex"] = index
            if(not index%5):
                print("Line number = {}".format(index))
            with open(output_file, 'a') as outfile:
                json.dump(key_phrase, outfile, indent=2)
                if(index!=len(series)):
                    outfile.write(',\n')
            index += 1