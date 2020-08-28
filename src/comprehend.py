import boto3
import json

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