from azureml.core import Workspace, Dataset
import json
import pandas as pd
import os.path

class Preprocess():

    def __init__(self):
        self.ws = Workspace.from_config()
        self.keywords = {}
        self.reviews_labelled = pd.DataFrame()
        self.keyword_flag = 'y'

    def classify_review_using_keywords(self,text,labeled_keywords):
        for label in labeled_keywords:
            if any(keyword.lower() in text.lower() for keyword in labeled_keywords[label]["keywords"]):
                return(label)
        return ("other")

    def label_reviews(self, reviews, labelled_keywords):
        reviews_labelled = reviews
        reviews_labelled['category'] = reviews.apply(lambda row: self.classify_review_using_keywords(row['content'], labelled_keywords),axis=1)
        return reviews_labelled
    
    def add_keywords(self, category, keywords):
        # Find similar keywords using word2vec model
        # w2v = W2V()
        # similar_words = w2v.get_similar_words(category)
        # keywords = keywords + similar_words
        
        # save keywords to dictionary
        self.keywords[category] = {"keywords":keywords}
    
    def save_result_json(self, data,  filename, src_dir, targ_path):
        src_path = os.path.join(src_dir, filename)
        with open(src_path,'w') as f:
            json.dump(data,f)
    
        # Get file storage associated with the workspace
        datastore = self.ws.get_default_datastore()

        # upload local folder from source directory to taget directory
        datastore.upload(src_dir=src_dir, target_path=targ_path)
    
    def get_user_input(self):
        while(self.keyword_flag=="y"):
            category = input("Enter category:\n")
            keywords = input("Enter relevant keywords:\n")
            keywords = keywords.split(',')
            self.add_keywords(category,keywords)
            self.keyword_flag = input("Press 'y' + ENTER to add more keywords. Press any other letter + ENTER to continue:\n")
    
    def save_result_csv(self, result, filename, src_dir, targ_path):
        # Get file storage associated with the workspace
        datastore = self.ws.get_default_datastore()

        # save dataframe to local directory as csv
        filepath = os.path.join(src_dir,filename)
        result.to_csv(filepath)

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
    preprocess = Preprocess()
    
    dataset = Dataset.get_by_name(preprocess.ws, name='reviews')
    df = dataset.to_pandas_dataframe()
    preprocess.get_user_input()
    preprocess.save_result_json(preprocess.keywords, 'keywords.json', 'data',  'data/keywords.json')
    reviews_labelled = preprocess.label_reviews(df, preprocess.keywords)
    preprocess.save_result_csv(reviews_labelled, "reviews_labelled", 'data', 'data')
    preprocess.register_dataset(dataset, "labelled_reviews", "reviews with category labels")

if __name__ == "__main__":
    run()
