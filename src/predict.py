import json
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from transformers import TextClassificationPipeline

class Predict():
    def __init__(self,model_dir):
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_dir,
                                                                     id2label={
                                                                       0: 1,
                                                                       1: 2,
                                                                       2: 3,
                                                                       3: 4,
                                                                       4: 5,
                                                                       5: 6,
                                                                       6: 7
                                                                     },
                                                                     label2id={
                                                                       1: 0,
                                                                       2: 1,
                                                                       3: 2,
                                                                       4: 3,
                                                                       5: 4,
                                                                       6: 5,
                                                                       7: 6
                                                                        })
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.inference_pipeline = TextClassificationPipeline(model=self.model, 
                                                tokenizer=self.tokenizer,
                                                framework='tf',
                                                device=-1) # -1 for CPU, >=0 for GPU      

    def category_map(self,label):
        label_map = {
            "1" : "login",
            "2" : "security",
            "3" : "cameras",
            "4" : "automation",
            "5" : "network",
            "6" : "android",
            "7" : "other"
        }
        return label_map[label]
    
    def predict_classification(self,text):
        prediction = self.inference_pipeline(text)
        return self.category_map(str(prediction[0]["label"]))
    
def main():
    predict = Predict("./tmp/model")
    text = "The login does not work"
    prediction = predict.predict_classification(text)
    print("text = {} | prediction = {}".format(text,prediction))
    

# if __name__ == "__main__":
#     main()
