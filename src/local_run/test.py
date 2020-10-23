import pandas as pd
import csv

from predict import Predict
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

SEQ_LENGTH = 512
class Test():
    def __init__(self,filepath):
        self.reviews = pd.read_csv(filepath)

    def truncate_element(self,element):
        if(len(element) > SEQ_LENGTH):
            return(element[0:SEQ_LENGTH])
        return element

    def predict_csv(self):
        predict = Predict("./tmp/model")

        # Truncate reviews to max sequence length (512)
        split_reviews = self.reviews["text"].str.split()
        truncated = pd.Series(list(map(self.truncate_element, split_reviews)))
        return list(map(predict.predict_classification,self.reviews["text"].str.split())) 
    
    def evaluate(self, actual, predicted):
        print("Evaluating model....")
        predicted_series = pd.Series(predicted)
        frame = {"actual" : actual, "predicted": predicted}
        eval = pd.DataFrame(frame)
        
        print(classification_report(y_true=eval["actual"],y_pred=eval["predicted"]))
        print('Accuracy: ', accuracy_score(y_pred=eval["actual"], y_true=eval["predicted"]))

def main():
    test = Test("labelled_reviews.csv")
    pred = test.predict_csv()
    true = test.reviews["category"]
    test.evaluate(true,pred)

if __name__== "__main__":
    main()