import gensim

class W2V():
    def __init__(self):
        self.model_flag = False
        self.model = None

    # Get similar keywords with w2v
    def get_similar_words(self,word):
        # Import model if not already imported
        if(not self.model_flag):
            self.model = gensim.models.KeyedVectors.load_word2vec_format('resources/model/GoogleNews-vectors-negative300.bin', binary=True)
            self.model_flag = True
        similar_words = self.model.wv.most_similar(word)
        similar_words = [a_tuple[0] for a_tuple in similar_words]
        return(similar_words)