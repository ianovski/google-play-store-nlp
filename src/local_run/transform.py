class Transform():
    def sum_ratings_per_cetegory(self,reviews,labeled_keywords):
        # Create empty dictionary
        ratings_per_category = {'1star':0,
                '2star':0,
                '3star':0,
                '4star':0,
                '5star':0}

        # Populate dictionary with user ratings
        for idx, review in reviews.iterrows():
            if any(keyword.lower() in review['text'].lower() for keyword in labeled_keywords['keywords']):
                if(review['score']==1):
                    ratings_per_category['1star'] +=1
                elif(review['score']==2):
                    ratings_per_category['2star'] +=1
                elif(review['score']==3):
                    ratings_per_category['3star'] +=1
                elif(review['score']==4):
                    ratings_per_category['4star'] +=1
                elif(review['score']==5):
                    ratings_per_category['5star'] +=1
        return (ratings_per_category)

    def classify_review_using_keywords(self,text,labeled_keywords):
        for label in labeled_keywords:
            if any(keyword.lower() in text.lower() for keyword in labeled_keywords[label]["keywords"]):
                return(label)
        return ("other")
    
    def get_mean_rating_for_keywords(self,reviews_labelled):
        df = reviews_labelled[['category', 'score']]
        category_means = df.groupby('category').mean()
        return category_means
    
    def get_category_counts(self,df):
        df = df['category']
        category_counts = df.value_counts().to_frame()
        category_counts.reset_index(level=0, inplace=True)
        category_counts.columns = ['category','counts']
        return category_counts        