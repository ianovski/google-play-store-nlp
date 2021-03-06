import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from transform import Transform
from w2v import W2V
import nltk
# nltk.download('stopwords') # Uncomment to download stopwords (only use once)
from nltk.corpus import stopwords

class Visualize():
    sns.set_style = 'seaborn-whitegrid'

    sns.set(rc={"font.style":"normal",
            "axes.facecolor":"white",
            'grid.color': '.8',
            'grid.linestyle': '-',
            "figure.facecolor":"white",
            "figure.titlesize":20,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":True,
            'axes.labelsize':10,
            'figure.figsize':(10.0, 10.0),
            'xtick.labelsize':10,
            'font.size':10,
            'ytick.labelsize':10})

    def __init__(self):
        self.keywords = {}
        self.reviews_labelled = pd.DataFrame()
        self.reviews = pd.DataFrame()
        
    def show_values_barplot(self,axs, space):
        def _show_on_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = round(float(p.get_width()),2)
                ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_plot(ax)
        else:
            _show_on_plot(axs)

    def plot_x_y(self,df,xplot,yplot):
        df.set_index(xplot)[yplot].plot()
        plt.show()

    def get_average_score_per_month(self,df,xplot,yplot):
        # Remove reviews prior to 2017
        df = df[(df['date'].dt.year >= 2017)]

        # Set date as index
        df = df.set_index(xplot)[yplot] 

        # sort by monthly average
        mean_date = df.resample('M').mean().dropna() 
        mean_date.plot()
        plt.show()
    
    def set_reviews(self, reviews):
        self.reviews = reviews
    
    def get_reviews(self):
        return self.reviews

    def read_reviews_csv(self, filename):
        reviews = read_csv(filename, parse_dates=True, squeeze=True)
        self.set_reviews(reviews)

    def plot_word_count(self,review_index):
        reviews = self.get_reviews()
        count = reviews[review_index].str.len()
        plt.hist(count,100,alpha = 0.5)
        plt.xlim(0, 500)
        plt.title("Word count")
        plt.xlabel("Number of words")
        plt.ylabel("Frequency")
        plt.show()

    def plot_word_cloud(self,filename,add_stop=None):
        text = ""
        with open(filename, 'r') as f:
            data = json.load(f)
        for review in data:
            for phrase in review['KeyPhrases']:
                if(phrase['Score']>0.9):
                    text = text + phrase["Text"] + " "
        stop_words = stopwords.words('english')
        stop_words.extend(add_stop)
        cloud = WordCloud(stopwords=stop_words,background_color='white',colormap='coolwarm').generate(text)
        plt.imshow(cloud, interpolation='bilinear')
        plt.title('Word Cloud',fontsize=20)
        plt.axis("off")
        plt.gca().set_facecolor('white')
        plt.show()
    
    def set_reviews_labelled(self,reviews_labelled):
        self.reviews_labelled = reviews_labelled

    def get_reviews_labelled(self):
        return self.reviews_labelled

    def save_reviews_labelled(self,filename):
        self.get_reviews_labelled().to_csv(filename, index=False)
    
    def read_reviews_labelled_csv(self, filename):
        reviews = read_csv(filename, parse_dates=True, squeeze=True)
        self.set_reviews_labelled(reviews)

    def plot_category_count_with_unlabelled(self):
        tm = Transform()
        df = tm.get_category_counts(self.get_reviews_labelled())
        df.plot(kind='bar',
                                    figsize=(14,8),
                                    title="Customer review category count")
        plt.xlabel("Review category")
        plt.ylabel("Frequency")
        plt.show()

    def label_reviews(self, tm, reviews, labelled_keywords):
        reviews_labelled = reviews
        reviews_labelled['category'] = reviews.apply(lambda row: tm.classify_review_using_keywords(row['text'], labelled_keywords),axis=1)
        return reviews_labelled

    def plot_mean_ratings(self, labelled_keywords, filename):
        tm = Transform()
        num_categories = len(labelled_keywords)

        # Label reviews with categories
        reviews_labelled = self.label_reviews(tm, self.get_reviews(), labelled_keywords)
        self.set_reviews_labelled(reviews_labelled)

        self.save_reviews_labelled(filename)

        mean_ratings = tm.get_mean_rating_for_keywords(reviews_labelled)
        count_ratings = tm.get_category_counts(reviews_labelled)

        # Create plot
        barplot = sns.barplot(y=mean_ratings.index, x='score', data = mean_ratings, saturation=1)
        # if num_categories < 10:
            # sns.set(rc={'figure.figsize':(10.0, 6.0)})

        # Set title and x-axis ticks 
        plt.title('Average Rating by Product Category', fontsize=20)
        plt.xticks([1, 2, 3, 4, 5], ['1-Star', '2-Star', '3-Star','4-Star','5-Star'], fontsize=14)
        plt.yticks(fontsize=14)


        # Helper code to show actual values afters bars 
        self.show_values_barplot(barplot, 0.1)
        plt.xlabel("Average Rating",fontsize=18)
        plt.ylabel("Product Category",fontsize=18)

        # Export plot if needed
        plt.tight_layout()
        # plt.savefig('avg_ratings_per_category.png', dpi=300)

        # Show graphic
        plt.show()

    def add_keywords(self, category, keywords):
        # w2v = W2V()
        # similar_words = w2v.get_similar_words(category)
        # keywords = keywords + similar_words
        self.keywords[category] = {"keywords":keywords}

    def plot_category_count(self,labeled_keywords):
        """ Partially from AWS ml workshop"""
        tm = Transform()
        num_categories = len(labeled_keywords)
        # Store average star ratings
        average_star_ratings = {}

        df = tm.get_category_counts(self.get_reviews_labelled())
        
        barplot = sns.barplot(y='category', x='counts', data = df, saturation=1)

        if num_categories < 10:
            sns.set(rc={'figure.figsize':(10.0, 5.0)})

        # Set title
        plt.title("Number of Ratings per Product Category", fontsize=20)

        # Set x-axis ticks to match scale 
        xrange = list(range(100,700,100)) # need to generalize this
        plt.xticks(xrange, map(str,xrange), fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 720)

        plt.xlabel("Number of Ratings",fontsize=18)
        plt.ylabel("Product Category",fontsize=18)

        plt.tight_layout()

        # Export plot if needed
        # plt.savefig('ratings_per_category.png', dpi=300)

        # Show the barplot
        plt.show()
    

    def plot_rating_distribution_per_category(self,labeled_keywords):
        """Partially from AWS ML workshop"""
        tm = Transform()
        ratings_per_category = {}

        # Populate dictionary for each categories' user ratings
        for label in labeled_keywords:
            ratings_per_category[label] = tm.sum_ratings_per_cetegory(self.get_reviews(), labeled_keywords[label])

        df = pd.DataFrame.from_dict(ratings_per_category,orient='index')
        categories = ratings_per_category.keys()

        # Convert ratings to list
        star1 = df['1star'].tolist()
        star2 = df['2star'].tolist()
        star3 = df['3star'].tolist()
        star4 = df['4star'].tolist()
        star5 = df['5star'].tolist()

        # Calculate total number of reviews for all categories
        total = df.sum().sum()
        
        # Calculate relative value for eahc rating
        proportion_star1 = np.true_divide(star1, total) * 100
        proportion_star2 = np.true_divide(star2, total) * 100
        proportion_star3 = np.true_divide(star3, total) * 100
        proportion_star4 = np.true_divide(star4, total) * 100
        proportion_star5 = np.true_divide(star5, total) * 100

        colors = ['red', 'purple','blue','orange','green']

        # The position of the bars on the x-axis
        r = range(len(categories))
        barHeight = 1
        num_categories = df.shape[0]
        if num_categories > 10:
            plt.figure(figsize=(10,10))
        else: 
            plt.figure(figsize=(10,5))

        # plot horizontal bar graph
        ax5 = plt.barh(r, proportion_star5, color=colors[4], edgecolor='white', height=barHeight, label='5-Star Ratings')
        ax4 = plt.barh(r, proportion_star4, left=proportion_star5, color=colors[3], edgecolor='white', height=barHeight, label='4-Star Ratings')
        ax3 = plt.barh(r, proportion_star3, left=proportion_star5+proportion_star4, color=colors[2], edgecolor='white', height=barHeight, label='3-Star Ratings')
        ax2 = plt.barh(r, proportion_star2, left=proportion_star5+proportion_star4+proportion_star3, color=colors[1], edgecolor='white', height=barHeight, label='2-Star Ratings')
        ax1 = plt.barh(r, proportion_star1, left=proportion_star5+proportion_star4+proportion_star3+proportion_star2, color=colors[0], edgecolor='white', height=barHeight, label="1-Star Ratings")

        plt.title("Distribution of Reviews Per Rating Per Category",fontsize='20')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.yticks(r, categories, fontweight='regular',fontsize='14')
        plt.xticks(fontsize='14')

        plt.xlabel("% Breakdown of Star Ratings", fontsize='16')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # plt.savefig('proportion_star_per_category.png', dpi=300)
        plt.show()
    
    def plot_categories_time_series(self):
        df = self.get_reviews_labelled()

        # Convert int score to float
        df['score'] = df['score'].astype(float)

        # Sort by date
        df = df.sort_values(by='date')

        # Convert index to datetime
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

        # Filter for 2019+
        # TODO: don't hardcode date
        df = df[(df.index > '2019-1-1')]

        # Resample for quarterly time series
        grouped = df.groupby(['category'])
        data_columns = ['score','category']
        df_quarterly_mean = grouped[data_columns].resample('Q').mean()
        
        # Plot
        fix, ax = plt.subplots()
        for key, group in grouped:
            group['score'].plot(label=key, ax=ax)

        plt.legend(loc='best')

        plt.show()

	# Title: Visualizing Word Vectors with t-SNE
	# Author: Delaney, J
	# Date: 2017
	# Code version: 3.0
	# Availability: https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    def plot_tsne(self, model,labels): 
        """Plot 2D t-SNE representation of word embeddings"""
        print("\nPlotting t-SNE...")
        # labels = []
        tokens = []
        
        for i in range(len(model)):
            tokens.append(model['features'][i][0]['layers'][0]['values'])

        # for word in model.wv.vocab:
        #     tokens.append(model[word])
        #     labels.append(word)
    
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(25, 25)) 
        for i in range(len(x)-1):
            plt.scatter(x[i+1], y[i+1])
            plt.annotate(labels[i],
                    xy=(x[i+1], y[i+1]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
        # filename = 'figures/'+plot_name+'.png'
        # plt.savefig(filename)
        plt.show()

def main():
    vis = Visualize()
    # vis.read_reviews_csv("reviews.csv")
    # vis.plot_word_count("text")
    # stop_words = ["Rogers", "Roger", "app", "phone"]
    # vis.plot_word_cloud("key_phrases.json", stop_words)

    vis.add_keywords('login', ["login", "sign", "signin","log","logging"])
    vis.add_keywords('security', ["alarm", "arm", "security", "arming"])
    vis.add_keywords('cameras', ["cam", "video", "record","camera","cameras",'cctv'])
    vis.add_keywords('automation', ["rule", "scene", "automat",'lights','lock'])
    vis.add_keywords('network', ["network","wifi",'internet','connected'])
    # vis.add_keywords('ios', ["ios", "iphone"])
    vis.add_keywords('android', ["android", "pixel", "samsun", "huawei",' lg '])
      
    # vis.plot_mean_ratings(vis.keywords)
    # vis.plot_category_count(vis.keywords)
    # vis.plot_rating_distribution_per_category(vis.keywords)
    
    reviews_labelled = read_csv("labelled_reviews.csv")
    vis.set_reviews_labelled(reviews_labelled)
    vis.plot_categories_time_series()

if __name__ == "__main__":
    main()
