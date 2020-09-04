import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
# nltk.download('stopwords') # Uncomment to download stopwords (only do once)
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
        df = df[(df['date'].dt.year >= 2017)] # remove reviews prior to 2017
        df = df.set_index(xplot)[yplot] # set date as index
        mean_date = df.resample('M').mean().dropna() # sort by monthly average
        mean_date.plot()
        plt.show()

    def plot_word_count(self,df,index):
        count = df[index].str.len()
        plt.hist(count,100,alpha = 0.5)
        plt.xlim(right=500)
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
        cloud = WordCloud(stopwords=stop_words).generate(text)
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    
    def get_mean_rating_for_keywords(self,filename,labeled_keywords):
        total_score, total_reviews = 0,0
        reviews = read_csv(filename,parse_dates=True,squeeze=True)
        for idx, review in reviews.iterrows():
            if any(keyword.lower() in review['text'].lower() for keyword in labeled_keywords['keywords']):
                total_score += review['score']
                total_reviews += 1
        final = {
            'total_reviews':total_reviews,
            'mean_score':total_score/total_reviews,
        }
        return(final)

    def plot_mean_ratings(self,filename,labeled_keywords):
        ''' from aws ml workshop '''
        # Store number of categories
        num_categories = len(labeled_keywords)
        # Store average star ratings
        average_star_ratings = {}
        for label in labeled_keywords:
            average_star_ratings[label] = self.get_mean_rating_for_keywords(filename,labeled_keywords[label]) 
        df = pd.DataFrame.from_dict(average_star_ratings,orient='index')
        print("[debug] df = \n{}".format(df))
        print("[debug df.index = {}".format(df.index))
        barplot = sns.barplot(y=df.index, x='mean_score', data = df, saturation=1)
        if num_categories < 10:
            sns.set(rc={'figure.figsize':(10.0, 5.0)})
    
        # Set title and x-axis ticks 
        plt.title('Average Rating by Product Category')
        plt.xticks([1, 2, 3, 4, 5], ['1-Star', '2-Star', '3-Star','4-Star','5-Star'])

        # Helper code to show actual values afters bars 
        self.show_values_barplot(barplot, 0.1)

        plt.xlabel("Average Rating")
        plt.ylabel("Product Category")

        # Export plot if needed
        plt.tight_layout()
        # plt.savefig('avg_ratings_per_category.png', dpi=300)

        # Show graphic
        print("[debug] barplot = {}".format(barplot))
        plt.show()
                

vis = Visualize()
# series = read_csv('reviews.csv',parse_dates=True,squeeze=True)
# series["date"] = pd.to_datetime(series["date"]) # format date
# vis.get_average_score_per_month(series,'date','score')
# vis.plot_word_count(series,"text")
# vis.plot_word_cloud("key_phrases.json",['word1','words2'])
# login_keywords = {"category":"login","keywords":}
# security_keywords = {"category":"security","keywords":}
app_keywords = {'login':
            {"keywords":["login","sign","signin"]},
            "security":
            {"keywords":["alarm","arm","security","arming"]}}
# print("app_keywords length = {}".format(len(app_keywords)))
# print("categories = {}".format(app_keywords))
# print(vis.get_mean_rating_for_keywords("reviews.csv",login_keywords))
vis.plot_mean_ratings("review_small.csv",app_keywords)

# plot_x_y(series,'date','score')
