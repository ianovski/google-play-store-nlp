import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
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
        def _show_on_plot(self,ax):
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
    
    def plot_word_cloud(self,filename):
        text = ""
        with open(filename, 'r') as f:
            data = json.load(f)
        for review in data:
            for phrase in review['KeyPhrases']:
                if(phrase['Score']>0.9):
                    text = text + phrase["Text"] + " "
        stop_words = stopwords.words('english')
        stop_words.extend(['Roger','app','issue','Rogers'])
        cloud = WordCloud(stopwords=stop_words).generate(text)
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

vis = Visualize()
# series = read_csv('reviews.csv',parse_dates=True,squeeze=True)
# series["date"] = pd.to_datetime(series["date"]) # format date
# vis.get_average_score_per_month(series,'date','score')
# vis.plot_word_count(series,"text")
vis.plot_word_cloud("key_phrases.json")

# plot_x_y(series,'date','score')
