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

    def __init__(self):
        self.keywords = {}

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
        try:
            mean_score = total_score/total_reviews
        except ZeroDivisionError:
            mean_score = 0
        final = {
            'total_reviews':total_reviews,
            'mean_score':mean_score,
        }
        return(final)

    def plot_mean_ratings(self,filename,labeled_keywords):
        ''' from aws ml workshop '''
        num_categories = len(labeled_keywords)
        # Store average star ratings
        average_star_ratings = {}
        for label in labeled_keywords:
            average_star_ratings[label] = self.get_mean_rating_for_keywords(filename,labeled_keywords[label]) 
        df = pd.DataFrame.from_dict(average_star_ratings,orient='index')
        # Create plot
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
        plt.show()
    
    def add_keywords(self,category,keywords):
        self.keywords[category] = {"keywords":keywords}

    def plot_category_count(self,filename,labeled_keywords):
        """ From AWS ml workshop"""
        num_categories = len(labeled_keywords)
        # Store average star ratings
        average_star_ratings = {}
        for label in labeled_keywords:
            average_star_ratings[label] = self.get_mean_rating_for_keywords(filename,labeled_keywords[label]) 
        df = pd.DataFrame.from_dict(average_star_ratings,orient='index')
        barplot = sns.barplot(y=df.index, x='total_reviews', data = df, saturation=1)

        if num_categories < 10:
            sns.set(rc={'figure.figsize':(10.0, 5.0)})
        # Set title
        plt.title("Number of Ratings per Product Category for Subset of Product Categories")

        # Set x-axis ticks to match scale 
        plt.xticks([100, 200, 300, 400], ['100', '200', '300', '400'])
        plt.xlim(0, 420)

        plt.xlabel("Number of Ratings")
        plt.ylabel("Product Category")

        plt.tight_layout()

        # Export plot if needed
        # plt.savefig('ratings_per_category.png', dpi=300)

        # Show the barplot
        plt.show()
    
    def sum_ratings_per_cetegory(self,filename,label,labeled_keywords):
        print("debug: labeled_keywords = {}".format(labeled_keywords))
        ratings_per_category = {}
        ratings_per_category[label] = {'1star':0,
                '2star':0,
                '3star':0,
                '4star':0,
                '5star':0}
        reviews = read_csv(filename,parse_dates=True,squeeze=True)
        for idx, review in reviews.iterrows():
            if any(keyword.lower() in review['text'].lower() for keyword in labeled_keywords['keywords']):
                if(review['score']==1):
                    ratings_per_category[label]['1star'] +=1
                if(review['score']==2):
                    ratings_per_category[label]['2star'] +=1
                if(review['score']==3):
                    ratings_per_category[label]['3star'] +=1
                if(review['score']==4):
                    ratings_per_category[label]['4star'] +=1
                if(review['score']==5):
                    ratings_per_category[label]['5star'] +=1
        print(ratings_per_category)

    def plot_rating_distribution_per_category(self,filename,labeled_keywords):
        ratings_per_category = {}
        for label in labeled_keywords:
            ratings_per_category[label] = self.sum_ratings_per_cetegory(filename,label,labeled_keywords[label])

    # def plot_rating_distribution_per_category(self,filename,labeled_keywords):
    #     """ From AWS ml workshop """ 
    #     average_star_ratings = {}
    #     for label in labeled_keywords:
    #         average_star_ratings[label] = self.get_mean_rating_for_keywords(filename,labeled_keywords[label]) 

    #     # Sort distribution by highest average rating per category
    #     sorted_distribution = {}
    #     df = pd.DataFrame.from_dict(average_star_ratings,orient='index')

    #     print(df)
    #     # Create grouped DataFrames by category and by star rating
    #     grouped_category = df.groupby(df.index)
    #     grouped_star = df.groupby('score')

    #     # Create sum of ratings per star rating
    #     df_sum = df.groupby(['score']).sum()

    #     # Calculate total number of star ratings
    #     total = df_sum['total_reviews'].sum()
        
    #     distribution = {}
    #     count_reviews_per_star = []
    #     i=0
    
    #     for category, ratings in grouped_category:
    #         count_reviews_per_star = []
    #         for star in ratings['mean_score']:
    #             count_reviews_per_star.append(ratings.at[i, 'total_reviews'])
    #             i=i+1;
    #         distribution[category] = count_reviews_per_star

    #     average_star_ratings = df
    #     average_star_ratings.iloc[:,0]
    #     for index, value in average_star_ratings.iloc[:,0].items():
    #         sorted_distribution[value] = distribution[value]

    #     # Build array per star across all categories
    #     star1 = []
    #     star2 = []
    #     star3 = []
    #     star4 = []
    #     star5 = []

    #     for k in sorted_distribution.keys():
    #         stars = sorted_distribution.get(k)
    #         star5.append(stars[0])
    #         star4.append(stars[1])
    #         star3.append(stars[2])
    #         star2.append(stars[3])
    #         star1.append(stars[4])
        
    #     # Plot the distributions of star ratings per product category
    #     categories = sorted_distribution.keys()

    #     total = np.array(star1) + np.array(star2) + np.array(star3) + np.array(star4) + np.array(star5)

    #     proportion_star1 = np.true_divide(star1, total) * 100
    #     proportion_star2 = np.true_divide(star2, total) * 100
    #     proportion_star3 = np.true_divide(star3, total) * 100
    #     proportion_star4 = np.true_divide(star4, total) * 100
    #     proportion_star5 = np.true_divide(star5, total) * 100

    #     # Add colors
    #     colors = ['red', 'purple','blue','orange','green']

    #     # The position of the bars on the x-axis
    #     r = range(len(categories))
    #     barHeight = 1

    #     # Plot bars
    #     if num_categories > 10:
    #         plt.figure(figsize=(10,10))
    #     else: 
    #         plt.figure(figsize=(10,5))

    #     ax5 = plt.barh(r, proportion_star5, color=colors[4], edgecolor='white', height=barHeight, label='5-Star Ratings')
    #     ax4 = plt.barh(r, proportion_star4, left=proportion_star5, color=colors[3], edgecolor='white', height=barHeight, label='4-Star Ratings')
    #     ax3 = plt.barh(r, proportion_star3, left=proportion_star5+proportion_star4, color=colors[2], edgecolor='white', height=barHeight, label='3-Star Ratings')
    #     ax2 = plt.barh(r, proportion_star2, left=proportion_star5+proportion_star4+proportion_star3, color=colors[1], edgecolor='white', height=barHeight, label='2-Star Ratings')
    #     ax1 = plt.barh(r, proportion_star1, left=proportion_star5+proportion_star4+proportion_star3+proportion_star2, color=colors[0], edgecolor='white', height=barHeight, label="1-Star Ratings")

    #     plt.title("Distribution of Reviews Per Rating Per Category",fontsize='16')
    #     plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    #     plt.yticks(r, categories, fontweight='regular')

    #     plt.xlabel("% Breakdown of Star Ratings", fontsize='14')
    #     plt.gca().invert_yaxis()
    #     plt.tight_layout()

    #     # plt.savefig('proportion_star_per_category.png', dpi=300)
    #     plt.show()
        
vis = Visualize()
# vis.get_average_score_per_month(series,'date','score')
# vis.plot_word_count(series,"text")
# stop_words = ["word1","word2"]
# vis.plot_word_cloud("key_phrases.json",stop_words)

vis.add_keywords('login',["login","sign","signin"])
vis.add_keywords('security',["alarm","arm","security","arming"])
vis.add_keywords('cameras',["cam","video","record"])
vis.add_keywords('automation',["rule","scene","automat"])
vis.add_keywords('ios',["ios","iphone"])
vis.add_keywords('android',["android","pixel","samsun","huawei"])

# vis.plot_mean_ratings("reviews.csv",vis.keywords)
# vis.plot_category_count("reviews.csv",vis.keywords)
# vis.plot_rating_distribution_per_category("reviews.csv",vis.keywords)
vis.plot_rating_distribution_per_category("reviews.csv",vis.keywords)