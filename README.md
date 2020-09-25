# google-play-store-nlp
ML pipeline for analyzing Google Play Store reviews

## Introduction
In the competetive mobile app marketplace, focus on customer experience is critical to maintaining a loyal user-base and driving long-term business strategy.
Customer reviews provide app developers and business leaders with valuable insight into how users perceive their applications. 
Due to the unstructured nature of customer reviews, analyzing this data and deriving actionable insights can be challenging. Manually reviewing text data is also not practical or scalabale for large volumes. 
Natural language processing (NLP) enables software to ingest and analyze unstructured text, automatically generating insights such as sentiment, key phrases, and entities.
This pipeline was designed to:
1. Scrape Google Play Store reviews [complete]
2. Classify reviews based on user-defined categories [complete]
3. Visualize trends across categories [complete]
4. Input text to fine-tune an ML model (BERT) [complete]
5. Generate clusters from resulting word-embeddings [wip]
6. Automatically classify incoming [wip]

Features:
* Apache Spark
* Docker

ML Models Used:
* Word2Vec
* BERT
* K-means cluster
