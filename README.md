
# Document Classification with Scikit Learn

## Description

This workshop provides an introduction Python and the scikit-learn module. For a working example, we will train a machine learning model to assess the sentiment (positive or negative) of a movie review.

This material is designed for programmers, data scientists, and researchers who would like to get started with machine learning for natural language processing and analysis.

Scikit-learn is a widely used Python based machine learning and natural language processing module. Scikit-learn provides methods for supervised and unsupervised learning, implementations of different machine learning algorithms, and tools for common data preparation tasks involved in machine learning programming projects.

This workshop will focus on programming with scikit-learn, rather than the theory behind ML algorithms. The material here is designed for a 2-3 hour workshop and consists of two workbooks. The first workbook introduces machine learning, document categorization, and sentiment analysis using a very small dataset. The second workbook applies these techniques to a larger dataset using a scikit-learn pipeline.

Through these two workbooks, participants will use scikit-learn to:

1. Find the vocabulary used in a collection of documents (a "bag of words")
2. Create a term frequency vector for each document in a collection
3. Scale term frequency vectors to reflect the relevance of certain terms
4. Use term frequency vectors and sentiment scores for each document to train a machine learning model to recognize and assess document sentiment
5. Apply a machine learning model to assess the sentiment (or category) of new documents in the collection.

## Objectives

- Reading in data
- Converting data into a dataframe (or comma separated file) format
- Cleaning data
  - Removing punctuation
  - Removing capitalization
  - Removing non-alphanumeric characters
  - Tokenization
  - Removing stop words
  - Using a pre-defined vocabulary
  - Stemming
  - Lemmatizing
- Vectorizing
  - Count Vector
  - tf-idf
  - n-grams
- Feature engineering
- Model Building
  - Creating a Random Forest Model
  - not covered - Gradient Boosting, other ML algorithms
- Model Evaluation
  - Accuracy, Precision, Recall
- Model Application and Insight
  - Feature importance
  - Predictions

### Prerequisites

This tutorial is intended for participants who have basic programming experience with Python, including loops, conditionals, and lists. No previous experience with scikit-learn or machine learning is expected. Some familiarity with pandas will help, but we will cover these concepts in the context of populating a scikit-learn model.

### Instructor

Geoffrey Boushey is a computational and research data science specialist at UCSF and a member of the Library's Data Science and Open Scholarship team. Geoff teaches courses and provides consultations on foundational programming, data analysis, and AI techniques including Python, SQL, Unix, and Version Control.

## Data Citation

This workshop uses data from the Cornell Movie Review dataset:

http://www.cs.cornell.edu/people/pabo/movie-review-data/

This workshop will use the polarity dataset v1.1:

http://www.cs.cornell.edu/people/pabo/movie-review-data/mix20_rand700_tokens_0211.tar.gz

## Workbooks

This repository contains a series of notebooks that introduce document preparation, vectorization, classification, clustering, model evaluation, and application of these techniques to larger text collections.

### [Sentiment-Example.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Sentiment-Example.ipynb)

Introduces document classification using a very small set of movie reviews. The notebook creates a bag-of-words representation with `CountVectorizer`, trains a Random Forest classifier, examines feature importance, and uses the model to predict the sentiment of new reviews.

### [Stem-Lemma-Example.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Stem-Lemma-Example.ipynb)

Demonstrates the differences between stemming and lemmatization using NLTK's `PorterStemmer` and `WordNetLemmatizer`, including examples of how each approach handles different forms of the same word.

### [Example-Prepare-Document.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Example-Prepare-Document.ipynb)

Walks through preparing raw text for document analysis. Starting with a sample document, the notebook demonstrates vectorization, removal of non-alphabetic characters, normalization to lowercase, stop-word removal, stemming, and lemmatization.

### [Sentiment-Pipeline-Example.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Sentiment-Pipeline-Example.ipynb)

Expands on `Sentiment-Example.ipynb` by combining document vectorization, TF-IDF transformation, and Random Forest classification into a scikit-learn `Pipeline`. The notebook also examines feature importance, prediction probabilities, precision, recall, and accuracy.

### [Movie-Reviews-Pipeline.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Movie-Reviews-Pipeline.ipynb)

Applies the classification pipeline to the larger movie-review dataset. The notebook splits the data into training and testing sets, builds a TF-IDF and Random Forest pipeline, examines important features, generates predictions and probabilities, and evaluates the resulting classifier using precision, recall, accuracy, F-score, and cross-validation.

### [Create-Movie-Review-Dataframes.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Create-Movie-Review-Dataframes.ipynb)

Prepares the Cornell movie-review data used by the classification examples. The notebook reads positive and negative review files, assigns sentiment labels, removes non-alphabetic characters and stop words, lemmatizes the text, and creates a dataframe suitable for subsequent analysis.

### [Kmeans-Sample-Data.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Kmeans-Sample-Data.ipynb)

Demonstrates unsupervised document clustering using TF-IDF vectors and scikit-learn's `KMeans`. The notebook clusters a small set of movie-review examples and examines the terms associated with each cluster.

### [Advertising-Legal-Classification.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Advertising-Legal-Classification.ipynb)

Applies document classification to a dataset of computer-generated transcripts categorized as advertising or legal content. The notebook builds a Random Forest text-classification pipeline, examines feature importance, generates class probabilities, adjusts classification thresholds, and evaluates the model using precision, recall, accuracy, F-score, and cross-validation.

### [Washington-Post-Collection.ipynb](https://github.com/geoffswc/Document-Classification/blob/master/Washington-Post-Collection.ipynb)

Provides a short example of loading and inspecting a Washington Post opioid collection dataset, including its available fields and OCR text.

## Links and Further Reading

These workbooks barely scratch the surface of machine learning and scikit-learn. Fortunately, the documentation for scikit-learn is extensive, including a programming guide for using the library, discussion on general use, mathematical background, theory and algorithms, and examples. Once you've gotten familiar with the examples provided in this repository, one good way to get more familiar with machine learning is work through the scikit-learn documentation.

As an entry point to scikit-learn, you may want to read more about the libraries and methods used in this workshop.

- scikit-learn: https://scikit-learn.org/
- count vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- term frequency: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
- random forest classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
