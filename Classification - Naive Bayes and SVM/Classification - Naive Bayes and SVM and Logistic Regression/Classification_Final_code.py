#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import pandas as pd
import os
import glob
import math
import re
import time
import warnings
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')
#Set Random seed
np.random.seed(500)



#Read unclean just processed tweet
df = tweets = pd.read_excel("./healthcare30000.xlsx")
df = df[['TweetText','Polarity']]
stop_words = set(['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours', 'ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves'])

def processRow(row):
    tweet = row.lower()    #Lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)    #delete any url
    tweet = re.sub('@[^\s]+','',tweet) #delete any @Username
    tweet = re.sub('[\s]+', ' ', tweet)#Remove additional white spaces
    tweet = re.sub('[\n]+', ' ', tweet) #Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]', ' ', tweet) #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #Remove Digits
    tweet = re.sub(" \d+", '', tweet)
    tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet)
    tweet = tweet.replace(':)','')    #Remove :( or :)
    tweet = tweet.replace(':(','')
    tweet = tweet.strip('\'"')    #trim
    tweet = [word for word in tweet.split() if word not in stop_words and len(word) > 1]#Removes stopwords and single letter words
    return ''.join(str(e) + " " for e in tweet)
testing = np.array(list(df['TweetText'][:30000]))  
for x in range(0,testing.shape[0]):
    testing[x] = processRow(testing[x])

#print(df.TweetText = testing)

### Write Pre-processed with stopwords Data into CSV file (uncomment to write)
###df.to_csv("clean_stopwordsremoved_healthcaretweet30000.csv", index = False)
def calPerformanceofModels(path,label,max_features):
    # Add the Data using pandas
    start = time.time()
    Corpus = pd.read_csv(path,encoding='latin-1')
    Corpus['Polarity'] = Corpus['Polarity'].apply(str) #converts the float string into string/obj for processing
    Corpus.dropna()
    #print(Corpus.shape)
    # Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['TweetText']= [word_tokenize(str(entry)) for entry in Corpus['TweetText']]
    # Step - 1b: Perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['TweetText']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
    #print(Corpus['text_final'].head())
    end = time.time()
    tok_time = end-start
    
    # Split the model into Train and Test Data set
    start  = time.time()
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Polarity'],test_size=0.2)
    # Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    # Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=max_features)
    Tfidf_vect.fit(Corpus['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    # Now we can run different algorithms to classify our data check for accuracy
    end  = time.time()
    vect_time = end-start
    # Classifier - Algorithm - Naive Bayes
    # fit the training dataset on the classifier
    start  = time.time()
    
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    predictions_NB = Naive.predict(Test_X_Tfidf) # predict the labels on validation dataset
    
    end  = time.time()
    nb_time = end -start
    #NAIVE BAYES END
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    start  = time.time()
    
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)    # predict the labels on validation dataset
    
    end  = time.time()  
    svm_time = end -start
    #SVM END
    
    # Classifier - Algorithm - Logistic Regression
    # fit the training dataset on the classifier
    start  = time.time() 
    
    LogReg = LogisticRegression()
    LogReg.fit(Train_X_Tfidf,Train_Y)
    predictions_LR = LogReg.predict(Test_X_Tfidf) # predict the labels on validation dataset
    
    end = time.time()
    lr_time = end -start
    #lOGISTIC REGRESSION END

    #We compute the precison, recall and fscore of each of the models
    #We use Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for
    #label imbalance; it can result in an F-score that is not between precision and recall.

    prf_NB = precision_recall_fscore_support(predictions_NB, Test_Y,average='weighted')
    prf_SVM = precision_recall_fscore_support(predictions_SVM, Test_Y,average='weighted')
    prf_LR = precision_recall_fscore_support(predictions_LR, Test_Y,average='weighted')
    print("~~~~For N labels = ", label , ", Max features = ", max_features  ,"~~~~~\n")
    print("===Naive Bayes===\nPrecision, Recall, F1-Score: ",  prf_NB[0]*100,prf_NB[1]*100,prf_NB[2]*100 )
    print("Naive Bayes Accuracy Score -> ",accuracy_score(Test_Y, predictions_NB)*100)
    print("===SVM===\nPrecision, Recall, F1-Score: ",  prf_SVM[0]*100,prf_SVM[1]*100,prf_SVM[2]*100)
    print("SVM Accuracy Score -> ",accuracy_score(Test_Y, predictions_SVM)*100)
    print("===Logistic Regression===\nPrecision, Recall, F1-Score: ",  prf_LR[0]*100,prf_LR[1]*100,prf_LR[2]*100)
    print("LR Accuracy Score -> ",accuracy_score(Test_Y,predictions_LR )*100)
    print("Time spent on tokenizing for bag of words: ", tok_time, "\n")
    print("Time spent on vectorizing for NB/SVM/LR", vect_time, "\n")
    print("Time spent on predicting for NB/SVM/LR models (respectively): ", "\n",
          nb_time, "/", svm_time ,"/", lr_time,"\n")
    
##Split Data and Vectorize(Train and Split Method)
def data(path,label,max_features):
    # Add the Data using pandas
    Corpus = pd.read_csv(path,encoding='latin-1')
    Corpus['Polarity'] = Corpus['Polarity'].apply(str) #converts the float string into string/obj for processing
    Corpus.dropna()
    #print(Corpus.shape)
    # Step - 1a : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['TweetText']= [word_tokenize(str(entry)) for entry in Corpus['TweetText']]
    # Step - 1b: Perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['TweetText']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
    #print(Corpus['text_final'].head())
    # Step - 2: Split the model into Train and Test Data set
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Polarity'],test_size=0.2)

    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=max_features)
    Tfidf_vect.fit(Corpus['text_final'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    return Train_X_Tfidf, Test_X_Tfidf, Train_Y,Test_Y
# Use Stratify method to split data and vectorize (Used to compared with train split method)

def stratifyData(path,label,max_features):
    # Add the Data using pandas

    Corpus = pd.read_csv(path,encoding='latin-1')
    Corpus['Polarity'] = Corpus['Polarity'].apply(str) #converts the float string into string/obj for processing
    Corpus.dropna()
    #print(Corpus.shape)
    # Step - 1a : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['TweetText']= [word_tokenize(str(entry)) for entry in Corpus['TweetText']]
    # Step - 1b: Perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['TweetText']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
    #print(Corpus['text_final'].head())
    
    # Step - 2: Split the model into Train and Test Data set
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    # X is the feature set and y is the target
    for train_index, test_index in skf.split(X,y): 
        print("Train:", train_index, "Validation:", val_index) 
        Train_X, Test_X = X[train_index], X[val_index] 
        Train_Y, Test_Y = y[train_index], y[val_index]
    #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Polarity'],test_size=0.2)

    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    Tfidf_vect = TfidfVectorizer(max_features=max_features)
    Tfidf_vect.fit(Corpus['text_final'])

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    return Train_X_Tfidf, Test_X_Tfidf, Train_Y,Test_Y    

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = ',   accuracy_score(test_labels, predictions)*100)
    
    return accuracy

# Used to Test model performance against ensembles using maxvoting method
def maxVotingmodels(x_train, x_test, y_train, y_test ):
    #Models Used
    Naive = naive_bayes.MultinomialNB()
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier( random_state=1)
    model3 = RandomForestClassifier(n_estimators=500, random_state=1)
    model4 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model5 = KNeighborsClassifier(n_neighbors=7)# KNN
   
    #knn
    #boosted tree
    #random forest
    
    #Voting classifiers, num indicates num of model inside
    ens2 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),], voting='hard')
    ens3 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive)], voting='hard')
    ens4 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),], voting='hard')
    ens5 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),('svm', model4)], voting='hard')
    ens6 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),('svm', model4), ('knn', model5)], voting='hard')
 
    Naive.fit(x_train,y_train)
    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)
    model4.fit(x_train,y_train)
    model5.fit(x_train,y_train)
    ens6.fit(x_train,y_train)
    
    print("6 model and all ensemble: Accuracy")
    for clf, label in zip([Naive, model1, model2,model3,model4,model5, ens2,ens3, ens4, ens5, ens6], ['Naive Bayes', 'Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Machine', 'K-Nearest Neighbour', 'Ensemble2', 'Ensemble3', 'Ensemble4', 'Ensemble5', 'Ensemble6']):
        scores = cross_val_score(clf, x_test, y_test, scoring='accuracy', cv=5)
        print("Accuracy: %0.6f (+/- %0.6f) [%s]" % (scores.mean(), scores.std(), label))

def maxVotingmodels2(x_train, x_test, y_train, y_test ):
    #Models Used
    Naive = naive_bayes.MultinomialNB()
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier( random_state=1)
    model3 = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, min_samples_leaf =3,
                                   max_features= 'auto',max_depth=90, bootstrap= True)
    model4 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model5 = KNeighborsClassifier(n_neighbors=7)# KNN
   
    #knn
    #boosted tree
    #random forest
    
    #Voting classifiers, num indicates num of model inside
    ens2 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),], voting='hard')
    ens3 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive)], voting='hard')
    ens4 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),], voting='hard')
    ens5 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),('svm', model4)], voting='hard')
    ens6 = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('nb', Naive), ('rf', model3),('svm', model4), ('knn', model5)], voting='hard')
 
    Naive.fit(x_train,y_train)
    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)
    model4.fit(x_train,y_train)
    model5.fit(x_train,y_train)
    ens6.fit(x_train,y_train)
    
    print("6 model and all ensemble: Accuracy")
    for clf, label in zip([Naive, model1, model2,model3,model4,model5, ens2,ens3, ens4, ens5, ens6], ['Naive Bayes', 'Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Machine', 'K-Nearest Neighbour', 'Ensemble2', 'Ensemble3', 'Ensemble4', 'Ensemble5', 'Ensemble6']):
        scores = cross_val_score(clf, x_test, y_test, scoring='accuracy', cv=5)
        print("Accuracy: %0.6f (+/- %0.6f) [%s]" % (scores.mean(), scores.std(), label))


##Hyper parameter tuning



##Qn 4.
###################################

##TESTS DONE: (Uncomment Sections to run)
#Start Test 1
calPerformanceofModels('./clean_healthcaretweet750.csv',750,5000)
#calPerformanceofModels('./clean_healthcaretweet1500.csv',1500,5000)
#calPerformanceofModels('./clean_healthcaretweet2250.csv',2250,5000)
#calPerformanceofModels('./clean_healthcaretweet3000.csv',3000,5000)
#END OF TEST

#Start Test 3
#calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,5000)
#calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,5000)
#calPerformanceofModels('./clean_healthcaretweet2250_30k.csv',2250,5000)
#calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,5000)
#END OF TEST

#Start Test 2
##750 labelled data, change MF 
#calPerformanceofModels('./clean_healthcaretweet750.csv',750, 5000)
#calPerformanceofModels('./clean_healthcaretweet750.csv',750, 2500)
#calPerformanceofModels('./clean_healthcaretweet750.csv',750, 7500)
#calPerformanceofModels('./clean_healthcaretweet750.csv',750, 10000)
##1500 labelled data, change MF 
#calPerformanceofModels('./clean_healthcaretweet1500.csv',1500, 5000)
#calPerformanceofModels('./clean_healthcaretweet1500.csv',1500, 2500)
#calPerformanceofModels('./clean_healthcaretweet1500.csv',1500, 7500)
#calPerformanceofModels('./clean_healthcaretweet1500.csv',1500, 10000)
##2250 labelled data, change MF 
#calPerformanceofModels('./clean_healthcaretweet2250.csv',2250, 5000)
#calPerformanceofModels('./clean_healthcaretweet2250.csv',2250, 2500)
#calPerformanceofModels('./clean_healthcaretweet2250.csv',2250, 7500)
#calPerformanceofModels('./clean_healthcaretweet2250.csv',2250, 10000)
##3000 labelled data, change MF 
#calPerformanceofModels('./clean_healthcaretweet3000.csv',3000, 5000)
#calPerformanceofModels('./clean_healthcaretweet3000.csv',3000, 2500)
#calPerformanceofModels('./clean_healthcaretweet3000.csv',3000, 7500)
#calPerformanceofModels('./clean_healthcaretweet3000.csv',3000, 10000)
#END OF TEST

#Start Test 4
# #750 labelled data, change MF 
#calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,2500)
# calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,5000)
# calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,7500)
# calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,10000)
# #1500 labelled data, change MF 
# calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,2500)
# calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,5000)
# calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,7500)
# calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,10000)
# #2250 labelled data, change MF 
# ## Validate Results using clean_healthcaretweet1500 dataset
# calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,2500)
# calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,5000)
# calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,7500)
# calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,10000)
# #3000 labelled data, change MF 

#END OF TEST
#Start Test 5
# #750
# print("Before-----------------\n")
#calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,5000)
# print("After-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_30k.csv',750,5000)
# #1500
# print("Before-----------------\n")
# calPerformanceofModels('./clean_healthcaretweet1500_30k.csv',1500,5000)
# print("After-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet1500_30k.csv',1500,5000)
# #3000
# print("Before-----------------\n")
# calPerformanceofModels('./clean_healthcaretweet3000_30k.csv',3000,5000)
# print("After-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet3000_30k.csv',3000,5000)
#END OF TEST

## Scaling Tests (uncomment to run)
## Removing Stopwords
#print("15k with stopwords-----------------\n")
#calPerformanceofModels('./clean_healthcaretweet750.csv',750,5000)
# print("15k without stopwords-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_15k.csv',750,5000)
# print("30k with stopwords-----------------\n")
# calPerformanceofModels('./clean_healthcaretweet750_30k.csv',750,5000)
# print("30k without stopwords-----------------\n")
# print("30k-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_30k.csv',750,5000)

#Scaling, Increasing MF
#print("15k without stopwords MF increased-----------------\n")
#calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_15k.csv',750,5000)
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_15k.csv',750,7500)
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_15k.csv',750,10000)
# print("30k without stopwords MF increased-----------------\n")
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_30k.csv',750,5000)
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_30k.csv',750,7500)
# calPerformanceofModels('./clean_stopwordsremoved_healthcaretweet750_30k.csv',750,10000)

#Qn5
###################################################
#Split Train Test
#x_train, x_test, y_train, y_test = data('clean_healthcaretweet3000_30k.csv', 750,5000)
#maxVotingmodels(x_train, x_test, y_train, y_test)
# #Stratified K-fold
# x_train1, x_test1, y_train1, y_test1 = data('clean_healthcaretweet3000_30k.csv', 750,5000)
# maxVotingmodels(x_train1, x_test1, y_train1, y_test1)

#5bi
##Hyper Parameter Tuning (uncomment to tune will take around an hour)
#Start of primary tuning:Random Search CV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, 
#                                random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(x_train1, y_train1)
# # Best Parameters for Random Forest
# print(rf_random.best_params_)

######SET BEST PARAM FOR PRIMARY TUNING AND TEST IMPROVEMENT
# rf = RandomForestClassifier(n_estimators = 2000, min_samples_split = 2, min_samples_leaf =2 ,
#                                    max_features= 'auto',max_depth=90, bootstrap= True)
# rf.fit(x_train1, y_train1)
# base_accuracy = evaluate(rf, x_test1, y_test1)
# # End of primary tuning


#5bii

## Secondary/Final Tuning: GridSearch CV

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': ['auto'],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [3, 4, 5],
#     'n_estimators': [100, 200, 300, 1000, 1500, 2250, 2500 ,3000]
# }

# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1,
#                            verbose = 2)
# grid_search.fit(x_train1 ,y_train1)
# print(grid_search.best_params_)
# rf = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, min_samples_leaf =3,
#                                    max_features= 'auto',max_depth=90, bootstrap= True)
# rf.fit(x_train1, y_train1)
# base_accuracy = evaluate(rf, x_test1, y_test1)
# maxVotingmodels2(x_train1, x_test1, y_train1, y_test1)


# In[ ]:




