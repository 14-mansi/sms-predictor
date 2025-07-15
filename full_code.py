# Libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter

df = pd.read_csv('spam.csv') # Libraries required
df.head() # Libraries required
df.shape

# 1. Data Cleaning

df.info() # Checking info

# no need for last three columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
# since it is a permanent action so inplace is true
df.head()

# v1 and v2 are confusing labels so we rename it 
df.rename(columns={'v1':'predict', 'v2':'sms'}, inplace = True)
df.head()

# It is generally recommended that target variables (labels) should be encoded with numerical values in classification problems, where the numerical labels do not imply any ordinal relationship.
encoder = LabelEncoder()
df['predict'] = encoder.fit_transform(df['predict'])
df.head()

# Checking null values
df.isnull().sum()

# Checking duplicate values
df.duplicated().sum()

# Since there are duplicate values so drop them
df = df.drop_duplicates(keep = 'first')
df.duplicated().sum()
df.shape

# 2. EDA

# Number of spam and ham sms
df['predict'].value_counts()

# Graph for representing percent of spa and ham sms
plt.pie(df['predict'].value_counts(), labels = ['ham', 'spam'], autopct = "%0.2f")
plt.show()

# Adding new columns for number of characters, words and sentences in each input
df = df.assign(num_char = df['sms'].apply(len))
df = df.assign(num_words = df['sms'].apply(lambda x:len(nltk.word_tokenize(x))))
df = df.assign(num_sen = df['sms'].apply(lambda x:len(nltk.sent_tokenize(x))))
df.head()

# Analysis of all sms
df[['num_char', 'num_words', 'num_sen']].describe()

#ham messages analysis
df[df['predict'] == 0][['num_char', 'num_words', 'num_sen']].describe()

#spam messages analysis
df[df['predict'] == 1][['num_char', 'num_words', 'num_sen']].describe()

# Analysis for number of characters used in ham and spam sms
plt.figure(figsize = (12, 6))
sns.histplot(df[df['predict'] == 0]['num_char'])
sns.histplot(df[df['predict'] == 1]['num_char'], color = 'red')

# Analysis for number of words used in ham and spam sms
plt.figure(figsize = (12, 6))
sns.histplot(df[df['predict'] == 0]['num_words'])
sns.histplot(df[df['predict'] == 1]['num_words'], color = 'red')

# 3. Data Processing

# stopwords are considered to have little impact on the meaning of the text
stopwords.words('english')

string.punctuation

# There are many words which have same meaning so reduce them to their base form
ps = PorterStemmer()
ps.stem("your'll")

# stopwords and punctuations are removed from the sms for better analysis
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

# Adding a new column with transformed text
df = df.assign(transformed_text = df['sms'].apply(transform_text))
df.head()

# array containing words commonly used in spam sms
spam_words = []
for msg in df[df['predict'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_words.append(word)

len(spam_words)

# Representing usage of common words in spam sms
sns.barplot(x=pd.DataFrame(Counter(spam_words).most_common(30))[0],y=pd.DataFrame(Counter(spam_words).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

# array containing words commonly used in ham sms
ham_words = []
for msg in df[df['predict'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_words.append(word)

len(ham_words)

# Representing usage of common words in ham sms
sns.barplot(x=pd.DataFrame(Counter(ham_words).most_common(30))[0],y=pd.DataFrame(Counter(ham_words).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()

# 4. Model Building

# Creating an object
tfidf = TfidfVectorizer()

# #spliitng the data into x and y  features
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['predict'].values

# using train_test_split to split the dataset into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Creating an object for model
mnb = MultinomialNB()

# Training the model
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

# Prediction of new sms
def predict_sms(text):
    text1 = transform_text(text)
    text2 = tfidf.transform([text1])
    prediction = mnb.predict(text2)[0]
    if (prediction==0):
        print('Ham sms')

    else:
        print('Spam sms')

new_sms = input("Enter a new SMS to classify: ")
predict_sms(new_sms)

