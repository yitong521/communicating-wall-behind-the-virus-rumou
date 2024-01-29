import csv
import codecs
import requests 
from urllib.request import Request,urlopen
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords 
from nltk import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
#Step1: Get data by using the method of web scraping
### Get and save  English data
# Scrape it from the rumour database and save it into a file 
for i in range(1,424):      # 424 is the page number plus one 
    url = "https://www.poynter.org/ifcn-covid-19-misinformation/page/{}/".format(i)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    print(webpage, file=open("output.txt", "a"))

# filter out the useful information that I need 
with open('output.txt', 'r') as file:
    engdata = file.read().replace('\n', '')

soup2 = BeautifulSoup(engdata, features='lxml')
# filter out the title and the label, and save them into a list 
try1 = soup2.find_all('h2', class_='entry-title')
titletext2 = [r.text for r in try1]
converted_list = []
for element in titletext2:
    converted_list.append(element.strip("\\n\\t\\t"))
label, rumoureng  = zip(*(s.split(":") for s in converted_list ))
# filter out the time and space 
timespace = soup2.find_all('p', class_='entry-content__text')
timespacetext = []
for i in timespace:
    rt = i.findAll('strong')  
    for r in rt:
        time2 = r.text
        timespacetext.append(time2)
time, space = zip(*(s.split("|") for s in timespacetext ))

# save each the data into a csv file, with four lines: time, space, label, rumour in english. 
with open('rumourseng.csv', 'w', newline='') as csvfile:
    fieldnames = ['time','space','label', 'rumoureng']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, j,k,l in zip(time,space, label, rumoureng):
        writer.writerow({'time': i,'space':j ,'label':k,'rumoureng': l}) 
#Step 2 : Natural Language Processing        
###Get and save Chinese rumours 

html = urlopen("http://py.qianlong.com/2020/0502/3625023.shtml").read().decode('utf-8')
soup = BeautifulSoup(html, features='lxml')
allch1 = soup.findAll('p')[3:]
alltext1 = [r.text for r in allch1]

chineserumour = open("chineserumour.csv",'w',encoding='utf-8-sig')
for r in alltext1:   
    chineserumour.write(r + "\n")
chineserumour.close()
###Select training part and test part randomly. 
#  For the english data. I will divide them into three parts: training(60%),validating(20%) and testing(20%). 

alldata=pd.read_csv("rumourseng.csv", encoding= 'unicode_escape')

def split_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

traineng = split_train(alldata,0.2)[0]
testeng = split_train(alldata,0.2)[1]

traineng.to_csv('traineng.csv', index = False, header=True)
testeng.to_csv('testeng.csv', index = False, header=True)


### Then I code the training and validating part manually 
alltraindata = pd.read_excel('traineng.xlsx',index = False, header=None) 
testeng = pd.read_excel('testeng.xlsx',index = False, header=None) 
chidata = pd.read_excel('rumourchi.xlsx',index = False, header=None) 

# change the column names 
alltraindata.columns = ['factcheckingcountry', 'rate','date','rumour','label']
testeng.columns = ['factcheckingcountry', 'rate','date','rumour']
chidata.columns = ['chineserumour', 'label','date','rumour']

### natural language processing 

# as there are other words that frequently used in the data set such as "facebook" I will filter them out first 
#freq=pd.Series(' '.join(alltraindatapro['rumour']).split()).value_counts()[:30]
#freq=list(freq.index) # i filter out some of frequent word manually such as 'Italy','pandemic'
frequent = ['coronavirus', 'covid19', 'video', 'people', 'shows', 'new', 'novel', 'facebook', 'claim', 'shared', 'photo', 'claims', 'virus', 'due', 'says', 'posts']
stop = stopwords.words('english') + frequent

def preprocessing(data):
    data['rumour']=data['rumour'].apply(lambda sen:" ".join(x.lower() for x in sen.split()))
    data['rumour'] = data['rumour'].str.replace('[^\w\s]','')
    data['rumour']=data['rumour'].apply(lambda sen:" ".join(x for x in sen.split() if x not in stop))
    return data

alltraindatapro = preprocessing(alltraindata)
testengpro = preprocessing(testeng)
chidatapro = preprocessing(chidata)

# divid train dataset into train and validation part 
trainpart = split_train(alltraindata,0.25)[0]
valipart = split_train(alltraindata,0.25)[1]

#Step 3: Machine Learning
# Vectorizer
train_texts = trainpart['rumour'].tolist()
train_labels = trainpart['label'].tolist()
test_texts = valipart['rumour'].tolist()
test_labels = valipart['label'].tolist()

def mlselector( classifier,clasname ):
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',classifier)])
    text_clf=text_clf.fit(train_texts,train_labels)
    predicted=text_clf.predict(test_texts)
        
    return print( "Using ",clasname," The mean accuracy is : ",np.mean(predicted==test_labels))

# Select the best classifier 
mlselector(MultinomialNB(),'MultinomialNB')
mlselector(SGDClassifier(),'SGDClassifier')
mlselector(LogisticRegression(),'LogisticRegression')
mlselector(SVC(),'SVC')
mlselector(LinearSVC(),'LinearSVC')
mlselector(MLPClassifier(),'MLPClassifier')
mlselector(KNeighborsClassifier(),'KNeighborsClassifier')
mlselector(RandomForestClassifier(n_estimators=8),'RandomForestClassifier')
mlselector(GradientBoostingClassifier(),'GradientBoostingClassifier')
mlselector(AdaBoostClassifier(),'AdaBoostClassifier')
mlselector(DecisionTreeClassifier(),'DecisionTreeClassifier')

train_texts = alltraindatapro['rumour'].tolist()
train_labels = alltraindatapro['label'].tolist()
test_texts = testengpro['rumour'].tolist()

text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),('clf', LinearSVC() )])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)

testengpro['label'] = predicted
allengpro = pd.concat([ alltraindatapro,testengpro])

engplot = allengpro.groupby('date')['label'].value_counts().unstack().plot(legend=True, figsize=(20, 8))
legendlist = ["About celebrities",
           "Mortality and existing infected cases",
           "Policies to control the virus",
           "Forecasting the epidemic",
           "Related to international relations",
           "Basic protection knowledge",
           "Related home quarantine order",
           "Funny memes or environmental issues",
           "Related to economic policy"]
engplot.legend(legendlist)

#Step4 : Using LDA to check topics that find by the machine.

def ldamap(df):
    
    n_features = 1000 
    tf_vectorizer = CountVectorizer(strip_accents='unicode', 
        stop_words='english', 
        max_df=0.5, 
        min_df=10) 
    tf = tf_vectorizer.fit_transform(df)

    n_topics = 10 
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', max_iter=50, learning_offset=50., random_state=0 )
    lda.fit(tf)

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print('\nTopic Nr.%d:' % int(topic_idx + 1))
            print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2)) + ' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
    n_top_words = 20 
    tf_feature_names = tf_vectorizer.get_feature_names()
    #print_top_words(lda, tf_feature_names, n_top_words)

    pyLDAvis.enable_notebook()
    return pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

# for all the english rumours 
com1 = alltraindatapro.drop(columns=['label'])
frames = [com1, testengpro]
allengpro = pd.concat(frames)
dfeng = allengpro['rumour']
ldamap(dfeng)

# Step 5: compare common knowledge
# In the rumors, people summarized many ways to treat the virus and the cause of the virus. In these texts, nouns are the subject, and I extract all the nouns.

eng6 = allengpro.loc[allengpro['label']==6]
chi6 = chidatapro.loc[chidatapro['label']==6]
len(eng6) # 2006
len(chi6) #256

def nounprocess(data):
    m = data['rumour'].tolist()
    
    s = []
    for i in m: 
        p= i.split()
        s.extend(p) 
    
    my_lst_str = ' '.join(map(str,s))

    blob = TextBlob(my_lst_str)
    nounphrases = blob.noun_phrases
    b = Counter(nounphrases).most_common(60)
    
    
    stem = ' '.join(map(str,nounphrases))
    wordcloud = WordCloud(background_color="white",width=1000, height=860, margin=2).generate(stem)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    return b
