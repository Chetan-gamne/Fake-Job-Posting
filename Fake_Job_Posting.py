#!/usr/bin/env python
# coding: utf-8

# In[52]:


import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# In[12]:


df = pd.read_csv('./fake_job_data.csv')
df


# In[13]:


df.head()


# In[14]:


df.tail()


# In[15]:


df.shape


# In[16]:


df.describe()


# In[17]:


df.isnull().sum() #using to check null values in dataset.


# In[18]:


df = df.drop(["job_id","telecommuting","has_company_logo","has_questions","salary_range","employment_type"],axis=1)


# In[19]:


df.head()


# In[20]:


df.fillna("",inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


plt.figure(figsize=(10,5))
sns.countplot(y='fraudulent',data=df)
plt.show()


# In[23]:


df.groupby('fraudulent')['fraudulent'].count()


# In[24]:


exp = dict(df.required_experience.value_counts())
exp
del exp['']
exp


# In[25]:


plt.figure(figsize=(10,5))
plt.title("Jobs by Experience",size=25)
plt.bar(exp.keys(),exp.values())


# In[26]:


df


# In[27]:


def split(location):
    loc = location.split(',')
    return loc[0];
df['country'] = df.location.apply(split)


# In[28]:


df.head()


# In[29]:


country = dict(df.country.value_counts()[:10])
del country[''] #Deleting Blank value countries from dictionaries
country


# In[30]:


plt.bar(country.keys(),country.values())
plt.title("Country-wise Job Postings")


# In[31]:


edu = dict(df.required_education.value_counts()[:6])
del edu[''] #Deleting Blank value edu from dictionaries
edu


# In[32]:


plt.figure(figsize=(15,7))
plt.title('Jobs posting by education',size=25)
plt.bar(edu.keys(),edu.values(),color="lightblue")
plt.xlabel("Education",size=15)
plt.ylabel('Jobs',size=15)


# In[33]:


print(df[df.fraudulent==0].title.value_counts()[:20]) #Genuine jobs postings comes usually with this titles


# In[34]:


print(df[df.fraudulent==1].title.value_counts()[:20]) #Fraudulent jobs postings comes usually with this titles


# In[35]:


df['text'] = df['title']+' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']


# In[36]:



df2 = df.copy()


# In[37]:


del df['title']
del df['location']
del df['department']
del df["company_profile"]
del df["description"]
del df["requirements"]
del df["benefits"]
del df["required_experience"]
del df["required_education"]
del df["industry"]
del df["function"]
del df["country"]


# In[38]:


df.head()


# In[39]:


df.tail()


# In[40]:


df.shape


# In[41]:


fraudjobs_text = df[df.fraudulent == 1].text # 1 for fraud jobs
realjobs_text = df[df.fraudulent == 0].text # 0 for real jobs


# In[42]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(20,15))
wc = WordCloud(min_font_size = 3,max_words = 2000, width = 1600, height = 800, stopwords=STOPWORDS).generate(str("".join(fraudjobs_text)))
plt.imshow(wc,interpolation = "bilinear")  #fraud jobs keywords 


# In[43]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(20,15))
wc = WordCloud(min_font_size = 3,max_words = 2000, width = 1600, height = 800, stopwords=STOPWORDS).generate(str("".join(realjobs_text)))
plt.imshow(wc,interpolation = "bilinear")  #real job keywords


# In[44]:


get_ipython().system('pip install spacy && python -m spacy download en')


# In[53]:


punctuations = string.punctuation

nlp  = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser  = English()

def spacy_tokenizer(sentence):
    mytoken = parser(sentence)
    mytokens = [word.lemma.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

class predictors(TransformerMixin):
    def transform(self,X,**transform_params):
        return [clean_text(text) for text in X]
    
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self,deep = True):
        return {}
    
def clean_text(text):
    return text.strip().lower()


# In[54]:


df['text'] = df['text'].apply(clean_text)


# In[58]:


cv = TfidfVectorizer(max_features=100)
x = cv.fit_transform(df['text'])
df1 = pd.DataFrame(x.toarray(),columns = cv.get_feature_names())
df.drop(["text"],axis=1,inplace=True)
main_df = pd.concat([df1,df],axis=1)


# In[59]:


main_df.head()


# In[72]:


Y = main_df.iloc[:,-1]
X = main_df.iloc[:,:-1]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import svm


# In[86]:


lg_model = LogisticRegression(random_state=0).fit(x_train,y_train)
dtc_model = DecisionTreeClassifier(random_state=0).fit(x_train,y_train)
rfc_model = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy").fit(x_train,y_train)
svm_model = svm.SVC().fit(x_train,y_train)
knn_model = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)


# In[88]:


lg_pred  = lg_model.predict(x_test)
dtc_pred = dtc_model.predict(x_test)
rfc_pred = rfc_model.predict(x_test)
svm_pred = svm_model.predict(x_test)
knn_pred = knn_model.predict(x_test)


# In[102]:


target_names = ["Real-0","Fake-1"]
lg_score = accuracy_score(y_test,lg_pred)
print("Logistic Regression Accuracy: " , lg_score*100)
print("Confusion Matrix: " ,confusion_matrix(y_test,lg_pred)) # For Logistic Regression Confusion Matrix
print(classification_report(y_test, lg_pred,target_names = target_names))


# In[106]:


target_names = ["Real-0","Fake-1"]
dtc_score = accuracy_score(y_test,dtc_pred)
print("Decision Tree Accuracy: " , dtc_score*100)
print("Confusion Matrix: " ,confusion_matrix(y_test,dtc_pred)) # For Decision Tree Confusion Matrix
print(classification_report(y_test, dtc_pred,target_names = target_names))


# In[107]:


target_names = ["Real-0","Fake-1"]
rfc_score = accuracy_score(y_test,rfc_pred)
print("Random Forest Classifier Accuracy: " , rfc_score*100)
print("Confusion Matrix: " ,confusion_matrix(y_test,rfc_pred)) # For Random Forest Classifier Confusion Matrix
print(classification_report(y_test, rfc_pred,target_names = target_names))


# In[108]:


target_names = ["Real-0","Fake-1"]
svm_score = accuracy_score(y_test,svm_pred)
print("Support Vector Machines (SVM) Accuracy: " , svm_score*100)
print("Confusion Matrix: " ,confusion_matrix(y_test,svm_pred)) # For Support Vector Machines Confusion Matrix
print(classification_report(y_test, svm_pred,target_names = target_names))


# In[109]:


target_names = ["Real-0","Fake-1"]
knn_score = accuracy_score(y_test,knn_pred)
print("K-Nearest Neighbor (KNN) Accuracy: " , knn_score*100)
print("Confusion Matrix: " ,confusion_matrix(y_test,knn_pred)) # For K-Nearest Neighbor (KNN) Confusion Matrix
print(classification_report(y_test, knn_pred,target_names = target_names))


# In[110]:


print("Logistic Regression Accracy: " , lg_score*100)
print("Decision Tree Accracy: " , dtc_score*100)
print("Random Forest Accracy: " , rfc_score*100)
print("Support Vector Machines (SVM) Accracy: " , svm_score*100)
print("KNN Accracy: " , knn_score*100) 


# ### KNN Algorithm gives us Higher Prediction which is almost 97.42% 
# 
# #### So, model train by KNN Algorithm is the best model for this dataset.

# 
