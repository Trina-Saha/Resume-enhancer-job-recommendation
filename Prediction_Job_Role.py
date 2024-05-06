#!/usr/bin/env python
# coding: utf-8

# # **Making Predictions of job role on different inputs**

# In[26]:


import joblib
import pandas as pd
from ast import literal_eval
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[27]:


#get_ipython().system('pip install joblib')


# ## Resume Skill Preprocessing

# In[28]:


# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    tokens = [token for token in tokens if token.isalnum() and token not in stopwords.words('english')]
    
    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Preprocessing function for skills column
def preprocess_skills(skill_entry):
    # Convert list to string
    if isinstance(skill_entry, list):
        skill_entry = ' '.join(skill_entry)
    
    # Preprocess the text
    skill_entry = preprocess_text(skill_entry)
    
    return skill_entry


# # Starting function

# In[29]:


def prediction(skills_resume, subject_encoded, degree_encoded, experience_encoded):
    # Load the tuned RFC model
    loaded_rfc_tuned = joblib.load('tuned_rfc_model.pkl')
    Role_encoding= {'Accountant': 0, 'Business development Analyst': 1, 'Data Scientist': 2, 'Database Administrator': 3, 'HR': 4, 'Nurse': 5, 'Sales': 6, 'Software Developer': 7, 'Therapist': 8, 'Web Designer': 9}
    Role_encoding_rev = {v: k for k, v in Role_encoding.items()}
    skills_resume= preprocess_skills(skills_resume)
    resume_data = pd.DataFrame({
   'Experience': [experience_encoded],
   'Degree_encoded': [degree_encoded],
   'Subject_encoded': [subject_encoded],
   'skills': [skills_resume]})
    # Load the TF-IDF vectorizer instance
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Transform skills data from the input resume using the existing TF-IDF vectorizer
    resume_tfidf_features = tfidf_vectorizer.transform(resume_data['skills'])
    
    # Convert TF-IDF features to DataFrame
    resume_tfidf_df = pd.DataFrame(resume_tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Add TF-IDF features to the resume_data DataFrame
    resume_data_with_tfidf = pd.concat([resume_data.drop('skills', axis=1), resume_tfidf_df], axis=1)
    
    # Make predictions using the trained RFC model
    predicted_role = loaded_rfc_tuned.predict(resume_data_with_tfidf)
    role = Role_encoding_rev.get(predicted_role[0],'role does not exist' )
    return role
