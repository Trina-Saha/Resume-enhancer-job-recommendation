#!/usr/bin/env python
# coding: utf-8

# # **Skill Prediction**

# In[38]:


import joblib
from ast import literal_eval
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import nltk
from nltk.stem import PorterStemmer
import string


# ### Function to preprocess skills

# In[39]:


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
    
    return preprocessed_text#returns string

# Preprocessing function for skills column
def preprocess_skills(skill_entry):
    # Convert list to string
    if isinstance(skill_entry, list):
        skill_entry = ' '.join(skill_entry)#converting list to string
    
    # Preprocess the text
    skill_entry = preprocess_text(skill_entry)
    
    return skill_entry#string


# In[40]:


#text= "I am good in html css e commerce platforms(shopify)"
#text= preprocess_skills(text)


# In[41]:


#text


# In[42]:


'''def search_word_in_skills(unique_skills, word):
    # Iterate over each skill in the unique_skills list
    for skill in unique_skills:
        # Check if the word exists in the skill
        if word.lower() in skill.lower():
            return True
    
    return False
search_word_in_skills(unique_skills, 'amazon ec2')'''


# ## Function to return all skills of the predicted cluster

# In[43]:


def get_cluster_skills(kmeans_model, skills_df, input_skill, tfidf_vectorizer):
    # Preprocess the input skill
    preprocessed_input_skill = preprocess_skills(input_skill)#input is list/ string output is string
    
    # Vectorize the input skill
    input_skill_vectorized = tfidf_vectorizer.transform([preprocessed_input_skill])
    
    # Predict the cluster for the input skill
    predicted_cluster = kmeans_model.predict(input_skill_vectorized)[0]
    
    # Get indices of skills in the predicted cluster
    cluster_indices = np.where(kmeans_model.labels_ == predicted_cluster)[0]
    
    # Extract skills associated with the predicted cluster
    cluster_skills = []
    for idx in cluster_indices:
        if isinstance(skills_df.iloc[idx], str):
            skills_list = skills_df.iloc[idx].split(', ')
        elif isinstance(skills_df.iloc[idx], list):
            skills_list = skills_df.iloc[idx]
        else:
            continue
        cluster_skills.append(skills_list)
    
    return cluster_skills


# ## Function to extract top 10 skills from that cluster

# In[44]:


def get_top_skills(cluster_skills):
    # Flatten the list of lists of skills into a single list
    all_skills = [skill for sublist in cluster_skills for skill in sublist]
    
    # Count the frequency of each skill
    skill_counts = Counter(all_skills)
    
    # Get the top 10 most common skills (without frequency)
    top_skills = [skill for skill, _ in skill_counts.most_common(10)]
    
    return top_skills


# ## Function to filter the top skills

# In[45]:


# Function to preprocess a skill and convert it to its root form
def preprocess_skill(skill):
    stemmer = PorterStemmer()
    # Remove punctuation and split into individual words
    skill_tokens = nltk.word_tokenize(skill.translate(str.maketrans('', '', string.punctuation)))
    # Stem each token and join them back
    root_tokens = [stemmer.stem(token.lower()) for token in skill_tokens]
    return ' '.join(root_tokens)

# Function to filter recommended skills based on user's skills
def filter_recommended_skills(user_skills, recommended_skills):
    filtered_skills = {'have': [], 'not_have': []}
    # Dictionary to map preprocessed skill to its original form
    skill_mapping = {}
    
    # Preprocess user skills
    preprocessed_user_skills = [preprocess_skill(skill) for skill in user_skills]
    
    # Preprocess recommended skills and filter
    for skill in recommended_skills:
        preprocessed_skill = preprocess_skill(skill)
        # Check if any word in the preprocessed user skills matches any word in the preprocessed recommended skill
        if not any(preprocessed_word in preprocessed_skill.split() for preprocessed_word in preprocessed_user_skills):
            filtered_skills['not_have'].append(skill)
        else:
            filtered_skills['have'].append(skill)
        
        # Map preprocessed skill to its original form
        skill_mapping[preprocessed_skill] = skill
    
    return filtered_skills, skill_mapping


# ## To return only unique skills and not redundant skills

# In[46]:


from nltk.stem import PorterStemmer

def preprocess_skill(skill):
    # Tokenize and stem the skill
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token.lower()) for token in skill.split()]
    return ' '.join(tokens)

def get_unique_skills(filtered_skills):
    # Create a mapping of reduced words to original skills
    reduced_to_original = {}
    for skill in filtered_skills:
        reduced_skill = preprocess_skill(skill)
        if reduced_skill not in reduced_to_original:
            reduced_to_original[reduced_skill] = skill
    
    # Filter unique reduced skills
    unique_reduced_skills = set(reduced_to_original.keys())
    
    # Check for substrings and remove them
    skills_to_remove = set()
    for skill in unique_reduced_skills:
        for other_skill in unique_reduced_skills:
            if skill != other_skill and skill in other_skill:
                skills_to_remove.add(other_skill)
    
    # Remove substrings
    unique_reduced_skills -= skills_to_remove
    
    # Convert reduced skills back to original mapping
    final_skills = [reduced_to_original[reduced_skill] for reduced_skill in unique_reduced_skills]
    
    return final_skills


# In[47]:


def skills_prediction(resume_skills, jd_skills, role):

    # Create the dictionary
    role_acronym_dict = {
        'Accountant': 'acc',
        'HR': 'hr',
        'Nurse': 'nur',
        'Sales': 'sal',
        'Business development Analyst': 'bda',
        'Therapist': 'th',
        'Web Designer': 'wd',
        'Database Administrator': 'dba',
        'Software Developer': 'sde',
        'Data Scientist': 'ds'
    }
    # Fetch the acronym based on the role
    acr = role_acronym_dict.get(role)
    
    # Form the file name using string formatting
    vector_name = f"tfidf_vectorizer_{acr}.pkl"
    model_name = f"kmeans_model_{acr}.pkl"
    skills_df_name = f"{acr}_skills_df.pkl"
    #Loading the respective pickle files
    tfidf_vectorizer = joblib.load(vector_name)
    kmeans_model = joblib.load(model_name)
    skills_df = joblib.load(skills_df_name)
    cluster_skills = get_cluster_skills(kmeans_model, skills_df, jd_skills, tfidf_vectorizer)
    top_skills = get_top_skills(cluster_skills)
    filtered_skills, skill_mapping = filter_recommended_skills(resume_skills, top_skills)
    final_skills = get_unique_skills(filtered_skills['not_have'])
    return filtered_skills['have'], final_skills
