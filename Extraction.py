#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[14]:


import nltk
import math
import spacy
import random
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
from datetime import datetime
from word2number import w2n
import os
import pandas as pd
import numpy as np
import jsonlines
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import fitz
from spacy.matcher import Matcher


# In[280]:


#get_ipython().system('pip install jsonlines')


# In[281]:


#get_ipython().system('pip install word2number')


# In[282]:


#get_ipython().system('pip install pymupdf')


# In[283]:


#get_ipython().system('python -m spacy download en_core_web_lg')


# In[ ]:





# # **NLP and JSONL Initialization**

# In[66]:


def load_skills_entity_ruler():
    """
    Load skill patterns into SpaCy's entity ruler.
    """
    warnings.filterwarnings('ignore')
    nlp = spacy.load("en_core_web_lg")
    skill_pattern_path = "jz_skill_patterns2.jsonl"
    
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(skill_pattern_path)
    else:
        # If entity_ruler is already in the pipeline, you might want to update its patterns
        existing_ruler = nlp.get_pipe("entity_ruler")
        existing_ruler.from_disk(skill_pattern_path)

    return nlp



# # **Skill Extraction Function**

# In[67]:


def get_technical_skills(text):
    nlp = load_skills_entity_ruler()
    doc = nlp(text)
    technical_skills = set()
    for ent in doc.ents:
        #print(ent)
        for token in ent:
            if token.ent_type_ == "SKILL":
                technical_skills.add(ent.text)
                break  # Exit the loop once a skill is found in the entity

    return list(technical_skills)


# # **Clean Resume Function**

# In[68]:


def clean_data(resume):
  clean = []
  review = re.sub(
      '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
      " ",
      resume,
      )
  review = review.lower()
  review = review.split()
  lm = WordNetLemmatizer()
  review = [
      lm.lemmatize(word)
      for word in review
      if not word in set(stopwords.words("english"))
      ]
  review = " ".join(review)
  clean.append(review)
  return clean


# # **Resume Analysis**

# ### **Text Extraction**

# In[69]:


def extract_text_from_panes(pdf_path):
    text_combined = ""

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Extract text from the current page
        page_text = page.get_text()

        # Append extracted text from the current page to overall text
        text_combined += page_text.strip() + "\n"



    return text_combined.strip()


# # ***Extracting Years of Experience***

# ### **Function to extract date string patterns**

# In[70]:


import re

def extract_experience_strings(resume_text):
    # Regular expression pattern to match years and months of experience
    pattern = r"(\d+\s+(?:years?|yrs?)\s*\d*\s*months?)|\b(\d+\s*months?)\b|\b(\d+\s+(?:years?|yrs?)\b)"
    
    # Initialize list to store extracted experience strings
    experience_strings = []

    # Search for patterns in the resume text
    matches = re.findall(pattern, resume_text)
    
    # Add the entire match string to the experience list
    for match in matches:
        for group in match:
            if group:
                experience_strings.append(group.strip())
    
    return experience_strings






# ### **Function to convert each item of date into years**

# In[71]:


def convert_to_years(experience_string):
    # Split the experience string by whitespace
    parts = experience_string.split()
    years = 0
    
    # Iterate through the parts and convert to years
    for i, part in enumerate(parts):
        if part.isdigit():
            if parts[i+1] == "years" or parts[i+1] == "year":
                years += int(part)
            elif parts[i+1] == "months" or parts[i+1] == "month":
                years += int(part) / 12
                
    return years


# ### **Function to finally calculate total years of experience**

# In[72]:


def calculate_total_years(experience_strings):
    total_years = 0
    for experience_string in experience_strings:
        total_years += convert_to_years(experience_string)
        print(total_years)
    return int(math.ceil(total_years))


# ## Function to extract EDUCATION(degree and subject)

# In[73]:


def extract_education_degrees(text):
    nlp = spacy.load("en_core_web_sm")
    degrees = []
    subject=[]
    total_subject=[]
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for education degrees
    patterns = [
        [{"LOWER": "bachelor"},{"LOWER":"degree"}],
        [{"LOWER": "bachelor"}],
        [{"LOWER": "master"},{"LOWER":"degree"}],
        [{"LOWER": "master"}],
        [{"LOWER": "educational"},{"LOWER":"degree"}],
        [{"LOWER": "phd"}],
        [{"LOWER": "mba"}],
        [{"LOWER": "mca"}],
        [{"LOWER": "md"}],
        [{"LOWER": "msc"}],
        [{"LOWER": "m"},{"LOWER": "sc"}],
        [{"LOWER": "ma"}],
        [{"LOWER": "bs"}],
        [{"LOWER": "be"}],
        [{"LOWER": "b"},{"LOWER": "e"}],
        [{"LOWER": "m"},{"LOWER": "e"}],
        [{"LOWER": "bsc"}],
        [{"LOWER": "b"},{"LOWER": "sc"}],
        [{"LOWER": "ms"}],
        [{"LOWER": "me"}],
        [{"LOWER": "btech"}],
        [{"LOWER": "mtech"}],
        [{"LOWER": "b"},{"LOWER": "tech"}],
        [{"LOWER": "m"},{"LOWER": "tech"}],
        [{"LOWER": "b"},{"LOWER": "com"}],
        [{"LOWER": "m"},{"LOWER": "com"}],
        [{"LOWER": "bcom"}],
        [{"LOWER": "mcom"}],
        [{"LOWER": "bba"}],
        [{"LOWER": "dnp"}],
        [{"LOWER": "bca"}]
        # Add more patterns as needed
    ]
    subs=['accounting', 'business administration', 'clinical psychology', 'computer science', 'counselling', 
          'data science', 'economics', 'engineering', 'finance', 'graphics', 'marketing', 'math', 'none','nursing','statistic', 'taxation']
    
    for pattern in patterns:
        matcher.add("EDUCATION_DEGREE", [pattern])
    
    doc = nlp(text)
    matches = matcher(doc)
    f=0
    for match_id, start, end in matches:
        span = doc[start:end]
        c=doc[start:end+5]
        for i in subs:
            if i in c.text:
              f=1
              subject.append(i)
              break
        if f==0:
          subject.append('none')
        f=0
        degrees.append(span.text)
    print(degrees)
    if not degrees:  # If dg list is empty
        degrees.append('anygrad')
        subject.append('none')
    return degrees, subject


# In[74]:


def highDegree_highEdu(text):
  nltk.download(['stopwords','wordnet'])
  warnings.filterwarnings('ignore')
  degree=[]
  subject=[]
  #degree_clean = clean_data(text)
  degree_resume1, subject_resume= extract_education_degrees(text)
  #print(degree_resume1)
  #print(subject_resume)
  d=['msc', 'be', 'bsc', 'anygrad', 'mtech', 'dnp', 'bba', 'mca', 'btech', 'bcom','mba','phd']
  f_degrees = [degree for degree in degree_resume1 if degree in d]
  if not f_degrees:  # Check if the list is empty that means degrees_resume1 only contains other degrees
    filtered_degrees= degree_resume1
  else:
    filtered_degrees= f_degrees
  if 'phd' in filtered_degrees:
    degree.append('phd')
  elif 'mba' in filtered_degrees:
    degree.append('mba')
  elif 'mtech' in filtered_degrees:
    degree.append('mtech')
  elif 'btech' in filtered_degrees or 'be' in filtered_degrees:
    if 'btech' in filtered_degrees:
      degree.append('btech')
    else:
      degree.append('be')
  elif 'dnp' in filtered_degrees:
    degree.append('dnp')
  else:
    for i in filtered_degrees:
      if i[0] =='m':
        degree.append(i)
        break
    if degree==[]:
        for i in filtered_degrees:
          if i[0] =='b':
            degree.append(i)
            break
  if not filtered_degrees:
      degree.append('anygrad')
      subject.append('none')
  else:
    subject=subject_resume[degree_resume1.index(degree[0])]
    if 'none' in subject:
        for item in subject_resume:
            if item != 'none':
                subject = item
                break
    if degree[0] not in d:
        degree[0] = 'anygrad'
  return degree[0],subject


# ## Experience Encode

# In[75]:


def encoding( subject, degree, total_years):
    #years
    #  0-1 years   -0
    #  1-5 years   -1
    #  5-10 years  -2
    #  10+ years   -3
    arr = np.array([0,2,6,11])
    total_years_encoded = arr.searchsorted(total_years, side='right') - 1
    #subject
    Subject_encoding={'accounting': 0, 'business administration': 1, 'clinical psychology': 2, 'computer science': 3, 'counselling': 4, 'data science': 5, 'economics': 6, 'engineering': 7, 'finance': 8, 'graphic': 9, 'marketing': 10, 'math': 11, 'none': 12, 'nursing': 13, 'statistic': 14, 'taxation': 15}
    encoded_subject = Subject_encoding.get(subject, -1)
    #degree
    Degree_encoding= {'anygrad': 0, 'bcom': 1, 'be': 2, 'btech': 3, 'bba': 4, 'bsc': 5, 'dnp': 6, 'mtech': 7, 'mba': 8, 'mca': 9, 'msc': 10, 'phd': 11}
    encoded_degree = Degree_encoding.get(degree, -1)
    return encoded_subject, encoded_degree, total_years_encoded


# ### Starting  Function

# In[76]:


def extract_resume_features(path):
    nltk.download(['stopwords','wordnet'])
    text=extract_text_from_panes(path)#text in string
    clean_resume=clean_data(text)#clean_resume in list
    skills_resume = get_technical_skills(clean_resume[0])#sending first string of list
    clean_resume = ' '.join(clean_resume)##converting list to string
    extracted_experience_strings = extract_experience_strings(clean_resume)
    total_years = calculate_total_years(extracted_experience_strings)
    degree, subject= highDegree_highEdu(clean_resume)
    subject_encoded, degree_encoded, experience_encoded= encoding(subject, degree, total_years)
    return skills_resume, subject_encoded, degree_encoded, experience_encoded


# In[ ]:





# In[ ]:




