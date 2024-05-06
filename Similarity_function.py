#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[1]:

import math
import nltk
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
import fitz
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#!pip install jsonlines


# In[3]:


#pip install word2number


# In[4]:


#!pip install pymupdf


# In[5]:


#!python -m spacy download en_core_web_lg


# # **NLP and JSONL Initialization**

# In[6]:


def update_entity_ruler(nlp, patterns_path):
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
    else:
        existing_ruler = nlp.get_pipe("entity_ruler")
        existing_ruler.from_disk(patterns_path)


# # **Skill Extraction Function**

# In[7]:


def get_technical_skills(text):
    nlp = spacy.load("en_core_web_sm")
    skill_pattern_path = "jz_skill_patterns2.jsonl"
    update_entity_ruler(nlp, skill_pattern_path)
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

# In[8]:


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

# In[9]:


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


# ### **Feature Extraction(Skills)**

# In[10]:


# Define a function to extract features
def extract_features(resume_text):
    features = {}

    # Extract Top Skills
    skills_pattern = r"Top Skills\n([\s\S]+?)\n(?:Summary|Languages|Certifications|Experience|Education)"
    skills_match = re.search(skills_pattern, resume_text)
    if skills_match:
        features['Skills'] = skills_match.group(1).strip().split('\n')
    else:
      features['Skills']=[]

    return features


# ### **Skills Extraction**

# In[11]:


def find_skills(text):
  clean_resume=clean_data(text)
  f=extract_features(text)
  skills_resume = get_technical_skills(clean_resume[0])
  if len(f['Skills']) > 0:
    if len(f['Skills']) == 3:
      for i in f['Skills']:
        skills_resume.append(str(i).lower())
    else:
      for i in f['Skills'][0:3]:
        skills_resume.append(str(i).lower())
    skills_resume=list(set(skills_resume))
    if 'microsoft office'in skills_resume:
      skills_resume.remove('microsoft office')
      skills_resume.append('microsoft word')
      skills_resume.append('microsoft powerpoint')
      skills_resume.append('excel')
      skills_resume=list(set(skills_resume))
  return skills_resume


# ### **Years of Experience**

# In[12]:


def extract_years_of_experience3(text):
    doc = nlp(text)
    total_years_of_experience = []

    for ent in doc.ents:
        i=0
        s=0
        if ent.label_ == "DATE":
            # Check if the entity text contains "years"
            if 'year' in ent.text or 'month' in ent.text:
              if 'month' in ent.text and 'year' in ent.text:
                for token in ent:
                  if token.pos_=="NUM" and len(token)<=2:
                    i=i+1
                    if i==2:
                      s=s+ (int((token.text)))/12
                    else:
                      s=s+int(token.text)
                total_years_of_experience.append(round(s,1))
              elif 'month' in ent.text:
                for token in ent:
                  if token.pos_=="NUM" and len(token)<=2:
                    total_years_of_experience.append(round((w2n.word_to_num(token.text))/12,1))
              else:
                for token in ent:
                  if token.pos_=="NUM" and len(token)<=2:
                    total_years_of_experience.append(w2n.word_to_num(token.text))


    return total_years_of_experience


# In[13]:


def find_exp(text):
    clean_resume=clean_data(text)
    exp_years=extract_years_of_experience3(clean_resume[0])
    total_year= round(sum(exp_years),1)
    return total_year


# ### **Education Degree**

# In[14]:


import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

def extract_education_degrees(text):
    degrees = []
    subject=[]
    total_subject=[]
    matcher = Matcher(nlp.vocab)

    # Define patterns for education degrees
    patterns = [
        [{"LOWER": "bachelor"},{"LOWER":"degree"}],
        [{"LOWER": "master"},{"LOWER":"degree"}],
        [{"LOWER": "educational"},{"LOWER":"degree"}],
        [{"LOWER": "master"}],
        [{"LOWER": "bachelor"}],
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
        [{"LOWER": "bsc"}],
        [{"LOWER": "b"},{"LOWER": "sc"}],
        [{"LOWER": "bca"}],
        [{"LOWER": "bba"}],
        [{"LOWER": "ms"}],
        [{"LOWER": "btech"}],
        [{"LOWER": "mtech"}],
        [{"LOWER": "b"},{"LOWER": "tech"}],
        [{"LOWER": "m"},{"LOWER": "tech"}],
        [{"LOWER": "bcom"}],
        [{"LOWER": "b"},{"LOWER": "com"}],
        [{"LOWER": "mcom"}],
        [{"LOWER": "m"},{"LOWER": "com"}],
        [{"LOWER": "ba"}]
        # Add more patterns as needed
    ]

    for pattern in patterns:
        matcher.add("EDUCATION_DEGREE", [pattern])

    doc = nlp(text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        degrees.append(span.text)

    return degrees



# In[15]:


def find_degree(text):
    text=text.replace('M.Sc','msc')
    text=text.replace('B.Sc','bsc')
    text=text.replace('(M.B.A.)','mba')
    degree_clean=clean_data(text)
    degree_resume=extract_education_degrees(degree_clean[0])
    degree_resume = list(np.unique(degree_resume))

    for i in degree_resume:
        if i[0] == 'b':
            degree_resume.append('bachelor degree')
            break
    for i in degree_resume:
        if i[0] == 'm':
            degree_resume.append('master degree')
            break

    degree_resume = list(np.unique(degree_resume))
    return degree_resume


# In[16]:


def remove_year_range(text):
    # Define a regular expression pattern to match the year range
    pattern = r'\s+to\s+\d+\s+years?\b'
    pattern1 = r'\s*-\s*(\d+)\s*years'
    # Use re.sub to replace the matched pattern with an empty string
    updated_text = re.sub(pattern, ' year', text)
    updated_text = re.sub(pattern1, ' year', text)
    return updated_text


# ### **Job Description Skills**

# In[17]:


def find_skills_jd(job_des):
    job_clean=clean_data(job_des)
    skills_job = get_technical_skills(job_clean[0])
    return skills_job


# ### **Job Description years**

# In[18]:


def find_exp_jd(job_des):
    job_clean=clean_data(job_des)
    job_clean[0]=job_clean[0].replace('yr','year')
    job_exp_years=extract_years_of_experience3(job_clean[0])
    job_years=sum(job_exp_years)
    return job_years


# ### **Job Description Edu Degree**

# In[19]:


def find_degree_jd(job_des):
    job_clean=clean_data(job_des)
    degree = extract_education_degrees(job_clean[0])
    for i in degree:
      if i[0]=='b':
        degree.append('bachelor degree')
        break
    for i in degree:
      if i[0]=='m':
        degree.append('master degree')
        break
    degree=list(np.unique(degree))
    return degree


# # **Similarity**

# ### **Skills**

# In[20]:


def cosine(skills_resume,skills_job):

    # Join the words into strings
    text1 = ' '.join(skills_job)
    text2 = ' '.join(skills_resume)

    # Vectorize the texts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(X[0], X[1])
    cosine_sim=round((float(cosine_sim)),1)
    #print("Cosine Similarity:", cosine_sim)
    return cosine_sim


# ### **Work years**

# In[21]:


def simi_years(total_year,job_years,text,job_des):
    skills_resume =find_skills(text)
    skills_job=find_skills_jd(job_des)
    match_skills=cosine(skills_resume,skills_job)
    if match_skills!=0.0:
      if total_year==0 and job_years==0:
        match_year=1.0
      elif total_year==0 and job_years!=0:
        match_year=0.0
      elif job_years<=total_year:
        match_year=1
      else:
        match_year=0.1
    else:
      if total_year==0 and job_years==0:
        match_year=0.7
      elif total_year==0 and job_years!=0:
        match_year=0.0
      elif job_years<=total_year:
        match_year=0.7
      else:
        match_year=0.1
    return match_year


# ### **Education Degree**

# In[22]:


def simi_edu(degree,degree_resume,text,job_des):
    skills_resume =find_skills(text)
    skills_job=find_skills_jd(job_des)
    match_skills=cosine(skills_job,skills_resume)
    score = 0
    if len(degree) >0:
      for x in degree:
          if x in degree_resume:
              score += 1
      job_degree_len = len(degree)
      match_per = round(score / job_degree_len * 100, 1)
    else:
      match_per=100

    if match_skills !=0.0:
      if match_per >0.0 :
        match_degree=1
      else:
        match_degree=0.1
    else:
      if match_per >0.0 :
        match_degree=0.7
      else:
        match_degree=0.1
    return match_degree


# ## **Overall Similarity**

# In[23]:


def find_similarity(path, job_des):
  text=extract_text_from_panes(path)
  skills_resume = find_skills(text)
  total_year= find_exp(text)
  degree_resume=find_degree(text)
  job_des=remove_year_range(job_des)
  skills_job=find_skills_jd(job_des)
  job_years=find_exp_jd(job_des)
  degree=find_degree_jd(job_des)
  match_skills=cosine(skills_resume,skills_job)
  match_year= simi_years(total_year,job_years,text,job_des)
  match_degree= simi_edu(degree,degree_resume,text,job_des)
  matching=((match_skills+match_year+match_degree)/3.0)*100
  matching = round(matching,2)

  if match_degree==0.1:
    str1="Your educational degree doesn't match."
  else:
    str1=" "
  if match_year==0.1 or match_year==0.0:
    str2= 'You need {match} more years of experience.'.format(match=math.ceil(abs(total_year-job_years)))
  else:
    str2=" "
  return matching,str1,str2