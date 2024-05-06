#!/usr/bin/env python
# coding: utf-8

# # **Skills Extraction**

# In[1]:


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


# # **Clean Resume Function**

# In[2]:


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


# # **NLP and JSONL Initialization**

# In[3]:


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

# In[4]:


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


# ## Starting Function

# In[5]:


def skills_extraction(text):
    clean_text= clean_data(text)
    skills= get_technical_skills(clean_text[0])
    return skills#list

