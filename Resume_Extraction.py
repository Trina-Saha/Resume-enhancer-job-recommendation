#!/usr/bin/env python
# coding: utf-8

# ## Resume Extraction

# ### **Text Extraction**

# In[44]:


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


# In[45]:


#!pip install jsonlines


# In[46]:


#!pip install word2number


# In[47]:


#!pip install pymupdf


# In[48]:


#!python -m spacy download en_core_web_lg


# In[49]:


def resume_extraction(pdf_path):
    text_combined = ""

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Extract text from the current page
        page_text = page.get_text()

        # Append extracted text from the current page to overall text
        text_combined += page_text.strip() + "\n"



    return text_combined.strip()#string output
