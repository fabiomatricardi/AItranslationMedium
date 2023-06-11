import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import datetime


# 🈚🆗✅💬🇮🇹🇺🇸
#LOCAL MODEL EN-IT
#---------------------------------
#  Helsinki-NLP/opus-mt-en-it
Model_IT = './model_it/'   #torch
#---------------------------------
English = "Another common struggle that Python programmers face with pip is package installation errors. Some errors may come because the library you are installing requires compiling on your local computer: others can happen when there are issues with the package itself or when there are network or permission issues."
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# INITIALIZE TRANSLATION FROM ENGLISH TO ITALIAN TORCH MODEL      
tokenizer_tt0it = AutoTokenizer.from_pretrained(Model_IT)  
print(' Initializing AI Model & pipeline...')
model_tt0it = AutoModelForSeq2SeqLM.from_pretrained(Model_IT)  #Helsinki-NLP/opus-mt-en-it  or #Helsinki-NLP/opus-mt-it-en
print("pipeline")
TToIT = pipeline("translation", model=model_tt0it, tokenizer=tokenizer_tt0it)
# ITERATE OVER CHUNKS AND JOIN THE TRANSLATIONS
print("translation in progress")
start = datetime.datetime.now() #not used now but useful
finaltext = TToIT(English)
stop = datetime.datetime.now() #not used now but useful
elapsed = stop - start
print(f'Translation generated in {elapsed}...')
print(finaltext[0]['translation_text'])
print(f"Translated number {len(English.split(' '))} of words")