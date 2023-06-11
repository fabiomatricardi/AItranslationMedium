import tensorflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import datetime

# 🈚🆗✅💬🇮🇹🇺🇸
#LOCAL MODEL EN-ES
#---------------------------------
#  Helsinki-NLP/opus-mt-tc-big-en-es
#---------------------------------
Model_ES = './model_es/'  # tensorflow
#---------------------------------
English = "Another common struggle that Python programmers face with pip is package installation errors. Some errors may come because the library you are installing requires compiling on your local computer: others can happen when there are issues with the package itself or when there are network or permission issues."
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# INITIALIZE TRANSLATION FROM ENGLISH TO ENGLISH with TENSORFLOW MODEL       
tokenizer_tt0es = AutoTokenizer.from_pretrained(Model_ES)  
print(' Initializing AI Model & pipeline...')
model_tt0es = AutoModelForSeq2SeqLM.from_pretrained(Model_ES, from_tf=True)  #Helsinki-NLP/opus-mt-en-it  or #Helsinki-NLP/opus-mt-it-en
print("pipeline")
TToES = pipeline("translation", model=model_tt0es, tokenizer=tokenizer_tt0es)
start = datetime.datetime.now() #not used now but useful
print('Translation in progress...')
finaltext = TToES(English)
stop = datetime.datetime.now() #not used now but useful
elapsed = stop - start
print(f'Translation generated in {elapsed}...')
print('*'*50)
print(finaltext[0]['translation_text'])
print('*'*50)
print(f"Translated number {len(English.split(' '))} of words")