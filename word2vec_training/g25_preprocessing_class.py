from string import punctuation
import string
import pyarrow.parquet as reader 
import pandas
import os
import utils
import nltk
from gensim.models import Word2Vec
from collections import defaultdict
import re
#adding unicodes for punct present on reddit but not in string.punctuation
punct = string.punctuation.__add__('—’"”“')
import emoji

class Reader25:
'''
    Class to preprocess Reddit data from 2025
'''
    def __init__(self, directory):
        self.directory = directory 

    def __iter__(self):  
        for fname in os.listdir(self.directory):
            fname = os.path.join(self.directory, fname)
            
            if not os.path.isfile(fname):
                continue
            if fname.endswith(".parquet"):
                dataframe = pandas.read_parquet(fname)
                # filtering out posts from other years
                dataframe['now'] = dataframe['datetime'].str.contains('2025')
                filtered_df = dataframe[dataframe['now']]
                for text in filtered_df['text']:  
                    tokenized_data = self.preprocess(text)
                    yield tokenized_data
                        

    def preprocess(self, post):
        tokenized_data = []
        list_of_sentences = nltk.sent_tokenize(post)
        for sentence in list_of_sentences:
            token_words = nltk.word_tokenize(sentence)
            sentence.strip().split("/")
            clean_tokens = []
            for i, word in enumerate(token_words):
                if word and not all(chr in punct for chr in word) and not word.startswith(("//", "gif", "giphy","https")):
                    word_lower = word.lower()
                    if any(char in emoji.EMOJI_DATA for char in word_lower):
                        word_lower = ''.join(
                            char for char in word_lower
                            if char not in emoji.EMOJI_DATA
                                )
                    if "/" in word_lower and not word_lower.startswith(("/u", "/r", "r/", "u/")):
                        parts = word_lower.split("/")
                        
                        for part in parts:
                            clean_tokens.append(part)
                        continue
                     
                    max_match = re.match(r"^(.*?)(maxx+)(ing|ed|er|es)?$", word_lower)  
                    if max_match and word not in {"tj-maxx", "tk-maxx", "tjmaxx", "tkmaxx"}:
                        if(
                            word_lower.startswith("maxx")
                            and i > 0
                            and token_words[i - 1].lower() in {"tj", "tk"}
                                ):
                            clean_tokens.append("tj_maxx")
                            continue
                
                        prefix, root, suffix = max_match.group(1), max_match.group(2), max_match.group(3)
                        if prefix:
                            clean_tokens.append(prefix)
                        clean_tokens.append("maxx" + (suffix or ""))
                        continue
                    ma_match = re.match(r"^(.*?)(max)(ing|ed|er|es)?$", word_lower)
                    
                    if ma_match and word not in {"tj-max", "tk-max", "tjmax", "tkmax"}:
                        if(
                            word_lower.startswith("max")
                            and i > 0
                            and token_words[i - 1].lower() in {"tj", "tk"}
                                ):
                            clean_tokens.append("tj_max")
                            continue
                        prefix, root, suffix = ma_match.group(1), ma_match.group(2), ma_match.group(3)
                        if prefix:
                            clean_tokens.append(prefix)
                        clean_tokens.append("max" + (suffix or ""))
                        continue
                    pill_match = re.match(r"^(.*?)(pill+)(ing|ed|er|s)?$", word_lower)
                    if pill_match and not word_lower.startswith("spill"):
                        prefix, root, suffix = pill_match.group(1), pill_match.group(2), pill_match.group(3)
                        if prefix:
                            clean_tokens.append(prefix)
                        clean_tokens.append("pill" + (suffix or ""))
                        continue
                    mog_match = re.match(r"^(.*?)(mog+)(ing|ed|er|s)?$", word_lower)
                    if mog_match:
                        prefix, root, suffix = mog_match.group(1), mog_match.group(2), mog_match.group(3)
                        if prefix:
                            clean_tokens.append(prefix)
                        clean_tokens.append("mogg" + (suffix or ""))
                        continue
                    clean_tokens.append(word_lower) 
            clean_tokens.insert(0, '<s>')
            tokenized_data.extend(clean_tokens)
        return tokenized_data
