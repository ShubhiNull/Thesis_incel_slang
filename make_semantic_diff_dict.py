from gensim.models import Word2Vec
import nltk
import numpy
from scipy.spatial import distance
import json

'''
    finding local neighbourhood change using merged method
    Small model: model trained on smaller/specialized datset
    Big model: model trained on bigger/general dataset 

'''

small_model = Word2Vec.load("g17")
big_model = Word2Vec.load("incel")
smallvocab = list(small_model.wv.key_to_index.keys()) 
bigvocab = list(big_model.wv.key_to_index.keys()) 

word_sim = {}
vocabulary = (set(smallvocab) & set(bigvocab))
for word in vocabulary:
    #secondary vectors
    vec1 = []
    vec2 = [] 
    bigsim= [word for word, score in big_model.wv.most_similar(word)]
    smallsim = [word for word, score in small_model.wv.most_similar(word)]
    smallneighbours = list(set (smallsim))
    #if a word is present in the big dataset but not in the small datset, ignore the word
    bigneighbours = list(set (bigsim)& set (smallvocab))
    n = 0
    #append list of neighbouring words for the big model until it is equal to 10
    while len(bigneighbours) < 10:
        cw = [word for word, score in big_model.wv.most_similar(word, topn = n+10)]
        if cw[-1] in smallvocab:
            bigneighbours.append(cw[-1])
        n+=1
    word_neighbours = list(set(smallneighbours) | set(bigneighbours))
    for neighbour in word_neighbours:
        #if a word is present in the small dataset but not the big dataset, set sim= 0
        try:
            b = big_model.wv.similarity(word, neighbour) 
        except:
            b = 0
        s = small_model.wv.similarity(word, neighbour)
        vec1.append(b)
        vec2.append(s)
    #append cosine distance of secondary vectors to a dictionary
    cos_dis = distance.cosine(numpy.array(vec1), numpy.array(vec2))
    word_sim[word] = float(cos_dis)

#name json file   
with open("g17_incel_merge.json", "w") as file:
    json.dump (word_sim, file, indent = 4)