from gensim.models import Word2Vec
import nltk
import os
import csv
g25 = Word2Vec.load("g25")
g17 = Word2Vec.load("g17")
incel = Word2Vec.load("incel")

'''
Return csv with the word similarity for each context, for each target word
 - word similarity measured in comparision with the average word similarity for that context
slang_list: list of target words
'''

slang_list = ["incel", "femcel", "chad", "chads", "tyrone", "tyrones", "alpha", "alphas", "beta", "betas", "cope", "copes", "pill", "pills", "pilled", "piller", "pilling", "maxx", "maxing", "maxxing", "maxed", "maxxed", "mogg", "moggs", "mogging", "mogged", "femoid", "foid"]
with open("analysis.csv", "w") as f:
    analysis_writer = csv.writer(f)
    analysis_writer.writerow(["fname", *slang_list, "average"])
    for fname in os.listdir("final_analysis"):
        fname = os.path.join("final_analysis", fname)
        print (fname)
        if fname.endswith(".json"):
            with open(fname, "r") as f:
                c_dict = json.load(f)
                total = 0
                for word in c_dict.keys():
                    total += c_dict[word]
                    avg = total/len(c_dict)
                    all_entries = []
                    for word in slang_list:
                        try:
                            entry = c_dict[word]/avg
                        except:
                            entry = "NaN"
                        all_entries.append(entry)
            analysis_writer.writerow ([fname, *all_entries, avg])
    per_slang = []
    incelsum =sum(incel.wv.get_vecattr(word, "count") for word in incel.wv.key_to_index.keys())/100000
    g17sum = sum(g17.wv.get_vecattr(word, "count") for word in g17.wv.key_to_index.keys())/100000
    g25sum = sum(g25.wv.get_vecattr(word, "count") for word in g25.wv.key_to_index.keys())/100000
    for word in slang_list:
        try:
            count = g17.wv.get_vecattr(word, "count")
            per = count/g17sum
        
        except:
            per = "NaN"
        per_slang.append(per)
    analysis_writer.writerow (["g17", *per_slang])
    per_slang = []
    for word in slang_list:
        try:
            count = g25.wv.get_vecattr(word, "count")
            per = count/g25sum
        
        except:
            per = "NaN"
        per_slang.append(per)
    analysis_writer.writerow (["g25", *per_slang])
    per_slang = []
    for word in slang_list:
        try:
            count = incel.wv.get_vecattr(word, "count")
            per = count/incelsum
        except:
            per = "NaN"
        per_slang.append(per)
    analysis_writer.writerow (["incel", *per_slang])

