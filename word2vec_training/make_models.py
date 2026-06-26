import g17_preprocessing_class
import g25_preprocessing_class
import incel_preprocessing_class
from gensim.models import Word2Vec

continuous_data = Reader("general17")
model17 = Word2Vec(continuous_data, min_count=75, window=5, sg=1, negative =5)
model17.save("g17")

reddit_data = Reader25("reddit_dataset")
model25 = Word2Vec(reddit_data, min_count=75, window=5, sg=1, negative =5)
model25.save("g25")

continuous_data = ReaderIncel("incel_dataset")
model_I = Word2Vec(continuous_data, min_count=75, window=5, sg=1, negative =5)
model_I.save("incel")
