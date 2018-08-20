import gensim
import os
import numpy as np
from numpy import float32
from six import string_types, integer_types
from gensim import utils, matutils
print(os.getcwd())
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

# vector = model.wv["loop"]

a = matutils.unitvec(model.wv["woman"])


b = matutils.unitvec(model.wv["king"])
c = matutils.unitvec(model.wv["man"])
d = a+b-c # by itself, this is NOT a vector...most likely we need to do X
d=d/3
# the actual implementation calls for this:
# take the projection weight vectors of the given words, then compare these to all other words in the model
# that is, take a simple average of all the vectors, then compute the dot product between every vector and this vector
# and find the largest ones. Most similar_by_vector should do this however!
d = matutils.unitvec(d).astype(float32)

#actually behind the scenes, most similar by vector CALLS most similar. But we lose some of the information
# we must rescale as a unit vector, most likely

# we want to get: X and Y out simply
# numpy get l2 norm of vector

# d=d/3


print(d)
#        This method computes cosine similarity between a simple mean of the projection
#         weight vectors of the given words and the vectors for each word in the model.


print(model.most_similar(positive=['woman', 'king'], negative=["man"]))
print(model.similar_by_vector(d, topn=10, restrict_vocab=None))
# d_word_vec = np.array(d, dtype=np.float32)
# print(model.most_similar(d_word_vec))
# print(vector)



print(model.most_similar(positive=['woman'],topn=10))

# vs
print(model.similar_by_vector(a,topn=10))