__author__ = 'raghav'
# This file is used to compute the pretrained vector embeddings in the format
# It's expecting each line to have a word and a vector separated by a space. e.g.
# pizza oFZJvTg5KL1D2188TLuNPXvpJL3YZI49bFeWvL8RjbzOo5s94I7aPIzzHj0zaIw7MFl6PTka7b3+sne8asDvPEPb37wJC5E7hwPIPKBWyby4yIU9CHRqvVwu4Tu5q
# \n

import gensim


def loadword2VecModel(word2vecBinaryModelFile):
    print "Loading word2vec Model-!!!!"
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format(word2vecBinaryModelFile, binary=True)

    return model

def loadGloveModel(gloveFile):
    print "Loading Glove Model-!!!!"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded-!!!!!"
    return model




word2vecModel = loadword2VecModel('/Users/raghav/Downloads/GoogleNews-vectors-negative300.bin')
glove_model = loadGloveModel("/Users/raghav/Downloads/glove.840B.300d.txt")


print "Word2vec representation .... of word in 300 dimension",word2vecModel['pizza']
print "Glove vector representation of word in 300 dimension",glove_model['pizza']
