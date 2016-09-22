from __future__ import division  # Python 2 users only
import glob
import re
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from random import randint
import math

#
# Classes
#

#
# NGram recursively calls itself
#
class NGram:
    def __init__(self, depth=1):
        self.depth = depth
        self.counts = defaultdict(NGram)
        self.total = 0
        self.smoothTotal = 0
    def add(self, words):
        # if self.depth != len(words) :
        #     raise IndexError("Words must be an array of length equal to the depth of the nGram")
        if self.depth == 1:
            self.counts[words[0]].total +=1
        else:
            # continue adding the rest of the words through layers of the nGram
            self.counts[words[0]].add(words[1:])
        self.total +=1
#
# Functions
#
def filePreprocessor(string):
    # Remove text up to and including these strings;
    removeTokens = ["From :", "Re :", "Subject :", "write :", "wrote :"]
    for token in removeTokens:
        string = string.split(token)[-1]
    return string

def stringPreprocessor(string):
    # regex to remove any word with @ in it
    # also remove From :, Subject :, lone < or >
    email_regex = '[^ ]+\@[^ ]+\.[^ ]+'
    chars = ' [^a-zA-Z\d\s:!(.\!?)]|\> | \<'

    regex = re.compile('(' + email_regex + '|' + chars + ')', re.VERBOSE)
    return re.sub(regex, '', string)

#
# Loops over files to read, for bigrams replaces count 1 tokens with <unk> via readToken
def collectionIterator(file_paths,ngrams, unigrams=False):
    for path in file_paths:
        # print ("Now reading %s", path)
        raw = open(path, 'r').read()
        processed_file = filePreprocessor(raw)
        sen_tokens = sent_tokenize(processed_file)
        for sen in sen_tokens:
            # might need to point </s> to <s> for bigrams

            sentence = stringPreprocessor(sen)
            tokens = word_tokenize(sentence)
            tokens.append("</s>")
            for idx, token in enumerate(tokens):
                if unigrams:
                    # doing bigram
                    tokenIdx_1 = "<s>"
                    if idx != 0:
                        tokenIdx_1 = readToken(tokens[idx-1],unigrams)
                    token = readToken(token, unigrams)
                    ngrams.add([tokenIdx_1,token])
                else:
                    ngrams.add([token])
    return ngrams;

def readToken(token, unigrams):
    if unigrams.counts[token].total > 1:
        return token
    return "<unk>"

def readTestToken(token, unigrams):
    if unigrams.counts[token].total > 0:
        return token
    return "<unk>"

def unigramGenSentence(unigrams, initial):
    new_word = initial
    while new_word != "</s>":
        word_index = randint(0,unigrams.total)
        new_word = findSeedWord(unigrams, word_index)
        initial += " "+ new_word
    return initial

def findSeedWord(ngram, random_word_index):
    for k,v in ngram.counts.iteritems():
        random_word_index -= v.total
        if random_word_index <= 0:
            return k
    return "</s>"

def bigramGenSentence (bigrams, initial):
    seed_word = initial
    seed_gram = bigrams.counts[seed_word]
    while seed_gram.total > 0 and seed_word != "</s>":
        word_index = randint(0,seed_gram.total)
        seed_word = findSeedWord(seed_gram, word_index)
        initial += " "+ seed_word
        seed_gram = bigrams.counts[seed_word]
    return initial

#
# smoothing funcitons
def removeUnk(unigrams):
    deleteKeys = []
    for k,v in unigrams.counts.iteritems():
        if v.total == 1:
            deleteKeys.append(k)
    for k in deleteKeys:
        unigrams.counts.pop(k,None)
        unigrams.add(["<unk>"])
    return unigrams

def readCmap(count, cmap):
    if count >= len(cmap):
        return count;
    else:
        return cmap[count]

def smoothGoodTuring(bigram):
    vocab_size = len(bigram.counts)
    # N0 = V^2 - N
    Narray = [0]*12
    Narray[0] = vocab_size * vocab_size - bigram.total
    for k,v in bigram.counts.iteritems():
        for k2, v2 in v.counts.iteritems():
            if v2.total < 12:
                Narray[v2.total] += v2.total

    cmap = [0]*11
    for c in range(11):
        cmap[c] = (c+1)* Narray[c+1] / Narray[c]
    print("and the cmap is ", cmap)

    # compute adjusted totals
    for k,v in bigram.counts.iteritems():
        numNgrams = 0
        smoothTotal = 0
        for k2, v2 in v.counts.iteritems():
            numNgrams += 1
            v2.smoothTotal = readCmap(v2.total, cmap)
            smoothTotal += v2.smoothTotal
            v.counts[k2] = v2
        smoothTotal += ( vocab_size - numNgrams)*cmap[0]
        v.smoothTotal = smoothTotal
        bigram.counts[k] = v

    return (cmap, bigram)

#
# Computational Functions
def findPerplexity(bigrams, unigrams, cmap, file_paths):
    logsum = 0
    divisor = 0
    for path in file_paths:
        # print ("Now reading %s", path)
        raw = open(path, 'r').read()
        processed_file = filePreprocessor(raw)
        sen_tokens = sent_tokenize(processed_file)
        for sen in sen_tokens:
            # might need to point </s> to <s> for bigrams

            sentence = stringPreprocessor(sen)
            tokens = word_tokenize(sentence)
            tokens.append("</s>")
            for idx, token in enumerate(tokens):
                tokenIdx_1 = "<s>"
                if idx != 0:
                    tokenIdx_1 = readTestToken(tokens[idx-1],unigrams)
                token = readTestToken(token, unigrams)

                # if bigrams.counts[tokenIdx_1].smoothTotal == 0:
                    # print("uhg it's 0 ", bigrams.counts[tokenIdx_1].smoothTotal)
                    # print("and tk-1 ", tokenIdx_1)
                logsum -= math.log(readCmap(bigrams.counts[tokenIdx_1].counts[token].total, cmap)/bigrams.counts[tokenIdx_1].smoothTotal)
                divisor += 1

    return math.exp(1/divisor * logsum);


def generateUnigramBigram():
    category_paths =  glob.glob('data_corrected/classification_task/*')
    unigramCollection = {}
    bigramCollection = {}
    # category_paths = [category_paths[0]]
    for category in category_paths:
        # BEGIN iteration over categories
        print("---")
        print("------------",category,"------------")
        print("---")

        unigrams = NGram(1)
        bigrams = NGram(2)

        file_paths = glob.glob(category+"/train_docs/*.txt")
        # file_paths = [file_paths[0]]
        # BEGIN iteration over paths
        twothirdsmark = int(len(file_paths)*2/3)

        training_file_paths = file_paths[:twothirdsmark]
        test_file_paths = file_paths[twothirdsmark:]

        unigrams = collectionIterator(training_file_paths,unigrams)
        bigrams = collectionIterator(training_file_paths,bigrams,unigrams)

        mappedUnigrams = removeUnk(unigrams)
        # Now post process data for smoothing
        cmap, mappedBigrams = smoothGoodTuring(bigrams)
        perplexity = findPerplexity(mappedBigrams, mappedUnigrams, cmap, test_file_paths)
        print ("and the perplexity is ", perplexity)

        unigramCollection[category] = unigrams
        bigramCollection[category] = bigrams

        # print("-------- Unigram ----------")
        # for i in range(15):
        #     print(unigramGenSentence(unigrams, "<s>"))
        #
        # print("-------- Bigram ----------")
        # for i in range(15):
        #     print(bigramGenSentence(bigrams, "<s>"))

    return (unigramCollection, bigramCollection)

#
# Raw Code
#

# # Needed for initial run
# nltk.download('punkt')

# Initialization

uCollect, biCollect = generateUnigramBigram();
