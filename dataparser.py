from __future__ import division  # Python 2 users only
import glob
import re
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from random import randint

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
# Raw Code
#

# # Needed for initial run
# nltk.download('punkt')

# Initialization
category_paths =  glob.glob('data_corrected/classification_task/*')

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
    for path in file_paths:
        # print ("Now reading %s", path)
        raw = open(path, 'r').read()
        processed_file = filePreprocessor(raw)
        sen_tokens = sent_tokenize(processed_file)
        for sen in sen_tokens:
            # might want to add <s> token for beginning of sentence
            sentence = stringPreprocessor(sen)
            tokens = word_tokenize(sentence)
            tokens.append("</s>")
            for idx, token in enumerate(tokens):
                tokenIdx_1 = "<s>"
                if idx != 0:
                    tokenIdx_1 = tokens[idx-1]

                bigrams.add([tokenIdx_1,token])
                unigrams.add([token])


    print("-------- Unigram ----------")
    for i in range(15):
        print(unigramGenSentence(unigrams, "<s>"))

    print("-------- Bigram ----------")
    for i in range(15):
        print(bigramGenSentence(bigrams, "<s>"))
