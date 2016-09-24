from __future__ import division  # Python 2 users only
import glob
import re
import nltk, re, pprint
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from random import randint
import math
import operator


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
class SPNGram(NGram):
    def __init__(self, depth=1):
        NGram.__init__(self,depth)
        self.counts = defaultdict(SPNGram)
    def sub(self,words):
        # This is for negative counts in spell checking
        if self.depth == 1:
            self.counts[words[0]].total -=1
        else:
            # continue adding the rest of the words through layers of the nGram
            self.counts[words[0]].sub(words[1:])
        self.total +=1
#
# Functions
#
def filePreprocessor(string):
    # Remove text up to and including these strings
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
    return re.sub(regex, '', string.lower())

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
    return ngrams

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
def findPerplexity(bigrams, unigrams, cmap, file_paths, include_unigram=False):
    unilogsum = 0
    bilogsum = 0
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
                unipwgw1 = unigrams.counts[token].total/unigrams.total
                unilogsum -= math.log(unipwgw1)
                bipwgw1 = readCmap(bigrams.counts[tokenIdx_1].counts[token].total, cmap)/bigrams.counts[tokenIdx_1].smoothTotal
                bilogsum -= math.log(bipwgw1)
                divisor += 1
    if include_unigram:
        return (math.exp(1/divisor * unilogsum),math.exp(1/divisor * bilogsum))
    else:
        return math.exp(1/divisor * bilogsum)


def generateUnigramBigram():
    category_paths =  glob.glob('data_corrected/classification_task/*')
    unigramCollection = {}
    bigramCollection = {}
    cmapCollection = {}
    # category_paths = [category_paths[0]]
    for category in category_paths:
        # BEGIN iteration over categories
        category_title = category.split('/')[-1]
        if category_title == "test_for_classification":
            # non standard category, don't do this one
            continue
        print("---")
        print("------------",category_title,"------------")
        print("---")

        unigrams = NGram(1)
        bigrams = NGram(2)

        file_paths = glob.glob(category+"/train_docs/*.txt")
        # file_paths = [file_paths[0]]
        # BEGIN iteration over paths
        threefourthsmark = int(len(file_paths)*3/4)

        training_file_paths = file_paths[:threefourthsmark]
        test_file_paths = file_paths[threefourthsmark:]

        unigrams = collectionIterator(training_file_paths,unigrams)
        bigrams = collectionIterator(training_file_paths,bigrams,unigrams)

        # Section 3: generate fragments

        # print("-------- Unigram ----------")
        # for i in range(15):
        #     print(unigramGenSentence(unigrams, "<s>"))
        #
        # print("-------- Bigram ----------")
        # for i in range(15):
        #     print(bigramGenSentence(bigrams, "<s>"))

        # section 4 do good turning and add unknowns
        mappedUnigrams = removeUnk(unigrams)
        # Now post process data for smoothing
        cmap, mappedBigrams = smoothGoodTuring(bigrams)

        # section 5 compute Perplexity
        uni_perplexity, bi_perplexity = findPerplexity(mappedBigrams, mappedUnigrams, cmap, test_file_paths, True)
        print ("and the perplexity is ", uni_perplexity,bi_perplexity)

        unigramCollection[category_title] = mappedUnigrams
        bigramCollection[category_title] = mappedBigrams
        cmapCollection[category_title] = cmap


    return (unigramCollection, bigramCollection, cmapCollection)

#
# Section 6 functions: for classification
#
def testNgramModel(uCollect, biCollect, cmapCollect):
    category_paths =  glob.glob('data_corrected/classification_task/*')
    categories = sorted(uCollect.keys())
    category_accuracy = []

    for category in category_paths:
        accurate = 0
        inaccurate = 0
        category_title = category.split('/')[-1]
        if category_title == "test_for_classification":
            # non standard category, don't do this one
            continue

        file_paths = glob.glob(category+"/train_docs/*.txt")

        threefourthsmark = int(len(file_paths)*3/4)

        training_file_paths = file_paths[:threefourthsmark]
        validation_file_paths = file_paths[threefourthsmark:]

        for file_path in validation_file_paths:
            perplexity_set = {}
            for category in categories:
                perplexity_set[category] = findPerplexity(biCollect[category], uCollect[category], cmapCollect[category], [file_path])

            prediction_category_order = sorted(perplexity_set.items(), key=operator.itemgetter(1))
            predicted_category, cat_perplex = prediction_category_order[0]
            if predicted_category == category_title:
                accurate +=1
            else:
                inaccurate +=1

        accuracy = accurate/(accurate+inaccurate)
        print accuracy
        category_accuracy.append((category_title, accuracy))
    return category_accuracy


def createTestNgramModelOutput(uCollect, biCollect, cmapCollect):
    predictions = []
    test_files_paths = glob.glob('data_corrected/classification_task/test_for_classification/*.txt')

    categories = sorted(uCollect.keys())
    category_to_id_map = {}
    for index,category in enumerate(categories):
        category_to_id_map[category] = index

    text_file = open("kaggle_prediction.csv", "w")
    text_file.write("Id,Prediction\n")

    for file_path in test_files_paths:
        perplexity_set = {}
        for category in categories:
            perplexity_set[category] = findPerplexity(biCollect[category], uCollect[category], cmapCollect[category], [file_path])

        prediction_category_order = sorted(perplexity_set.items(), key=operator.itemgetter(1))
        adjusted_file_path = file_path.split("/")[-1]
        predicted_category, cat_perplex = prediction_category_order[0]
        category_prediction = [adjusted_file_path, category_to_id_map[predicted_category]]
        predictions.append(category_prediction)
        text_file.write(",".join(str(s) for s in category_prediction)+"\n")

    return predictions

#
# Section 7 functions: for spell checking
#
# read confusion set from file
def getConfusionSet(file_path):
    text_file = open(file_path, "r")
    lines = text_file.read().decode("utf-8-sig").encode("utf-8").splitlines()
    return [[x for x in line.strip().split(" ") if x] for line in lines]

# generate forward bigrams
def generateSpellcheckSet(confusion_set_check, spellcheck_category_paths):
    sp_bigram_col = {}

    for sc_category_path in spellcheck_category_paths:
        category_title = sc_category_path.split('/')[-1]


        file_paths = glob.glob(sc_category_path+"/train_docs/*.txt")

        sp_bigram = SPNGram(2)
        sp_bigram_after = SPNGram(2)

        threefourthsmark = int(len(file_paths)*3/4)

        training_file_paths = file_paths[:threefourthsmark]
        validation_file_paths = file_paths[threefourthsmark:]

        for index,path in enumerate(training_file_paths):
            raw = open(path, 'r').read()
            path_list = path.split("/")
            path_list_split = path_list[-1].split(".")
            path_list_split[0] += "_modified"
            path_list[-2] = "train_modified_docs"
            path_list[-1] = ".".join(path_list_split)
            path_mod = "/".join(path_list)
            raw_mod = open(path_mod,'r').read()
            processed_file = filePreprocessor(raw)
            processed_file_mod = filePreprocessor(raw_mod)

            sen_tokens = sent_tokenize(processed_file)
            sen_tokens_mod = sent_tokenize(processed_file_mod)

            for i,sen in enumerate(sen_tokens):
                # might need to point </s> to <s> for bigrams

                sentence = stringPreprocessor(sen)
                sentence_mod = stringPreprocessor(sen_tokens_mod[i])

                tokens = word_tokenize(sentence)
                tokens_mod = word_tokenize(sentence_mod)

                # print tokens
                # print "----------------\n"
                # print tokens_mod

                for idx, token in enumerate(tokens):
                    token_mod = tokens_mod[idx]

                    if token == token_mod:
                        # this word is not in the confusion set and we don't care about it
                        continue


                    tokenIdx_1 = "<s>"
                    tokenIdx_1_mod = "<s>"
                    if idx != 0:
                        # print(idx)
                        tokenIdx_1 = tokens[idx-1]
                        tokenIdx_1_mod = tokens_mod[idx-1]
                        if tokenIdx_1_mod != tokenIdx_1:
                            # replace the wrong word with the right word for n-1
                            tokenIdx_1_mod = tokenIdx_1

                    tokenIdx_p1 = "</s>"
                    tokenIdx_p1_mod = "</s>"
                    if idx != len(tokens)-1:
                        tokenIdx_p1 = tokens[idx+1]
                        tokenIdx_p1_mod = tokens_mod[idx+1]
                        if tokenIdx_p1_mod != tokenIdx_p1:
                            # replace the wrong word with the right word for n+1
                            tokenIdx_p1_mod = tokenIdx_p1

                    sp_bigram.add([tokenIdx_1,token])
                    sp_bigram.sub([tokenIdx_1,token_mod])

                    sp_bigram_after.add([tokenIdx_p1,token])
                    sp_bigram_after.sub([tokenIdx_p1,token_mod])

        sp_bigram_col[category_title] = (sp_bigram, sp_bigram_after)
    return sp_bigram_col

def guessSPWord(word1, wordOptions, sp_gram, sp_gram_after, after_scalar):
    word = word1
    for word2 in wordOptions:
        if (sp_gram.counts[word2].total + after_scalar*sp_gram_after.counts[word2].total) > (sp_gram.counts[word].total + after_scalar*sp_gram_after.counts[word].total):
            word = word2
    return word

def checkSpellCheck(confusion_set_check, spellcheck_category_paths, sp_bigram_col, after_scalar):
    accuracies = {}
    for sc_category_path in spellcheck_category_paths:
        category_title = sc_category_path.split('/')[-1]


        file_paths = glob.glob(sc_category_path+"/train_docs/*.txt")
        twothirdsmark = int(len(file_paths)*2/3)

        training_file_paths = file_paths[:twothirdsmark]
        validation_file_paths = file_paths[twothirdsmark:]

        sp_bigram, sp_bigram_after = sp_bigram_col[category_title]
        category_accuracy= 0
        category_inaccuracy = 0

        for index,path in enumerate(validation_file_paths):
            raw = open(path, 'r').read()
            path_list = path.split("/")
            path_list_split = path_list[-1].split(".")
            path_list_split[0] += "_modified"
            path_list[-2] = "train_modified_docs"
            path_list[-1] = ".".join(path_list_split)
            path_mod = "/".join(path_list)
            raw_mod = open(path_mod,'r').read()
            processed_file = filePreprocessor(raw)
            processed_file_mod = filePreprocessor(raw_mod)

            sen_tokens = sent_tokenize(processed_file)
            sen_tokens_mod = sent_tokenize(processed_file_mod)

            for i,sen in enumerate(sen_tokens):

                sentence = stringPreprocessor(sen)
                sentence_mod = stringPreprocessor(sen_tokens_mod[i])

                tokens = word_tokenize(sentence)
                tokens_mod = word_tokenize(sentence_mod)

                sentence_guesses = {}
                for idx, token in enumerate(tokens):
                    token_mod = tokens_mod[idx]

                    if token == token_mod:
                        # this word is not in the confusion set and we don't care about it
                        continue


                    tokenIdx_1 = "<s>"
                    tokenIdx_1_mod = "<s>"
                    if idx != 0:
                        tokenIdx_1 = tokens[idx-1]
                        tokenIdx_1_mod = tokens_mod[idx-1]
                        if tokenIdx_1 != tokenIdx_1_mod:
                            # replace the wrong word with the right word for n-1
                            tokenIdx_1_mod = sentence_guesses[idx-1]

                    tokenIdx_p1_mod = "</s>"
                    if idx != len(tokens)-1:
                        tokenIdx_p1_mod = tokens_mod[idx+1]

                    a_scalar = after_scalar
                    if tokenIdx_p1_mod in confusion_set_check:
                        a_scalar = 0

                    word_guess = guessSPWord(token_mod, confusion_set_check[token_mod], sp_bigram.counts[tokenIdx_1_mod], sp_bigram_after.counts[tokenIdx_p1_mod], a_scalar)
                    sentence_guesses[idx] = word_guess

                    if word_guess == token:
                        category_accuracy +=1
                    else:
                        category_inaccuracy +=1
        accuracies[category_title] = category_accuracy/(category_accuracy+category_inaccuracy)
    print accuracies
    return accuracies

def testDataSpellCheck(confusion_set_check, spellcheck_category_paths, sp_bigram_col, after_scalar):
    accuracies = {}
    for sc_category_path in spellcheck_category_paths:
        category_title = sc_category_path.split('/')[-1]


        file_paths = glob.glob(sc_category_path+"/test_modified_docs/*.txt")

        sp_bigram, sp_bigram_after = sp_bigram_col[category_title]

        for index,path in enumerate(file_paths):
            raw = open(path, 'r').read()
            path_list = path.split("/")
            path_list_split = path_list[-1].split(".")
            path_list_split[0] = "_".join(path_list_split[0].split("_")[:-1])
            path_list[-2] = "test_docs"
            path_list[-1] = ".".join(path_list_split)
            path_mod = "/".join(path_list)
            output = open(path_mod,'w')

            sen_tokens = sent_tokenize(raw)
            edits = []

            for i,sen in enumerate(sen_tokens):

                sentence = stringPreprocessor(sen)

                tokens = word_tokenize(sentence)

                for idx, token in enumerate(tokens):

                    if token not in confusion_set_check:
                        # this word is not in the confusion set and we don't care about it
                        continue


                    tokenIdx_1 = "<s>"
                    if idx != 0:
                        tokenIdx_1 = tokens[idx-1]

                    tokenIdx_p1 = "</s>"
                    if idx != len(tokens)-1:
                        tokenIdx_p1= tokens[idx+1]

                    a_scalar = after_scalar
                    if tokenIdx_p1 in confusion_set_check:
                        a_scalar = 0

                    word_guess = guessSPWord(token, confusion_set_check[token], sp_bigram.counts[tokenIdx_1], sp_bigram_after.counts[tokenIdx_p1], a_scalar)

                    edits.append((token, word_guess))

                    tokens[idx] = word_guess

            raw_array = raw.split(" ")
            edit_index = 0
            word, replacement = edits[edit_index]
            for write_idx,tken in enumerate(raw_array):
                # print("looking at token "+tken +" comparing to edit: "+word)
                if tken.lower() == word:
                    if tken.isupper():
                        raw_array[write_idx] = replacement.capitalize()
                    else:
                        raw_array[write_idx] = replacement

                    edit_index +=1
                if edit_index >= len(edits):
                    break
                word, replacement = edits[edit_index]
            output.write(" ".join(raw_array))
            output.close()
#
# Raw Code
#

# # Needed for initial run
# nltk.download('punkt')

# Initialization
# sections 1-5 are iterated through generateUnigramBigram
uCollect, biCollect, cmapCollect = generateUnigramBigram()

# section 6 compute topic classification
# test_predictions = testNgramModel(uCollect, biCollect, cmapCollect)


# section 7
# NOTE: we only want letters here, so we can remove all punctuation.  We can also just look at lower case forms, something we should implement for the previous case as well.
# General approach: train bigrams for both sides of the word?

# spellcheck_category_paths =  glob.glob('data_corrected/spell_checking_task/*')
# if 'data_corrected/spell_checking_task/confusion_set.txt' in spellcheck_category_paths: spellcheck_category_paths.remove('data_corrected/spell_checking_task/confusion_set.txt')
#
# confusion_set = getConfusionSet('data_corrected/spell_checking_task/confusion_set.txt')
# confusion_set_check = defaultdict(set)
# for word_set in confusion_set:
#     confusion_set_check[word_set[0]].add(word_set[1])
#     confusion_set_check[word_set[1]].add(word_set[0])
# sp_bigram_col = generateSpellcheckSet(confusion_set_check, spellcheck_category_paths)
#
# # checkSpellCheck(confusion_set_check,spellcheck_category_paths, sp_bigram_col, 5)
# testDataSpellCheck(confusion_set_check,spellcheck_category_paths, sp_bigram_col, 5)
