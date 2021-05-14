import numpy as np
import argparse
import json
import string 
import sys
import re
import csv
import os

featdir = '/u/cs401/A1/feats/'
wordlistdir = '/u/cs401/Wordlists'

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs', 'they\'re', 'he\'s', 'she\'s'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

CLASS = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}


def getWordList(comm):
    toks = re.sub(r"(\S+)\/(\S+)", r"\1", comm).split(" ")
    return toks
    
def upperCase(comm):
    tok_list = getWordList(comm)
    cnt = 0
    for i in tok_list:
        if i.isupper() and len(i)>3:
            cnt = cnt + 1
    return cnt

# function to turn all text to lowercase but keeping PoS tags uppercase
def turnLower(comm):
    temp = ""
    copy = " " + comm
    for i in range(1, len(copy)):
        if copy[i].isupper() and copy[i-1] == " ":
            temp = temp + copy[i].lower() 
        else: 
            temp = temp + copy[i]
    temp = temp.strip()
    return temp

def firstSecondThirdSlangTok(comm):
    result = [0, 0, 0, 0, 0]

    token_list = getWordList(comm)
    sum_tok_len = 0

    for i in token_list:
        if i in FIRST_PERSON_PRONOUNS:
            result[0] = result[0] + 1
        if i in SECOND_PERSON_PRONOUNS:
            result[1] = result[1] + 1
        if i in THIRD_PERSON_PRONOUNS:
            result[2] = result[2] + 1
        if i in SLANG:
            result[3] = result[3] + 1
        sum_tok_len = sum_tok_len + len(i)

    result[4] = sum_tok_len/len(token_list)

    return result

#function to extract features of 
def PoSFeatures(comm):
    # index 0-8 of result will contain the count of steps 5-13
    result = [0]*9
    copy = comm

    #coordinating conjunctions
    result[0] = copy.count("/CC")
    
    #past tense verbs
    result[1] = copy.count("/VBD")

    # commas
    result[3] = copy.count(",/")

    # future tense verbs
    temp = copy.split(" ")
    first = ["'ll", "will", "gonna", "going/VBG to/TO"]
    for i in range(0, len(temp)-2):
        if any(temp[i] in s for s in first):
            if "VB" in temp[i+1]:
                result[2] = result[2] + 1

    # multichar punctuations
    for word in temp:
        if "/:" in word:
            if len(word)>3:
                result[4] = result[4] + 1

    # common nouns
    result[5] = copy.count("/NN") + copy.count("/NNS")

    # propernouns
    result[6] = copy.count("/NNP") + copy.count("/NNPS")

    # adverbs
    result[7] = copy.count("/RB") + copy.count("/RBR") + copy.count("/RBS")

    # wh- words
    result[8] = copy.count("/WDT") + copy.count("/WP") + copy.count("/WP$") + copy.count("/WRB")

    return result

def exBLGFeats(comm):
    aoa = []
    img = []
    fam = []
    toks = getWordList(comm)
    zeros = True

    for token in toks:
        if token in BGLFeats.keys():
            zeros = False
            aoa = aoa + [BGLFeats[token]["AoA"]]
            img = img + [BGLFeats[token]["IMG"]]
            fam = fam + [BGLFeats[token]["FAM"]]
    
    if zeros:
        return np.zeros((6,))
    
    aoa = np.array(aoa)
    img = np.array(img)
    fam = np.array(fam)

    return [aoa.mean(), img.mean(), fam.mean(), aoa.std(), img.std(), fam.std()]

def exWarrFeats(comm):
    v = []
    a = []
    d = []
    toks = getWordList(comm)
    zeros = True

    for tok in toks:
        if tok in WarrFeats.keys():
            zeros = False
            v = v + [WarrFeats[tok]["V.Mean.Sum"]]
            a = a + [WarrFeats[tok]["A.Mean.Sum"]]
            d = d + [WarrFeats[tok]["D.Mean.Sum"]]
    
    if zeros:
        return np.zeros((6,))

    v = np.array(v)
    a = np.array(a)
    d = np.array(d)

    return [v.mean(), a.mean(), d.mean(), v.std(), a.std(), d.std()]

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    feats = np.zeros((174))

    feats[0] = upperCase(comment)
    comment = turnLower(comment)

    feat_2to4_1415 = firstSecondThirdSlangTok(comment)
    feats[1] = feat_2to4_1415[0] 
    feats[2] = feat_2to4_1415[1]
    feats[3] = feat_2to4_1415[2]
    feats[13] = feat_2to4_1415[3]
    feats[15] = feat_2to4_1415[4]
    
    #feat[4] to feat[12]
    posfeat = PoSFeatures(comment)
    cnt = 4
    for i in posfeat:
        feats[cnt] = i
        cnt = cnt + 1

    #finding number of sentences
    temp = comment
    temp = temp.split("/.")
    feats[16] = len(temp)

    # ave length of sentences
    word_sum = 0
    for sent in temp:
        toks_split = sent.split(" ")
        word_sum = word_sum + len(toks_split)
    feats[14] = word_sum/len(temp)
    
    blgfeats = list(exBLGFeats(comment))
    warfeats = list(exWarrFeats(comment))
    cnt = 17
    for j in blgfeats:
        feats[cnt] = j
        cnt = cnt + 1
    
    cnt_2 = 23
    for h in warfeats:
        feats[cnt_2] = h
        cnt_2 = cnt_2 +1 
    return feats
    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    
    feat[29:-1] = idtoFeats[comment_id]
    feat[-1] = CLASS[comment_class]
    return feat

def buildLIWCFeats():
    res = {}
    for cat in ["Left", "Center", "Right", "Alt"]:
        idspath = os.path.join(featdir, f"{cat}_IDs.txt")
        with open(idspath, "r") as id_file:
            pathLIWC = os.path.join(featdir, f"{cat}_feats.dat.npy")
            featLIWC = np.load(pathLIWC)

            #match comment ids with LIWC rows
            for (row_num, id_line) in enumerate(id_file.readlines()):
                res[id_line.strip()] = featLIWC[row_num]
    return res

idtoFeats = buildLIWCFeats()

def main(args):
    #Declare necessary global variables here. 
    global BGLFeats, WarrFeats, idtoFeats

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    BGLFeats = {row["WORD"]: {
                    "AoA": float(row["AoA (100-700)"]),
                    "IMG": float(row["IMG"]),
                    "FAM": float(row["FAM"])
                    }
                    for row in csv.DictReader(open(os.path.join(wordlistdir, "BristolNorms+GilhoolyLogie.csv")))
                        if ((row["AoA (100-700)"] != "") or (row["IMG"] != "") or (row["FAM"] != ""))
                }

    WarrFeats = {row["Word"]: {
                    "V.Mean.Sum": float(row["V.Mean.Sum"]),
                    "A.Mean.Sum": float(row["A.Mean.Sum"]),
                    "D.Mean.Sum": float(row["D.Mean.Sum"])
                    }
                    for row in csv.DictReader(open(os.path.join(wordlistdir, "Ratings_Warriner_et_al.csv")))
                        if ((row["V.Mean.Sum"] != "") or (row["A.Mean.Sum"] != "") or (row["D.Mean.Sum"] != ""))
                }

    for (row_num, comment) in enumerate(data):
        comment = json.loads(comment)

        feats[row_num] = extract1(comment["body"])
        feats[row_num] = extract2(feats[row_num], comment["cat"], comment["id"])

        print(CLASS[comment["cat"]])

    print("Type feats: ", type(feats))
    print("Type rows: ", type(feats[1]))
    print("Type rows at 1", type(feats[1][1]))
    print("shape of feats: ", feats.shape)
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

