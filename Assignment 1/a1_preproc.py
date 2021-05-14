import sys
import argparse
import os
import json
import re
import spacy
import unicodedata
import html
from spacy.tokens import Doc 
import string

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

#def punctuations(comm)

    
def preproc(data, category):
    for i in range(0, len(data)):
        line = json.loads(data[i])

        print("~~~Original: ")
        print(line['body'])

        preproc_bod = preproc1(line['body'])

        print("!!!parsed: ")
        print(preproc_bod.encode('unicode_escape').decode('utf-8'))

        line['body'] = preproc_bod
        line['cat'] = category

        data[i] = json.JSONEncoder().encode(line)

    return data

def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  
        #modify this to handle other whitespace chars.
        #replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = re.sub(r'\s+', " ", modComm)
        modComm = modComm.replace('\\s+', " ")
        modComm = modComm.strip()
        print ("after step 1: ", modComm)

    if 2 in steps:  # unescape html
        modComm = modComm.strip()
        temp = html.unescape(modComm)
        modComm = unicodedata.normalize("NFD", temp).encode("ascii", "ignore").decode("utf-8").encode("ascii", "ignore").decode()
        
        print("after step 2: ", modComm)
        if (modComm == ""):
            return ""

    if 3 in steps:  # remove URLs
        modComm = modComm.strip()
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        print("after step 3: ", modComm)

        if (modComm == ""):
            return ""
        
    if 4 in steps: #remove duplicate spaces.

        modComm = modComm.strip()
        modComm = re.sub(r"\s+", " ", modComm) 
        print("after step 4: ", modComm)

        if (modComm == ""):
            return ""

    if 5 in steps:
        # TODO: get Spacy document for modComm
        
        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
        tag_line = ""
        
        doc = nlp(modComm)

        upper = False
        for i, token in enumerate(doc):
            upper = token.text.isupper()
            if ((str(token.lemma_)[0] != '-') or (str(token.lemma_)[0] == '-' and token.text[0] =='-')):
                if (upper == True):
                    tag_line = tag_line + token.lemma_.upper() + '/' + token.tag_
                else:
                    tag_line = tag_line + token.lemma_.lower() + '/' + token.tag_
            else: 
                if (upper == True):
                    tag_line = tag_line + token.text.upper() + '/' + token.tag_
                else:
                    tag_line = tag_line + token.text.lower() + '/' + token.tag_
            if ((token.text == '.') or (token.text == '!') or (token.text == '?') or (i == len(doc) - 1)):
                tag_line = tag_line + '\n '
            else: 
                tag_line = tag_line + " "

        modComm = tag_line
        print("after step 5.1, 5.2 and 5.3: ", modComm)

    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            #fullFile = "sample_in.json"
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            start = student_id % len(data)

            if (len(data)- (start + 1)<args.max):
                data = data[start : ] + data[0: args.max - len(data) + start]
            else:
                data = data[start : start + args.max]

            preprocData = preproc(data, file)
            allOutput = allOutput + preprocData

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    student_id = args.ID[0]

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
