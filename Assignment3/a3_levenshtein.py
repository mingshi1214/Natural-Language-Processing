import os
import numpy as np
import re
import sys
import string

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    # WER = (numsubs + numinserts + numdeletes) / numrefwords
    # make matrix
    # add s at beginning and /s to end
    # fill [0:end] along first row and col
    # for each ref word, for each hypothesis word, ... O(nm)

    cache = {}
    n = len(r)
    m = len(h)

    for i in range(0, n + 1, 1):
        for j in range(0, m + 1, 1):

            # target empty. delete everything in reference
            if j == 0:
                num_del = i
                num_ins = 0
                num_sub = 0

            # ref empty. make into target by inserting
            elif i == 0:
                num_del = 0
                num_ins = j
                num_sub = 0

            else:
                # r_head = r[:-1] (Deletion)
                sub_r, ins_r, del_r = cache[(i - 1, j)]
                r_dist = sum([sub_r, ins_r, del_r])  # Lev[i-1][j]

                # h_head = h[:-1] (Insertion)
                sub_h, ins_h, del_h  = cache[(i, j - 1)]
                h_dist = sum([sub_h, ins_h, del_h])  # Lev[i][j-1]

                # r_head and h_head
                sub_hr, ins_hr, del_hr  = cache[(i - 1, j - 1)]
                hr_dist = sum([sub_hr, ins_hr, del_hr])  # Lev[i-1][j-1]

                if r[i -1] == h[j - 1]:
                    const = 0
                else:
                    const = 1

                # find min for dist source
                delete = r_dist + 1
                insert = h_dist + 1
                subs = hr_dist + const
                lev_dist = min(delete, insert, subs)

                # deletion
                if lev_dist == delete:
                    num_del = del_r + 1
                    num_ins = ins_r
                    num_sub = sub_r

                # insertion
                elif lev_dist == insert:
                    num_del = del_h
                    num_ins = ins_h + 1
                    num_sub = sub_h

                elif lev_dist == subs:
                    # Carry forward deletion and insertion from r[i-1], h[j-1]
                    num_del = del_hr
                    num_ins = ins_hr
                    num_sub = (sub_hr) + const
                else:
                    print("something is wrong at [{}, {}]".format(i, j))

            # update dict
            cache[(i, j)] = (num_sub, num_ins, num_del)

    nS, nI, nD = cache[(n, m)]

    if n == 0:
        WER = float('inf')
    else:
        WER = (nS + nI + nD) / n

    return (WER, nS, nI, nD)


def preprocess(line):
    # remove all punctuation other than [and]
    # put all text to lowercase
    preproc = re.sub(r"[^a-zA-Z0-9\s\[\]]", r"", line)
    preproc = preproc.lower().strip().split()

    return preproc


if __name__ == "__main__":
    sys.stdout = open("asrDiscussion.txt", "w")

    googErr = []
    kaldiErr = []

    for root, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            rootPath = os.path.join(dataDir, speaker)
            refPath = os.path.join(rootPath, 'transcripts.txt')
            googPath = os.path.join(rootPath, 'transcripts.Google.txt')
            kaldiPath = os.path.join(rootPath, 'transcripts.Kaldi.txt')

            # open the files
            googFile = open(googPath, 'r')
            kaldiFile = open(kaldiPath, 'r')
            refFile = open(refPath, 'r')

            cnt = 0
            for goog_h, kaldi_h, ref in zip(googFile, kaldiFile, refFile):
                # preprocess the lines
                goog_h = preprocess(goog_h)
                kaldi_h = preprocess(kaldi_h)
                ref = preprocess(ref)

                # find errors of kaldi and google
                # add to the respective errors list
                # print result so we can write to file
                WER, nS, nI, nD = Levenshtein(ref, kaldi_h)
                kaldiErr.append(WER)
                print('{speaker} {system} {i} {wer} S:{nS}, I:{nI}, D:{nD}'.format(speaker = speaker, system = "Kaldi", i = cnt, wer = WER, nS = nS, nI = nI, nD = nD))
                
                WER, nS, nI, nD = Levenshtein(ref, goog_h)
                googErr.append(WER)
                print('{speaker} {system} {i} {wer} S:{nS}, I:{nI}, D:{nD}'.format(speaker = speaker, system = "Google", i = cnt, wer = WER, nS = nS, nI = nI, nD = nD))

                cnt += 1

            print("")

    googErr = np.array(googErr)
    kaldiErr = np.array(kaldiErr)

    g_mean = np.mean(googErr)
    k_mean = np.mean(kaldiErr)

    # this below also works for the record
    # g_std = np.sqrt(np.var(googErr))
    # k_std = np.sqrt(np.var(kaldiErr))
    g_std = np.std(googErr)
    k_std = np.std(kaldiErr)

    print("Google has a mean of: ", g_mean, " and a standard deviation of: ", g_std, " . Kaldi has a mean of :", k_mean, " and a standard deviation of: ", k_std)
