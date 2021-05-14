'''Compute BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Get all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    # create the return list of ngrams, a copy of the sequence, and a temporary string for the current n_gram
    ngrams = []
    temp_seq = seq
    n_gram = ""

    # iterate through each beginning of each n-gram
    for i in range(0, len(temp_seq)-n+1):

        # add the next words to each n_gram
        for j in range(0, n):
            n_gram = n_gram + " " + str(temp_seq[i+j])

        # add each n_gram to the list of n_grams
        n_gram = n_gram.strip()
        ngrams.append(n_gram)
        n_gram = ""

    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Compute the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to compute

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''

    p_n = 0

    # check for an empty candidate list
    if len(candidate)==0:
        return p_n

    can_ngrams = grouper(candidate, n)
    ref_ngrams = grouper(reference, n)

    # accumulate the number of matching n grams in candidate and reference
    C = 0
    for word in can_ngrams:
        if word in ref_ngrams:
            C = C + 1

    # check if the candidate n_grams list is 0.
    # if so, return 0
    N = len(can_ngrams)
    if N == 0:
        return 0

    # compute p_n
    p_n = C/N

    return p_n


def brevity_penalty(reference, candidate):
    '''Compute the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    BP = 0

    # compute r
    r = len(reference)/len(candidate)

    # compute BP
    if r < 1:
        return 1
    else:
        BP = exp(1-r)
    return BP


def BLEU_score(reference, candidate, n):
    '''Compute the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''

    # compute the p_scores using n_gram_precision for each n
    p_scores = 1
    for i in range (1, n+1):
        ngramPrec = n_gram_precision(reference, candidate, i)
        p_scores = p_scores * ngramPrec 

    bp = brevity_penalty(reference, candidate)

    # compute the BLEU score
    res = bp * (p_scores ** (1/n))

    return res
