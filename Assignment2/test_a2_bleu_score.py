'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Unit tests for a2_bleu_score.py

These are example tests solely for your benefit and will not count towards
your grade.
'''

import pytest
import numpy as np
import a2_bleu_score


@pytest.mark.parametrize("ids", [True, False])
def test_bleu(ids):
    reference = '''\
it is a guide to action that ensures that the military will always heed
party commands'''.strip().split()
    candidate = '''\
it is a guide to action which ensures that the military always obeys the
commands of the party'''.strip().split()
    if ids:
        # should work with token ids (ints) as well
        reference = [hash(word) for word in reference]
        candidate = [hash(word) for word in candidate]
    assert np.isclose(
        a2_bleu_score.n_gram_precision(reference, candidate, 1),
        15 / 18,
    )
    assert np.isclose(
        a2_bleu_score.n_gram_precision(reference, candidate, 2),
        8 / 17,
    )
    assert np.isclose(
        a2_bleu_score.brevity_penalty(reference, candidate),
        1.
    )
    assert np.isclose(
        a2_bleu_score.BLEU_score(reference, candidate, 2),
        1 * ((15 * 8) / (18 * 17)) ** (1 / 2)
    )
