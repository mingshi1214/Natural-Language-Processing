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
