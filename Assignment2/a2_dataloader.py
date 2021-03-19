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

'''Build Datasets for Hansards

Don't go publishing results based on this. We restrict a lot of things to make
this nicer.
'''

import locale
import os
import re
from string import punctuation
from collections import Counter
import gzip
import torch


TOKENIZER_PATTERN = re.compile(r'[' + re.escape(punctuation) + r'\d\s]+')

locale.setlocale(locale.LC_ALL, 'C')  # ensure reproducible sorting

__all__ = [
    'get_dir_lines',
    'build_vocab_from_dir',
    'word2id_to_id2word',
    'id2word_to_word2id',
    'write_word2id_to_file',
    'read_word2id_from_file',
    'get_common_prefixes',
    'HansardDataset',
    'HansardDataLoader',
]


def get_dir_lines(dir_, lang, filenames=None):
    '''Generate line info from data in a directory for a given language

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.
    lang : {'e', 'f'}
        Whether to tokenize the English sentences ('e') or French ('f').
    filenames : sequence, optional
        Only tokenize sentences with matching names. If :obj:`None`, searches
        the whole directory in C-sorted order.

    Yields
    ------
    tokenized, filename, offs : list
        `tokenized` is a list of tokens for a line. `filename` is the source
        file. `offs` is the start of the sentence in the file, to seek to.
        Lines are yielded by iterating over lines in each file in the order
        presented in `filenames`.
    '''
    _in_set_check('lang', lang, {'e', 'f'})
    lang = '.' + lang
    if filenames is None:
        filenames = sorted(os.listdir(dir_))
    for filename in filenames:
        if filename.endswith(lang):
            with open(os.path.join(dir_, filename)) as f:
                offs = f.tell()
                line = f.readline()
                while line:
                    yield [
                        w for w in TOKENIZER_PATTERN.split(line.lower()) if w
                    ], filename, offs
                    offs = f.tell()
                    line = f.readline()


def build_vocab_from_dir(train_dir_, lang, max_vocab=5000):
    '''Build a vocabulary (words->ids) from transcriptions in a directory

    Parameters
    ----------
    train_dir_ : str
        A path to the transcription directory. ALWAYS use the training
        directory, not the test, directory, when building a vocabulary.
    lang : {'e', 'f'}
        Whether to build the English vocabulary ('e') or the French one ('f').
    max_vocab : int, optional
        The size of your vocabulary. Words with the greatest count will be
        retained.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids. There will be an
        entry for each id between ``[0, max_vocab - 1]`` inclusive.
    '''
    _in_range_check('max_vocab', max_vocab, 3)
    word2count = Counter()
    for tokenized, _, _ in get_dir_lines(train_dir_, lang):
        word2count.update(tokenized)
    word2count = sorted(
        word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    word2count = word2count[:max_vocab - 3]
    return dict((v[0], i) for i, v in enumerate(word2count))


def word2id_to_id2word(word2id):
    '''word2id -> id2word'''
    return dict((v, k) for (k, v) in word2id.items())


def id2word_to_word2id(id2word):
    '''id2word -> word2id'''
    return dict((v, k) for (k, v) in id2word.items())


def write_word2id_to_file(word2id, file_):
    '''Write word2id map to a file

    Parameters
    ----------
    word2id : dict
        A dictionary of keys being words, values being ids
    file_ : str or file
        A file to write `word2id` to. If a path that ends with ``.gz``, it will
        be gzipped.
    '''
    if isinstance(file_, str):
        if file_.endswith('.gz'):
            with gzip.open(file_, mode='wt') as file_:
                return write_word2id_to_file(word2id, file_)
        else:
            with open(file_, 'w') as file_:
                return write_word2id_to_file(word2id, file_)
    id2word = word2id_to_id2word(word2id)
    for i in range(len(id2word)):
        file_.write('{} {}\n'.format(id2word[i], i))


def read_word2id_from_file(file_):
    '''Read word2id map from a file

    Parameters
    ----------
    file_ : str or file
        A file to read `word2id` from. If a path that ends with ``.gz``, it
        will be de-compressed via gzip.

    Returns
    -------
    word2id : dict
        A dictionary of keys being words, values being ids
    '''
    if isinstance(file_, str):
        if file_.endswith('.gz'):
            with gzip.open(file_, mode='rt') as file_:
                return read_word2id_from_file(file_)
        else:
            with open(file_) as file_:
                return read_word2id_from_file(file_)
    ids = set()
    word2id = dict()
    for line in file_:
        line = line.strip()
        if not line:
            continue
        word, id_ = line.split()
        id_ = int(id_)
        if id_ in ids:
            raise ValueError(f'Duplicate id {id_}')
        if word in word2id:
            raise ValueError(f'Duplicate word {word}')
        ids.add(id_)
        word2id[word] = id_
    _word2id_validity_check('word2id', word2id)
    return word2id


def get_common_prefixes(dir_):
    '''Return a list of file name prefixes common to both English and French

    A prefix is common to both English and French if the files
    ``<dir_>/<prefix>.e`` and ``<dir_>/<prefix>.f`` both exist.

    Parameters
    ----------
    dir_ : str
        A path to the transcription directory.

    Returns
    -------
    common : list
        A C-sorted list of common prefixes
    '''
    all_fns = os.listdir(dir_)
    english_fns = set(fn[:-2] for fn in all_fns if fn.endswith('.e'))
    french_fns = set(fn[:-2] for fn in all_fns if fn.endswith('.f'))
    del all_fns
    common = english_fns & french_fns
    if not common:
        raise ValueError(
            f'Directory {dir_} contains no common files ending in .e or '
            f'.f. Are you sure this is the right directory?')
    return sorted(common)


class HansardDataset(torch.utils.data.Dataset):
    '''A dataset of a partition of the Canadian Hansards

    Indexes bitext sentence pairs ``F, E``, where ``F`` is the source language
    sequence and ``E`` is the corresponding target language sequence.

    Parameters
    ----------
    dir_ : str
        A path to the data directory
    french_word2id : dict or str
        Either a dictionary of French words to ids, or a path pointing to one.
    english_word2id : dict or str
        Either a dictionary of English words to ids, or a path pointing to one.
    source_language : {'e', 'f'}, optional
        Specify the language we're translating from. By default, it's French
        ('f'). In the case of English ('e'), ``F`` is still the source language
        sequence, but it now refers to English.
    prefixes : sequence, optional
        A list of file prefixes in `dir_` to consider part of the dataset. If
        :obj:`None`, will search for all common prefixes in the directory.

    Attributes
    ----------
    dir_ : str
    source_language : {'e', 'f'}
    source_unk : int
        A special id to indicate a source token was out-of-vocabulary.
    source_pad_id : int
        A special id used for right-padding source-sequences during batching
    source_vocab_size : int
        The total number of unique ids in source sequences. All ids are bound
        between ``[0, source_vocab_size - 1]`` inclusive. Includes
        `source_unk` and `source_pad_id`.
    target_unk : int
        A special id to indicate a target token was in-vocabulary.
    target_sos : int
        A special id to indicate the start of a target sequence. One SOS token
        is prepended to each target sequence ``E``.
    target_eos : int
        A special id to indicate the end of a target sequence. One EOS token
        is appended to each target sequence ``E``.
    target_vocab_size : int
        The total number of unique ids in target sequences. All ids are bound
        between ``[0, target_vocab_size - 1]`` inclusive. Includes
        `target_unk`, `target_sos`, and `target_eos`.
    pairs : tuple
    '''

    def __init__(
            self, dir_, french_word2id, english_word2id, source_language='f',
            prefixes=None):
        _in_set_check('source_language', source_language, {'e', 'f'})
        if isinstance(french_word2id, str):
            french_word2id = read_word2id_from_file(french_word2id)
        else:
            _word2id_validity_check('french_word2id', french_word2id)
        if isinstance(english_word2id, str):
            english_word2id = read_word2id_from_file(english_word2id)
        else:
            _word2id_validity_check('english_word2id', english_word2id)
        if prefixes is None:
            prefixes = get_common_prefixes(dir_)
        english_fns = (p + '.e' for p in prefixes)
        french_fns = (p + '.f' for p in prefixes)
        english_l = get_dir_lines(dir_, 'e', english_fns)
        french_l = get_dir_lines(dir_, 'f', french_fns)
        if source_language == 'f':
            source_word2id = french_word2id
            target_word2id = english_word2id
        else:
            source_word2id = english_word2id
            target_word2id = french_word2id
        pairs = []
        F_unk, F_pad = range(len(source_word2id), len(source_word2id) + 2)
        E_unk, E_sos, E_eos = range(
            len(target_word2id), len(target_word2id) + 3)
        for (e, e_fn, _), (f, f_fn, _) in zip(english_l, french_l):
            assert e_fn[:-2] == f_fn[:-2]
            if not e or not f:
                assert not e and not f  # if either is empty, both should be
                continue
            if source_language == 'f':
                F, E = f, e
            else:
                F, E = e, f
            F = torch.tensor([source_word2id.get(w, F_unk) for w in F])
            E = torch.tensor(
                [E_sos] + [target_word2id.get(w, E_unk) for w in E] + [E_eos])
            if torch.all(F == F_unk) and torch.all(E[1:-1] == E_unk):
                # skip sentences that are solely OOV
                continue
            pairs.append((F, E))
        self.dir_ = dir_
        self.source_language = source_language
        self.source_vocab_size = len(source_word2id) + 2  # pad id and unk
        self.source_unk = F_unk
        self.source_pad_id = F_pad
        self.target_unk = E_unk
        self.target_sos = E_sos
        self.target_eos = E_eos
        self.target_vocab_size = len(target_word2id) + 3  # unk, sos, and eos
        self.pairs = tuple(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


class HansardDataLoader(torch.utils.data.DataLoader):
    '''A DataLoader yielding batches of bitext

    Consult :class:`HansardDataset` for a description of parameters and
    attributes

    Parameters
    ----------
    dir_ : str
    french_word2id : dict or str
    english_word2id : dict or str
    source_language : {'e', 'f'}, optional
    prefixes : sequence, optional
    kwargs : optional
        See :class:`torch.utils.data.DataLoader` for additional arguments.
        Do not specify `collate_fn`.
    '''

    def __init__(
            self, dir_, french_word2id, english_word2id, source_language='f',
            prefixes=None, **kwargs):
        if 'collate_fn' in kwargs:
            raise TypeError(
                "HansardDataLoader() got an unexpected keyword argument "
                "'collate_fn'")
        dataset = HansardDataset(
            dir_, french_word2id, english_word2id, source_language, prefixes)
        super().__init__(dataset, collate_fn=self.collate, **kwargs)

    def collate(self, seq):
        F, E = zip(*seq)
        F_lens = torch.tensor([len(f) for f in F])
        F = torch.nn.utils.rnn.pad_sequence(
            F, padding_value=self.dataset.source_pad_id)
        E = torch.nn.utils.rnn.pad_sequence(
            E, padding_value=self.dataset.target_eos)
        return F, F_lens, E


def _in_range_check(
        name, value, low=-float('inf'), high=float('inf'),
        error=ValueError):
    if value < low:
        raise error(f'{name} ({value}) is less than {low}')
    if value > high:
        raise error(f'{name} ({value}) is greater than {high}')


def _in_set_check(name, value, set_, error=ValueError):
    if value not in set_:
        raise error(f'{name} not in {set_}')


def _word2id_validity_check(name, word2id, error=ValueError):
    if set(word2id.values()) != set(range(len(word2id))):
        raise error(
            f'Ids in {name} should be contiguous and span [0, len({name}) - 1]'
            f' inclusive')
