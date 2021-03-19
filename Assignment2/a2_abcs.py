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

'''Abstract base classes for building seq2seq models'''

import platform
import abc
import torch
import warnings

BAD_ENV = '''\
It appears you're using an environment that doesn't match teach. Your code will
be run in an environment matching that of 'xxx@teach.cs.toronto.edu'. If your
code fails to run there, you'll get no pity marks! You've been warned!

Alternatively, you might be on teach, but called 'python3' instead of
'python3.7'. Use the latter!
'''
if (
        platform.python_version() != '3.7.4' or
        not torch.__version__.startswith('1.2.0')):
    warnings.warn(BAD_ENV)


__all__ = [
    'EncoderBase',
    'DecoderBase',
    'EncoderDecoderBase',
]

warnings.simplefilter('always', UserWarning)


class EncoderBase(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Encode an input source target sequence into a state sequence

    See :func:`__init__` and :func:`init_submodules` for a description of the
    attributes.

    Attributes
    ----------
    source_vocab_size : int
    pad_id : int
    word_embedding_size : int
    num_hidden_layers : int
    hidden_state_size : int
    dropout : float
    cell_type : {'rnn', 'lstm', 'gru'}
    embedding : torch.nn.Embedding
    rnn : {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
    '''

    def __init__(
            self, source_vocab_size, pad_id=-1, word_embedding_size=1024,
            num_hidden_layers=2, hidden_state_size=512, dropout=0.1,
            cell_type='lstm'):
        '''Initialize the encoder

        Sets some non-parameter attributes, then calls :func:`init_submodules`.

        Parameters
        ----------
        source_vocab_size : int
            The number of words in your source language vocabulary, including
            `pad_id`
        pad_id : int, optional
            The index within `source_vocab_size` which is used to right-pad
            shorter input to the length of the longest input in the batch.
            Negative values between ``-1`` and ``-vocab_size`` inclusive are
            converted to positive indices by ``pad_id' = vocab_size + pad_id``.
        word_embedding_size : int, optional
            The size of your static (source) word embedding vectors.
        num_hidden_layers : int, optional
            The number of stacked recurrent layers in your encoder.
        hidden_state_size : int, optional
            The size of the output of a recurrent layer for one slice of time
            in one direction.
        dropout : float, optional
            The probability of applying dropout to hidden states in the RNN.
        cell_type : {'rnn', 'lstm', 'gru'}, optional
            What underlying recurrent architecture to use when building the
            `rnn` submodule. See :func:`init_submodules` for more info
        '''
        _in_range_check('source_vocab_size', source_vocab_size, 2)
        if -source_vocab_size <= pad_id < 0:
            pad_id = source_vocab_size + pad_id
        else:
            _in_range_check(
                'pad_id', pad_id, -source_vocab_size, source_vocab_size - 1)
        _in_range_check('word_embedding_size', word_embedding_size, 1)
        _in_range_check('num_hidden_layers', num_hidden_layers, 1)
        _in_range_check('hidden_state_size', hidden_state_size, 1)
        _in_range_check('dropout', dropout, 0, 1)
        _in_set_check('cell_type', cell_type, {'rnn', 'lstm', 'gru'})
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.pad_id = pad_id
        self.word_embedding_size = word_embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout
        self.cell_type = cell_type
        self.embedding = self.rnn = None
        self.init_submodules()
        assert self.embedding is not None, 'initialize embedding!'
        assert self.rnn is not None, 'initialize rnn!'

    @abc.abstractmethod
    def init_submodules(self):
        '''Initialize the parameterized submodules of this network

        This method sets the following object attributes (sets them in
        `self`):

        embedding : torch.nn.Embedding
            A layer that extracts learned token embeddings for each index in
            a token sequence. It must not learn an embedding for padded tokens.
        rnn : {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
            A layer corresponding to the recurrent neural network that
            processes source word embeddings. It must be bidirectional.
        '''
        raise NotImplementedError()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.rnn.reset_parameters()

    def check_input(self, F, F_lens):
        _dim_check('F', F, 2)
        _dim_check('F_lens', F_lens, 1)
        if torch.any((F < 0) | (F >= self.source_vocab_size)):
            raise RuntimeError(
                f'F values must be between '
                f'[0, {self.source_vocab_size - 1}]')
        if torch.any((F_lens > F.shape[0]) | (F_lens < 1)):
            raise RuntimeError(
                f'F_lens for F of shape ({F.shape[0]}, ...) must be '
                f'between [0, {F.shape[0]}]')
        if F_lens.max() != F.shape[0]:
            raise RuntimeError(
                f'The maximum value in F_lens ({F_lens.max()}) does not '
                f'equal the sequence dimension of F ({F.shape[0]})')
        pad_mask = torch.arange(F.shape[0], device=F.device).unsqueeze(-1)
        pad_mask = pad_mask >= F_lens  # (S, N)
        if not torch.all(F.masked_select(pad_mask) == self.pad_id):
            raise ValueError(
                f'Values in F past F_lens are not padding ({self.pad_id})')
        if torch.any(F.masked_select(~pad_mask) == self.pad_id):
            raise ValueError(
                f'Some values in F before F_lens are not padding '
                f'({self.pad_id})')

    def forward(self, F, F_lens, h_pad=0.):
        self.check_input(F, F_lens)
        return self.forward_pass(F, F_lens, h_pad=h_pad)

    @abc.abstractmethod
    def forward_pass(self, F, F_lens, h_pad=0.):
        '''Defines the structure of the encoder

        Parameters
        ----------
        F : torch.LongTensor
            An integer tensor of shape ``(S, M)``, where ``S`` is the number of
            source time steps and ``M`` is the batch dimension. ``F[s, m]``
            is the token id of the ``s``-th word in the ``m``-th source
            sequence in the batch. ``F`` has been right-padded with
            ``self.pad_id`` wherever ``S`` exceeds the length of the original
            sequence.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` that stores the original
            lengths of each source sequence (and input sequence) in the batch
            before right-padding.
        h_pad : float
            The value to right-pad `h` with, wherever `x` is right-padded.

        Returns
        -------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.hidden_state_size)``
            where ``h[s,m,i]`` refers to the ``i``-th index of the encoder
            RNN's last layer's hidden state at time step ``s`` of the
            ``m``-th sequence in the batch. The 2 is because the forward and
            backward hidden states are concatenated. If
            ``x[s,m] == 0.``, then ``h[s,m, :] == h_pad``
        '''
        raise NotImplementedError()

    def get_all_rnn_inputs(self, F):
        '''Get all input vectors to the RNN at once

        Parameters
        ----------
        F : torch.LongTensor
            An integer tensor of shape ``(S, M)``, where ``S`` is the number of
            source time steps and ``M`` is the batch dimension. ``F[s, m]``
            is the token id of the ``s``-th word in the ``m``-th source
            sequence in the batch. ``F`` has been right-padded with
            ``self.pad_id`` wherever ``S`` exceeds the length of the original
            sequence.

        Returns
        -------
        x : torch.FloatTensor
            A float tensor of shape ``(S, M, I)`` of input to the encoder RNN,
            where ``I`` corresponds to the size of the per-word input vector.
            Whenever ``s`` exceeds the original length of ``F[s, m]`` (i.e.
            when ``F[s, m] == self.pad_id``), ``x[s, m, :] == 0.``
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_all_hidden_states(self, x, F_lens, h_pad):
        '''Get all encoder hidden states for from input sequences

        Parameters
        ----------
        x : torch.FloatTensor
            A float tensor of shape ``(S, M, I)`` of input to the encoder RNN,
            where ``S`` is the number of source time steps, ``M`` is the batch
            dimension, and ``I`` corresponds to the size of the per-word input
            vector. ``x[s, m, :]`` is the input vector for the ``s``-th word in
            the ``m``-th source sequence in the batch. `x` has been padded such
            that ``x[F_lens[m]:, m, :] == 0.`` for all ``m``.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` that stores the original
            lengths of each source sequence (and input sequence) in the batch
            before right-padding.
        h_pad : float
            The value to right-pad `h` with, wherever `x` is right-padded.

        Returns
        -------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.hidden_state_size)``
            where ``h[s,m,i]`` refers to the ``i``-th index of the encoder
            RNN's last layer's hidden state at time step ``s`` of the
            ``m``-th sequence in the batch. The 2 is because the forward and
            backward hidden states are concatenated. If
            ``x[s,m] == 0.``, then ``h[s,m, :] == h_pad``
        '''
        raise NotImplementedError()


class DecoderBase(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Decode source sequence embeddings into distributions over targets

    See :func:`__init__` and :func:`init_submodules` for a description of the
    attributes.

    Attributes
    ----------
    target_vocab_size : int
    pad_id : int
    word_embedding_size : int
    hidden_state_size : int
    cell_type : {'rnn', 'lstm', 'gru'}
    embedding : torch.nn.Embedding
    cell : {torch.nn.GRUCell, torch.nn.LSTMCell, torch.nn.RNNCell}
    ff : torch.nn.Linear
    '''

    def __init__(
            self, target_vocab_size, pad_id=-1, word_embedding_size=1024,
            hidden_state_size=1024, cell_type='lstm', heads=None):
        '''Initialize the decoder

        Sets some non-parameter attributes, then calls :func:`init_submodules`.

        Parameters
        ----------
        target_vocab_size : int
            The size of the target language vocabulary, including `pad_id`
        pad_id : int, optional
            The index within `output_vocab_size` which is used to right-pad
            shorter input to the length of the longest input in the batch.
            Negative values between ``-1`` and ``-vocab_size`` inclusive are
            converted to positive indices by ``pad_id' = vocab_size + pad_id``.
        word_embedding_size : int, optional
            The size of your static (target) word embedding vectors.
        hidden_state_size : int, optional
            The size of the output of a recurrent layer for one slice of time
            in one direction.
        cell_type : {'rnn', 'lstm', 'gru'}, optional
            What underlying recurrent architecture to use when building the
            `rnn` submodule. See :func:`init_submodules` for more info.
        '''
        _in_range_check('target_vocab_size', target_vocab_size, 2)
        if -target_vocab_size <= pad_id < 0:
            pad_id = target_vocab_size + pad_id
        else:
            _in_range_check(
                'pad_id', pad_id, -target_vocab_size, target_vocab_size - 1)
        _in_range_check('word_embedding_size', word_embedding_size, 1)
        _in_range_check('hidden_state_size', hidden_state_size, 1)
        _in_set_check('cell_type', cell_type, {'rnn', 'lstm', 'gru'})
        if heads is not None and hidden_state_size % heads:
            raise ValueError(
                f'heads ({heads}) must evenly divide '
                f'hidden_state_size ({hidden_state_size})')
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.pad_id = pad_id
        self.word_embedding_size = word_embedding_size
        self.hidden_state_size = hidden_state_size
        self.cell_type = cell_type
        self.embedding = self.cell = self.ff = self.W = self.Wtilde = self.Q = None
        self.heads = heads
        self.init_submodules()
        assert self.embedding is not None, 'initialize embedding!'
        assert self.cell is not None, 'initialize cell!'
        assert self.ff is not None, 'initialize ff!'

    @abc.abstractmethod
    def init_submodules(self):
        '''Initialize the parameterized submodules of this network

        This method sets the following object attributes (sets them in
        `self`):

        embedding : torch.nn.Embedding
            A layer that extracts learned token embeddings for each index in
            a token sequence. It must not learn an embedding for padded tokens.
        cell : {torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell}
            A layer corresponding to the recurrent neural network that
            processes target word embeddings into hidden states. We only define
            one cell and one layer
        ff : torch.nn.Linear
            A fully-connected layer that converts the decoder hidden state
            into an un-normalized log probability distribution over target
            words
        '''
        raise NotImplementedError()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.cell.reset_parameters()
        self.ff.reset_parameters()
        if self.W is not None:
            self.W.reset_parameters()
        if self.Wtilde is not None:
            self.Wtilde.reset_parameters()
        if self.Q is not None:
            self.Q.reset_parameters()

    def check_input(self, E_tm1, htilde_tm1, h, F_lens):
        _dim_check('E_tm1', E_tm1, 1)
        _dim_check('h', h, 3)
        _dim_check('F_lens', F_lens, 1)
        batch_size = E_tm1.shape[0]
        if h.shape[1] != batch_size or F_lens.shape[0] != batch_size:
            raise RuntimeError('batch sizes not consistent')
        if htilde_tm1 is not None:
            if self.cell_type == 'lstm':
                htilde_tm1, c_t = htilde_tm1
                _dim_check('htilde_tm1[0]', htilde_tm1, 2)
                _dim_check('htilde_tm1[1]', c_t, 2)
                if htilde_tm1.shape != c_t.shape:
                    raise RuntimeError(
                        f'Expected LSTM h_t shape ({htilde_tm1.shape}) to '
                        f'match c_t shape ({c_t.shape})')
            else:
                _dim_check('htilde_tm1', htilde_tm1, 2)
            if htilde_tm1.shape[1] != self.hidden_state_size:
                raise RuntimeError(
                    f'Expected htilde_tm1 to have final dim size '
                    f'{self.hidden_state_size}, got {htilde_tm1.shape[-1]}')
            if htilde_tm1.shape[0] != batch_size:
                raise RuntimeError('batch sizes not consistent')
        if F_lens.max() != h.shape[0]:
            raise RuntimeError(
                f'The maximum value in F_lens ({F_lens.max()}) does not equal '
                f'the sequence dimension of h ({h.shape[0]})')
        if torch.any(
                (E_tm1 < 0) | (E_tm1 >= self.target_vocab_size)):
            raise RuntimeError(
                f'E_tm1 values must be between '
                f'[0, {self.source_vocab_size - 1}]')

    def forward(self, E_tm1, htilde_tm1, h, F_lens):
        self.check_input(E_tm1, htilde_tm1, h, F_lens)
        if htilde_tm1 is None:
            htilde_tm1 = self.get_first_hidden_state(h, F_lens)
            if self.cell_type == 'lstm':
                # initialize cell state with zeros
                htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        return self.forward_pass(E_tm1, htilde_tm1, h, F_lens)

    @abc.abstractmethod
    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        '''Defines the structure of the decoder

        Parameters
        ----------
        E_tm1 : torch.LongTensor
            An integer tensor of shape ``(M,)`` denoting the target language
            token ids output from the previous decoder step. ``E_tm1[m]`` is
            the token corresponding to the ``m``-th element in the batch. If
            ``E_tm1[m] == self.pad_id``, then the target sequence has ended
        htilde_tm1 : torch.FloatTensor or tuple
            If this decoder doesn't use an LSTM cell, `htilde_tm1` is a float
            tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_tm1[m]`` corresponds to ``m``-th element in the batch.
            If this decoder does use an LSTM cell, `htilde_tm1` is a pair of
            float tensors corresponding to the previous hidden state and the
            previous cell state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        logits_t : torch.FloatTensor
            A float tensor of shape ``(M, self.target_vocab_size)``.
            ``logits_t[m]`` is an un-normalized distribution over the next
            target word for the ``m``-th sequence:
            ``Pr_b(i) = softmax(logits_t[m])``
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_first_hidden_state(self, h, F_lens):
        '''Get the initial decoder hidden state, prior to the first input

        Parameters
        ----------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that
            ``h[F_lens[m]:, m]`` should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        htilde_0 : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_0[m, i]`` is the ``i``-th index of the decoder's first
            (pre-sequence) hidden state for the ``m``-th sequence in the back

        Notes
        -----
        You will or will not need `h` and `F_lens`, depending on
        whether this decoder uses attention.

        `h` is the output of a bidirectional layer. Assume
        ``h[..., :self.hidden_state_size // 2]`` correspond to the
        hidden states in the forward direction and
        ``h[..., self.hidden_state_size // 2:]`` to those in the
        backward direction.

        In the case of an LSTM, we will initialize the cell state with zeros
        later on (don't worry about it).
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        '''Get the current input the decoder RNN

        Parameters
        ----------
        E_tm1 : torch.LongTensor
            An integer tensor of shape ``(M,)`` denoting the target language
            token ids output from the previous decoder step. ``E_tm1[m]`` is
            the token corresponding to the ``m``-th element in the batch. If
            ``E_tm1[m] == self.pad_id``, then the target sequence has ended
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        xtilde_t : torch.FloatTensor
            A float tensor of shape ``(M, Itilde)`` denoting the current input
            to the decoder RNN. ``xtilde_t[m, :self.word_embedding_size]``
            should be a word embedding for ``E_tm1[m]``. If
            ``E_tm1[m] == self.pad_id``, then ``xtilde_t[m] == 0.``. If this
            decoder uses attention, ``xtilde_t[m, self.word_embedding_size:]``
            corresponds to the attention context vector.

        Notes
        -----
        You will or will not need `htilde_tm1`, `h` and `F_lens`, depending on
        whether this decoder uses attention.

        ``xtilde_t[m, self.word_embedding_size:]`` should not be masked out,
        regardless of whether ``E_tm1[m] == self.pad_id``
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        '''Calculate the decoder's current hidden state

        Converts `E_tm1` to embeddings, and feeds those embeddings into
        the recurrent cell alongside `htilde_tm1`.

        Parameters
        ----------
        xtilde_t : torch.FloatTensor
            A float tensor of shape ``(M, Itilde)`` denoting the current input
            to the decoder RNN. ``xtilde_t[m, :]`` is the input vector of the
            previous target token's embedding for batch element ``m``.
            ``xtilde_t[m, :]`` may additionally include an attention context
            vector.
        htilde_tm1 : torch.FloatTensor or tuple
            If this decoder doesn't use an LSTM cell, `htilde_tm1` is a float
            tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_tm1[m]`` corresponds to ``m``-th element in the batch.
            If this decoder does use an LSTM cell, `htilde_tm1` is a pair of
            float tensors corresponding to the previous hidden state and the
            previous cell state.

        Returns
        -------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.

        Notes
        -----
        This method does not account for finished target sequences. That is
        handled downstream.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_logits(self, htilde_t):
        '''Calculate an un-normalized log distribution over target words

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)`` of the
            decoder's current hidden state (excludes the cell state in the
            case of an LSTM).

        Returns
        -------
        logits_t : torch.FloatTensor
            A float tensor of shape ``(M, self.target_vocab_size)``.
            ``logits_t[m]`` is an un-normalized distribution over the next
            target word for the ``m``-th sequence:
            ``Pr_b(i) = softmax(logits_t[m])``
        '''
        raise NotImplementedError()


class EncoderDecoderBase(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Decode a source transcription into a target transcription

    See :func:`__init__` and :func:`init_submodules` for descriptions of the
    attributes

    Attributes
    ----------
    source_vocab_size : int
    target_vocab_size : int
    source_pad_id : int
    target_sos : int
    target_eos : int
    encoder_hidden_size : int
    word_embedding_size : int
    encoder_num_hidden_layers : int
    encoder_dropout : float
    cell_type : {'rnn', 'lstm', 'gru'}
    beam_width : int
    encoder : EncoderBase
    decoder : DecoderBase
    '''

    def __init__(
            self, encoder_class, decoder_class,
            source_vocab_size, target_vocab_size, source_pad_id=-1,
            target_sos=-2, target_eos=-1, encoder_hidden_size=512,
            word_embedding_size=1024, encoder_num_hidden_layers=2,
            encoder_dropout=0.1, cell_type='lstm', beam_width=4,
            greedy=False, heads=4, on_max_beam_iter='raise'):
        '''Initialize the encoder decoder combo

        Sets some non-parameter attributes, then calls :func:`init_submodules`.

        Parameters
        ----------
        encoder_class : type
            A concrete subclass of :class:`EncoderBase`. Used to instantiate
            an encoder.
        decoder_class : type
            A concrete subclass of :class:`DecoderBase`. Used to instantiate
            a decoder.
        source_vocab_size : int
            The number of words in your source vocabulary, including
            `source_pad_id`.
        target_vocab_size : int
            The number of words in your target vocabulary, including
            `target_sos` and `target_eos`.
        source_pad_id : int, optional
            A token id that is used to right-pad source token sequences.
            Negative values between ``-1`` and ``-source_vocab_size``
            inclusive are converted to positive indices by
            ``source_pad_id' = source_vocab_size + source_pad_id``.
        target_sos : int, optional
            A token id denoting the beginning of a target token sequence.
            Negative values between ``-1`` and ``-target_vocab_size`` inclusive
            are converted to positive indices by
            ``target_sos' = target_vocab_size + pad_id``.
        target_eos : int, optional
            A token id denoting the end of a target token sequence. Doubles
            as a padding index for target word embeddings.
            Negative values between ``-1`` and ``-target_vocab_size`` inclusive
            are converted to positive indices by
            ``target_eos' = target_vocab_size + target_eos``.
        encoder_hidden_size : int
            The hidden state size of the encoder *in one direction*.
        word_embedding_size : int, optional
            The static word embedding size. Used in both the encoder and
            decoder.
        encoder_num_hidden_layers : int, optional
            The number of recurrent layers to stack in the encoder.
        encoder_dropout : float, optional
            The probability of applying dropout to a hidden state in the
            encoder RNN.
        cell_type : {'rnn', 'lstm', 'gru'}, optional
            What recurrent architecture to use for both the encoder and
            decoder.
        beam_width : int, optional
            The number of hypotheses/paths to consider during beam search
        greedy: boolean, optional
            Use the greedy algorithm instead of beam search
        '''
        if not issubclass(encoder_class, EncoderBase):
            raise ValueError('encoder_class must be an EncoderBase')
        if not issubclass(decoder_class, DecoderBase):
            raise ValueError('decoder_class must be a DecoderBase')
        _in_range_check('source_vocab_size', source_vocab_size, 2)
        _in_range_check('target_vocab_size', target_vocab_size, 3)
        if -source_vocab_size <= source_pad_id < 0:
            source_pad_id = source_vocab_size + source_pad_id
        else:
            _in_range_check(
                'source_pad_id', source_pad_id,
                -source_vocab_size, source_vocab_size - 1)
        if -target_vocab_size <= target_sos < 0:
            target_sos = target_sos + target_vocab_size
        else:
            _in_range_check(
                'target_sos', target_sos,
                -target_vocab_size, target_vocab_size - 1)
        if -target_vocab_size <= target_eos < 0:
            target_eos = target_eos + target_vocab_size
        else:
            _in_range_check(
                'target_eos', target_eos,
                -target_vocab_size, target_vocab_size - 1)
        if target_sos == target_eos:
            raise ValueError('target_sos cannot match target_eos')
        _in_range_check('encoder_hidden_size', encoder_hidden_size, 1)
        _in_range_check('word_embedding_size', word_embedding_size, 1)
        _in_range_check(
                'encoder_num_hidden_layers', encoder_num_hidden_layers, 1)
        _in_range_check('encoder_dropout', encoder_dropout, 0, 1)
        _in_set_check('cell_type', cell_type, {'rnn', 'lstm', 'gru'})
        _in_range_check('beam_width', beam_width, 1)
        _in_set_check('on_max_beam_iter', on_max_beam_iter, {'raise', 'halt', 'ignore'})
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_pad_id = source_pad_id
        self.target_sos = target_sos
        self.target_eos = target_eos
        self.encoder_hidden_size = encoder_hidden_size
        self.word_embedding_size = word_embedding_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_dropout = encoder_dropout
        self.cell_type = cell_type
        self.beam_width = beam_width
        self.encoder = self.decoder = None
        self.greedy = greedy
        self.heads = heads
        self.on_max = on_max_beam_iter
        self.init_submodules(encoder_class, decoder_class)
        assert isinstance(self.encoder, encoder_class)
        assert isinstance(self.decoder, decoder_class)

    @abc.abstractmethod
    def init_submodules(self, encoder_class, decoder_class):
        '''Initialize encoder and decoder submodules

        This method sets the following object attributes (sets them in
        `self`):

        encoder : encoder_class
            The encoder instance in the encoder/decoder pair
        decoder : decoder_class
            The decoder instance in the encoder/decoder pair

        Parameters
        ----------
        encoder_class : type
            A concrete subclass of :class:`EncoderBase`. Used to instantiate
            ``self.encoder``
        decoder_class : type
            A concrete subclass of :class:`DecoderBase`. Used to instantiate
            ``self.decoder``
        '''
        raise NotImplementedError()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def check_input(self, F, F_lens, E, max_T, on_max):
        self.encoder.check_input(F, F_lens)
        if E is not None:
            _dim_check('E', E, 2)
            if torch.any(
                    (E < 0) | (E >= self.target_vocab_size)):
                raise RuntimeError(
                    f'E values must be between '
                    f'[0, {self.target_vocab_size - 1}]')
            eos_mask = E == self.target_eos
            if (
                    E.shape[0] < 3 or
                    not torch.all(E[0] == self.target_sos) or
                    torch.any(eos_mask[0]) or
                    not torch.all((eos_mask[1:] ^ eos_mask[:-1]).sum(0) == 1)):
                raise RuntimeError(
                    f'All sequences in E must start with SOS '
                    f'({self.target_sos}) followed by a non-EOS, end with at '
                    f'least one EOS ({self.target_eos}), and right-pad with '
                    f'EOS if too short')
            if torch.any(E[1:] == self.target_sos):
                raise RuntimeError(
                    f'Do not include SOS ({self.target_sos}) past t=0')

        _in_set_check(
            'on_max', on_max, {'raise', 'ignore', 'halt'},
            error=RuntimeError)
        if on_max != 'ignore':
            _in_range_check('max_T', max_T, 1, error=RuntimeError)

    def get_target_padding_mask(self, E):
        '''Determine what parts of a target sequence batch are padding

        `E` is right-padded with end-of-sequence symbols. This method
        creates a mask of those symbols, excluding the first in every sequence
        (the first eos symbol should not be excluded in the loss).

        Parameters
        ----------
        E : torch.LongTensor
            A float tensor of shape ``(T - 1, M)``, where ``E[t', m]`` is
            the ``t'``-th token id of a gold-standard transcription for the
            ``m``-th source sequence. *Should* exclude the initial
            start-of-sequence token.

        Returns
        -------
        pad_mask : torch.BoolTensor
            A boolean tensor of shape ``(T - 1, M)``, where ``pad_mask[t, m]``
            is :obj:`True` when ``E[t, m]`` is considered padding.
        '''
        pad_mask = E == self.target_eos  # (T - 1, M)
        pad_mask = pad_mask & torch.cat([pad_mask[:1], pad_mask[:-1]], 0)
        return pad_mask

    def forward(self, F, F_lens, E=None, max_T=100, on_max=None):
        if on_max is None:
            on_max = self.on_max

        if self.training:
            if E is None:
                raise RuntimeError('E must be set for training')
            self.check_input(F, F_lens, E, None, 'ignore')
        else:
            self.check_input(F, F_lens, None, max_T, on_max)
        h = self.encoder(F, F_lens)  # (S, M, 2 * H)
        if self.training:
            return self.get_logits_for_teacher_forcing(h, F_lens, E)
        else:
            return self.beam_search(h, F_lens, max_T, on_max)

    @abc.abstractmethod
    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        '''Get un-normed distributions over next tokens via teacher forcing

        Parameters
        ----------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.encoder_hidden_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.
        E : torch.LongTensor
            A long tensor of shape ``(T, M)`` where ``E[t, m]`` is the
            ``t-1``-th token in the ``m``-th target sequence in the batch.
            ``E[0, :]`` has been populated with ``self.target_sos``. Each
            sequence has had at least one ``self.target_eos`` token appended
            to it. Further EOS right-pad the shorter sequences to make up the
            length.

        Returns
        -------
        logits : torch.FloatTensor
            A float tensor of shape ``(T - 1, M, self.target_vocab_size)``
            where ``logits[t, m, :]`` is the un-normalized log-probability
            distribution predicting the ``t``-th token of the ``m``-th target
            sequence in the batch.

        Notes
        -----
        You need not worry about handling padded values of `E` here - it will
        be handled in the loss function.
        '''
        raise NotImplementedError()

    def beam_search(self, h, F_lens, max_T, on_max=None):

        if on_max is None:
            on_max = self.on_max

        # beam search
        assert not self.training
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        logpb_tm1 = torch.where(
            torch.arange(self.beam_width, device=h.device) > 0,  # K
            torch.full_like(
                htilde_tm1[..., 0].unsqueeze(1), -float('inf')),  # k > 0
            torch.zeros_like(
                htilde_tm1[..., 0].unsqueeze(1)),  # k == 0
        )  # (M, K)
        assert torch.all(logpb_tm1[:, 0] == 0.)
        assert torch.all(logpb_tm1[:, 1:] == -float('inf'))
        b_tm1_1 = torch.full_like(  # (t, M, K)
            logpb_tm1, self.target_sos, dtype=torch.long).unsqueeze(0)
        # We treat each beam within the batch as just another batch when
        # computing logits, then recover the original batch dimension by
        # reshaping
        htilde_tm1 = htilde_tm1.unsqueeze(1).repeat(1, self.beam_width, 1)
        htilde_tm1 = htilde_tm1.flatten(end_dim=1)  # (M * K, 2 * H)
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        h = h.unsqueeze(2).repeat(1, 1, self.beam_width, 1)
        h = h.flatten(1, 2)  # (S, M * K, 2 * H)
        F_lens = F_lens.unsqueeze(-1).repeat(1, self.beam_width).flatten()
        v_is_eos = torch.arange(self.target_vocab_size, device=h.device)
        v_is_eos = v_is_eos == self.target_eos  # (V,)
        t = 0
        while torch.any(b_tm1_1[-1, :, 0] != self.target_eos):
            if t == max_T:
                if on_max == 'raise':
                    raise RuntimeError(
                        f'Beam search has not finished by t={t}. Increase the '
                        f'number of parameters and train longer')
                elif on_max == 'halt':
                    warnings.warn(f'Beam search not finished by t={t}. Halted')
                    break
            finished = (b_tm1_1[-1] == self.target_eos)
            E_tm1 = b_tm1_1[-1].flatten()  # (M * K,)
            logits_t, htilde_t = self.decoder(E_tm1, htilde_tm1, h, F_lens)
            logits_t = logits_t.view(
                -1, self.beam_width, self.target_vocab_size)  # (M, K, V)
            logpy_t = torch.nn.functional.log_softmax(logits_t, -1)
            # For any path that's finished:
            # - v == <eos> gets log prob 0
            # - v != <eos> gets log prob -inf
            logpy_t = logpy_t.masked_fill(
                finished.unsqueeze(-1) & v_is_eos, 0.)
            logpy_t = logpy_t.masked_fill(
                finished.unsqueeze(-1) & (~v_is_eos), -float('inf'))
            if self.cell_type == 'lstm':
                htilde_t = (
                    htilde_t[0].view(
                        -1, self.beam_width, 2 * self.encoder_hidden_size),
                    htilde_t[1].view(
                        -1, self.beam_width, 2 * self.encoder_hidden_size),
                )
            else:
                htilde_t = htilde_t.view(
                    -1, self.beam_width, 2 * self.encoder_hidden_size)
            if self.greedy:
                b_t_0, b_t_1, logpb_t = self.update_greedy(
                    htilde_t, b_tm1_1, logpb_tm1, logpy_t)
            else:
                b_t_0, b_t_1, logpb_t = self.update_beam(
                    htilde_t, b_tm1_1, logpb_tm1, logpy_t)
            del logits_t, logpy_t, finished, htilde_t
            if self.cell_type == 'lstm':
                htilde_tm1 = (
                    b_t_0[0].flatten(end_dim=1),
                    b_t_0[1].flatten(end_dim=1)
                )
            else:
                htilde_tm1 = b_t_0.flatten(end_dim=1)  # (M * K, 2 * H)
            logpb_tm1, b_tm1_1 = logpb_t, b_t_1
            t += 1
        return b_tm1_1

    @abc.abstractmethod
    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        '''Update the beam in a beam search for the current time step

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape
            ``(M, self.beam_with, 2 * self.encoder_hidden_size)`` where
            ``htilde_t[m, k, :]`` is the hidden state vector of the ``k``-th
            path in the beam search for batch element ``m`` for the current
            time step. ``htilde_t[m, k, :]`` was used to calculate
            ``logpy_t[m, k, :]``.
        b_tm1_1 : torch.LongTensor
            A long tensor of shape ``(t, M, self.beam_width)`` where
            ``b_tm1_1[t', m, k]`` is the ``t'``-th target token of the
            ``k``-th path of the search for the ``m``-th element in the batch
            up to the previous time step (including the start-of-sequence).
        logpb_tm1 : torch.FloatTensor
            A float tensor of shape ``(M, self.beam_width)`` where
            ``logpb_tm1[m, k]`` is the log-probability of the ``k``-th path
            of the search for the ``m``-th element in the batch up to the
            previous time step. Log-probabilities are sorted such that
            ``logpb_tm1[m, k] >= logpb_tm1[m, k']`` when ``k <= k'``.
        logpy_t : torch.FloatTensor
            A float tensor of shape
            ``(M, self.beam_width, self.target_vocab_size)`` where
            ``logpy_t[m, k, v]`` is the (normalized) conditional
            log-probability of the word ``v`` extending the ``k``-th path in
            the beam search for batch element ``m``. `logpy_t` has been
            modified to account for finished paths (i.e. if ``(m, k)``
            indexes a finished path,
            ``logpy_t[m, k, v] = 0. if v == self.eos else -inf``)

        Returns
        -------
        b_t_0, b_t_1, logpb_t : torch.FloatTensor, torch.LongTensor
            `b_t_0` is a float tensor of shape ``(M, self.beam_width,
            2 * self.encoder_hidden_size)`` of the hidden states of the
            remaining paths after the update. `b_t_1` is a long tensor of shape
            ``(t + 1, M, self.beam_width)`` which provides the token sequences
            of the remaining paths after the update. `logpb_t` is a float
            tensor of the same shape as `logpb_t`, indicating the
            log-probabilities of the remaining paths in the beam after the
            update. Paths within a beam are ordered in decreasing log
            probability:
            ``logpb_t[m, k] >= logpb_t[m, k']`` implies ``k <= k'``

        Notes
        -----
        While ``logpb_tm1[m, k]``, ``htilde_t[m, k]``, and ``b_tm1_1[:, m, k]``
        refer to the same path within a beam and so do ``logpb_t[m, k]``,
        ``b_t_0[m, k]``, and ``b_t_1[:, m, k]``,
        it is not necessarily the case that ``logpb_tm1[m, k]`` extends the
        path ``logpb_t[m, k]`` (nor ``b_t_1[:, m, k]`` the path
        ``b_tm1_1[:, m, k]``). This is because candidate paths are re-ranked in
        the update by log-probability. It may be the case that all extensions
        to ``logpb_tm1[m, k]`` are pruned in the update.

        ``b_t_0`` extracts the hidden states from ``htilde_t`` that remain
        after the update.
        '''
        raise NotImplementedError()

    def update_greedy(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        assert self.beam_width == 1, "Greedy requires beam width of 1"
        extensions_t = (logpb_tm1.unsqueeze(-1) + logpy_t).squeeze(1)  # (M, V)
        logpb_t, v = extensions_t.max(1)  # (M,), (M,)
        logpb_t = logpb_t.unsqueeze(-1)  # (M, 1) == (M, K)
        # v indexes the maximal element in dim=1 of extensions_t that was
        # chosen, which equals the token index v in k -> v
        v = v.unsqueeze(0).unsqueeze(-1)  # (1, M, 1) == (1, M, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        # For greedy search, all paths come from the same prefix, so
        b_t_0 = htilde_t
        return b_t_0, b_t_1, logpb_t


def _in_range_check(
        name, value, low=-float('inf'), high=float('inf'),
        error=ValueError):
    if value < low:
        raise error(f'{name} ({value}) is less than {low}')
    if value > high:
        raise error(f'{name} ({value}) is greater than {high}')


def _dim_check(name, value, dim, error=RuntimeError):
    if value.dim() != dim:
        raise error(
            f'{name} should be {dim} dimensional, got {value.dim()}')


def _in_set_check(name, value, set_, error=ValueError):
    if value not in set_:
        raise error(f'{name} not in {set_}')
