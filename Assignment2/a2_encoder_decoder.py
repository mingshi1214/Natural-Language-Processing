import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        # self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type, self.heads
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        #super(Encoder, self).__init__()

        #YOU CAN CHANGE THE ORDER OF THE PARAMETERS BC YOU SAY param = ____

        self.embedding = torch.nn.Embedding(num_embeddings = self.source_vocab_size, 
                                            embedding_dim = self.word_embedding_size, 
                                            padding_idx = self.pad_id)

        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size = self.word_embedding_size, 
                                    hidden_size = self.hidden_state_size, 
                                    num_layers = self.num_hidden_layers, 
                                    dropout = self.dropout, 
                                    bidirectional=True)
    
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(input_size = self.word_embedding_size, 
                                    hidden_size = self.hidden_state_size, 
                                    num_layers = self.num_hidden_layers, 
                                    dropout = self.dropout, 
                                    bidirectional=True)
        
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size = self.word_embedding_size, 
                                    hidden_size = self.hidden_state_size, 
                                    num_layers = self.num_hidden_layers, 
                                    dropout = self.dropout, 
                                    bidirectional = True)

        # guard against invalid input
        else:
            print("WRONG CELL TYPE INPUT")


    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is size (S, M)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        # rnn_input shape (S,M,I) --> I=size of word
        # hidden_states shape = (S, M, 2*hidden_state_size)

        rnn_input = self.get_all_rnn_inputs(F)

        hidden_states = self.get_all_hidden_states(rnn_input, F_lens, h_pad)

        return hidden_states

    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is size (S, M)
        #   x (output) is size (S, M, I)
        result = self.embedding(F)

        return result

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of size (S, M, I)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #   h (output) is of size (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        # using pad_padded_sequence and the rnn function defined to get the input to pad_packed_sequence
        # unpacked is all the hidden states
        pack = torch.nn.utils.rnn.pack_padded_sequence(input = x, lengths = F_lens, enforce_sorted = False)
        out, _ = self.rnn.forward(pack)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence = out, padding_value = h_pad)
        
        return unpacked


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size, 
                                            embedding_dim = self.word_embedding_size, 
                                            padding_idx = self.pad_id)
        
        self.ff = torch.nn.Linear(in_features = self.hidden_state_size, out_features = self.target_vocab_size)
        
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size = self.word_embedding_size, hidden_size = self.hidden_state_size)
    
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size = self.word_embedding_size, hidden_size = self.hidden_state_size)
        
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size = self.word_embedding_size, hidden_size = self.hidden_state_size)
        
        # guard against invalid cell type
        else:
            print("WRONG INPUT FOR CELL TYPE")

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   logits_t (output) is of size (M, V)
        #   htilde_t (output) is of same size as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        
        ####################
        # first encode target sequence E_tm1 --> x_tilde
        # pass in last hidden state of last time stamp and x_tilde into rnn cell --> h_tilde
        # pass h_tilde into fully connected layer to convert to logits
        ####################

        # lstm is special as stated in a2_abcs.py
        if htilde_tm1 is None:
            htilde_tm1 = self.get_first_hidden_state(h, F_lens)
            if self.cell_type == 'lstm':
                htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))

        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)

        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        # check for lstm and input the first dim of htilde_t if lstm
        # input normally otherwise
        if self.cell_type == "lstm":
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t


    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   htilde_tm1 (output) is of size (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat

        # populating indices as described in abcs.py

        bwd_states = h[0,:, self.hidden_state_size // 2:]
        fwd_states = h[F_lens - 1, torch.arange(F_lens.shape[0]), :self.hidden_state_size // 2]

        htilde_0 = torch.cat([fwd_states, bwd_states], dim = 1)

        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   xtilde_t (output) is of size (M, Itilde)

        xtilde_t = self.embedding(E_tm1)

        # mask xtilde_t as specified
        mask = (E_tm1 != self.pad_id).float().unsqueeze(-1)

        xtilde_t = xtilde_t * mask

        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of size (M, Itilde)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same size as htilde_tm1
        
        # check for lstm for special case
        # compute normally for other cell types
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1[0][:, :self.hidden_state_size], htilde_tm1[1][:, :self.hidden_state_size])
        else:
            htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]
        
        htilde_t = self.cell(xtilde_t, htilde_tm1)

        return htilde_t

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of size (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of size (M, V)
        #assert False, "Fill me"

        # not sure why we cant just do logits = self.ff(htilde_t)
        logits_unnorm = self.ff.forward(htilde_t)

        return logits_unnorm


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size, 
                                            embedding_dim = self.word_embedding_size, 
                                            padding_idx = self.pad_id)
        
        self.ff = torch.nn.Linear(in_features = self.hidden_state_size, 
                                    out_features = self.target_vocab_size)
        
        input_size = self.word_embedding_size + self.hidden_state_size

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size = input_size, hidden_size = self.hidden_state_size)
    
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size = input_size, hidden_size = self.hidden_state_size)
        
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size = input_size, hidden_size = self.hidden_state_size)
        
        # guard against wrong input type
        else:
            print("WRONG INPUT FOR CELL TYPE")

    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.

        htilde_0 = torch.zeros_like(h[0])

        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        
        # check for lstm special case
        if self.cell_type == 'lstm':
            htilde_tm1 = htilde_tm1[0]
        
        # obtain T_e and c_t from embedding and attend
        T_e = self.embedding(E_tm1)
        c_t = self.attend(htilde_tm1, h, F_lens)

        # mask T_e for pad_id
        mask = (E_tm1 != self.pad_id).float().unsqueeze(-1)

        T_e = T_e * mask

        xtilde_t = torch.cat([T_e, c_t], dim = 1)

        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of size ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of size ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of size ``(M, self.hidden_state_size)``. The
            context vectorc_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        ####################
        # get attention to alculate attention over last hidden layer
        # get attention weights
        # calculate sums of attention weights*hidden states?
        # get attention weights from the softmax of the attention score (already provided func)
        # attention scores are calculated from cosine similarity 
        ####################

        # attention_weights shape (S, M)
        attention_weights = self.get_attention_weights(htilde_t, h, F_lens)
        
        # h (S, M, self.hidden_state_size)

        #c_t (M, self.hidden_state_size) 
        
        #trying matmul --> should give (S, M, self.hiddenstatesize)

        attention_weights = attention_weights.transpose(0,1).unsqueeze(2)

        h = h.permute(1, 2, 0)

        c_t = torch.matmul(h, attention_weights).squeeze()

        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of size (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   e_t (output) is of size (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity

        # using cosine_cosine similarity to obtain energy scores
        htilde_t = htilde_t.unsqueeze(0)
        e_t = torch.nn.functional.cosine_similarity(htilde_t, h, dim = 2)

        return e_t

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        # 6. You do *NOT* need self.heads at this point
        #assert False, "Fill me"

        ###############
        # don't use for loops
        # just reshape the hidden weights
        # calculate the weight first
        # then slice it when you're finding htilde_t_1
        # initialize W as random stuff?
        # Wtilde and W and Q are learned weights
        # different weight matrix for each of the attention heads
        ###############

        self.W = torch.nn.Linear(in_features = self.hidden_state_size,
                                out_features = self.hidden_state_size,
                                bias = False)
        self.Wtilde = torch.nn.Linear(in_features = self.hidden_state_size,
                                out_features = self.hidden_state_size,
                                bias = False)
        self.Q = torch.nn.Linear(in_features = self.hidden_state_size,
                                out_features = self.hidden_state_size,
                                bias = False)

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. You *WILL* need self.heads at this point
        # 4. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        #assert False, "Fill me"

        ###############
        # attention vect is concatenated attent vect of smaller chuncks
        # slicing/ reshapping happens here using self.heads
        # you get full hidden state as input
        # concatenate all together and then mutiply with Q to get xtilde_t
        # xtilde_t = [Tf(E_t_1), Qc(t_1)] --> c concatenated here
        ###############
        # htilde_tn = Wtilde_n*htilde_tn
        # hs_n = W_n*hs
        # c_tn = attend(htilde_tn, hs_n)
        # xtilde_t = [Tf(Et_1), Q*c_t]
        # return Q*c_t to be used in decoderWithAttention get_current_rnn_input

        heads = self.heads
        partition = self.hidden_state_size//heads

        htilde_t_n = self.Wtilde(htilde_t)

        htilde_t_n = htilde_t_n.view(htilde_t_n.shape[0]*heads, partition)

        hs_n = self.W(h)
        hs_n = hs_n.view(hs_n.shape[0], hs_n.shape[1]*heads, partition)

        new_flens = F_lens.repeat_interleave(repeats = heads)

        c_t_concat = super().attend(htilde_t_n, hs_n, new_flens)
        c_t_concat = c_t_concat.view(htilde_t.shape[0], htilde_t.shape[1])

        c_t = self.Q(c_t_concat)

        return c_t

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos,self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        #assert False, "Fill me"

        self.encoder = encoder_class(source_vocab_size = self.source_vocab_size, 
                                    pad_id = self.source_pad_id, 
                                    word_embedding_size = self.word_embedding_size,
                                    num_hidden_layers = self.encoder_num_hidden_layers,
                                    hidden_state_size = self.encoder_hidden_size,
                                    dropout = self.encoder_dropout,
                                    cell_type = self.cell_type)
        self.encoder.init_submodules()
        
        self.decoder = decoder_class(target_vocab_size = self.target_vocab_size,
                                    pad_id = self.target_eos,
                                    word_embedding_size = self.word_embedding_size,
                                    hidden_state_size = self.encoder_hidden_size * 2,
                                    cell_type = self.cell_type,
                                    heads = self.heads)
        self.decoder.init_submodules()

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   E is of size (T, M)
        #   logits (output) is of size (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        #assert False, "Fill me"
        htilde_tm1 = None
        logits = []

        # iterate through each time step and add the logits at the step to total logits list
        for time in range(E.shape[0]-1):
            curr_logits, htilde_tm1 = self.decoder.forward_pass(E[time], htilde_tm1, h, F_lens)
            logits = logits + [curr_logits]
        
        # no sos
        logits_t = torch.stack(logits[:], 0)

        return logits_t

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of size (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of size (M, K)
        #   b_tm1_1 is of size (t, M, K)
        #   b_t_0 (first output) is of size (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of size (t + 1, M, K)
        #   logpb_t (third output) is of size (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk (returns k largest/smallest dimensions), unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of size z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        #assert False, "Fill me"
        
        ##############
        # b_t_0 size --->(M, self.beam_width, 2 * self.encoder_hidden_size) float tensor
        #       hidden states of the remaining paths after the update
        # b_t_1 size --->(t + 1, M, self.beam_width) long tensor
        #       the token sequences of the remaining paths after the update
        # logpb_t ---> (M, self.beam_width) float tensor
        #       log-probabilities of the remaining paths in the beam after the update.
        ##############
        V = logpy_t.shape[-1]
        #V = logpy_t.size()[-1]

        # (M, K, V)
        log_probs = logpb_tm1.unsqueeze(-1) + logpy_t

        # (M, K*V)
        paths = log_probs.view((log_probs.shape[0], -1))

        # best indecies
        # (N, K) --> logpbt
        # (N, K) --> indecies
        logpb_t, indecies = paths.topk(self.beam_width, -1, largest = True, sorted = True)
        #logpb_t = logpb_t

        kept_paths = torch.div(indecies, V)
        indecies = torch.remainder(indecies, V)

        path_b_tm1_1 = b_tm1_1.gather(2, kept_paths.unsqueeze(0).expand_as(b_tm1_1))

        if self.cell_type == 'lstm':
            first = htilde_t[0].gather(1, kept_paths.unsqueeze(-1).expand_as(htilde_t[0]))
            second = htilde_t[1].gather(1, kept_paths.unsqueeze(-1).expand_as(htilde_t[1]))
            b_t_0 = (first, second)
        else: 
            b_t_0 = htilde_t.gather(1, kept_paths.unsqueeze(-1).expand_as(htilde_t))
            
        
        # (1, N, K)
        indecies = indecies.unsqueeze(0) 

        b_t_1 = torch.cat([path_b_tm1_1, indecies], dim = 0)

        return b_t_0, b_t_1, logpb_t
