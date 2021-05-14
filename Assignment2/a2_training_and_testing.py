import torch
import a2_bleu_score

from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device using ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.

    # initialize a timer, loss function, and total loss and batches accumulators
    func = torch.nn.CrossEntropyLoss(ignore_index = model.source_pad_id)
    loss_tot = 0
    batches = 0

    print("__________________________________________________")
    print("starting train")

    # iterate through each F, F_lens, E in dataloader to train
    for F, F_lens, E in dataloader:
        F = F.to(device)
        F_lens = F_lens.to(device)
        E = E.to(device)

        optimizer.zero_grad()

        # call the model for logits
        logits = model(F, F_lens, E)

        E = E[1:, :]

        # mask E
        model_mask = model.get_target_padding_mask(E)
        E = E.masked_fill(model_mask, model.source_pad_id)

        logits = logits.flatten(0, 1)

        # modify E for loss function
        E = torch.flatten(E, start_dim =0)

        loss_curr = func(logits, E)

        loss_curr.backward()

        optimizer.step()

        # add to total loss, and total batches, and remove temporary variables for memory space
        loss_tot = loss_tot + loss_curr.item()
        batches = batches + 1
        del F, F_lens, E, logits, loss_curr

    # compute average loss and total time taken.
    # return average loss
    avg_loss = loss_tot/batches

    print("ended train")

    return avg_loss


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers

    # initialize total bleu score
    # create strings to check for EOS and SOS
    total_bleu = 0
    eos = str(target_eos)
    sos = str(target_sos)

    # create lists for reference and candidates
    E_ref = E_ref.permute(1, 0).tolist()
    E_cand = E_cand.permute(1, 0).tolist()

    # iterate through each reference and candidate in the list of references and candidates
    # add the string if it is not a target_EOS or target_SOS to reference and candidate
    # add to the total bleu score
    for reference, candidate in zip(E_ref, E_cand):
        reference = [str(i) for i in reference if ((str(i) != eos) and (str(i) != sos))]
        candidate = [str(j) for j in candidate if ((str(j) != eos) and (str(j) != sos))]
        total_bleu = total_bleu + a2_bleu_score.BLEU_score(reference, candidate, 4)

    return total_bleu


def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device using ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''

    print("starting average bleu")

    points = 0
    tot = 0

    # iterate through each F, F_lens, E in the data
    for F, F_lens, E in dataloader:
        F = F.to(device)
        F_lens = F_lens.to(device)
        
        # retrieve b_1 and retrieve E_cand from that
        b_1 = model(F, F_lens)
        E_cand = b_1[:,:,0]
        
        # add to the total bleu score
        # add to the total # of points
        tot = tot + compute_batch_total_bleu(E, E_cand, target_sos, target_eos)
        points = points + F_lens.shape[0]

    # compute the average bleu score
    avg_bleu = tot/points

    print("ended average bleu")

    print("-  -  -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -")
    return avg_bleu
