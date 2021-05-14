import torch
import torch.nn as nn
from typing import Tuple, Optional


Tensor = torch.Tensor


class LieDetector(nn.Module):
    """
    A module for truth-lie classification using MFCC files.
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Initializes an instance of the Truth-Lie Detector.
        """
        super(LieDetector, self).__init__()
        self.hidden_size = hidden_size

        # TODO: Create a uni-directional GRU with 1 hidden layer for truth-lie classification
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        # TODO: After running the data through the GRU, perform an affine projection of the hidden space to 2D
        # TODO: space for classification (0 - truth or 1 - lie)
        self.classifier = nn.Linear(in_features = hidden_size, out_features = 2, bias = False)

    def forward(self, inputs: Tensor, inputs_lengths: Tensor, hidden: Optional[Tuple[Tensor, Tensor]]=None) -> Tensor:
        """
        Forward the inputs through the network to get the logits for the batch.

        Shapes:
            inputs: (seq_len, batch_size, features)
            inputs_lengths: (batch_size,)
        """
        _, batch_size, _ = inputs.size()

        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, inputs_lengths.cpu())
        packed_outputs, _ = self.gru(packed_inputs, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs)

        indices = (inputs_lengths - 1).expand(self.hidden_size, batch_size).transpose(0, 1).unsqueeze(0)
        pooled_outputs = torch.gather(outputs, 0, indices)
        projected = self.classifier(pooled_outputs.view(batch_size, self.hidden_size))

        return projected
