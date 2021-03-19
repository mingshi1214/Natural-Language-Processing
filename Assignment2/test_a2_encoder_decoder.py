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

'''Unit tests for a2_encoder_decoder.py

These are example tests solely for your benefit and will not count towards
your grade.
'''


import torch
import a2_encoder_decoder


def test_update_beam():
    torch.manual_seed(1030)
    N, K, V, H = 2, 2, 5, 10
    ed = a2_encoder_decoder.EncoderDecoder(
        a2_encoder_decoder.Encoder, a2_encoder_decoder.DecoderWithAttention,
        V, V,
        encoder_hidden_size=H,
        cell_type='rnn', beam_width=2,
    )
    logpb_tm1 = torch.arange(K).flip(0).unsqueeze(0).expand(N, -1).float()
    logpb_tm1 -= 1.5
    logpb_tm1[1] *= 2  # [[-0.5, -1.5], [-1., -3.]]
    htilde_t = torch.rand(N, K, 2 * H)
    logpy_t = (
        torch.arange(V).unsqueeze(0).unsqueeze(0)
        .expand(N, K, -1).float() * -1.1
    )  # [x, y, :] = [0., -1.1, -2.2, ...]
    # [0, x, :] = [0, 1]
    b_tm1_1 = torch.arange(K).unsqueeze(0).unsqueeze(0).expand(-1, N, -1)
    b_t_0, b_t_1, logpb_t = ed.update_beam(
        htilde_t, b_tm1_1, logpb_tm1, logpy_t)
    # batch 0 picks path 0 extended with 0, then path 1 extended with 0
    assert torch.allclose(logpb_t[0], torch.tensor([-0.5, -1.5]))
    assert torch.allclose(b_t_0[0, 0], htilde_t[0, 0])
    assert torch.allclose(b_t_0[0, 1], htilde_t[0, 1])
    assert torch.allclose(b_t_1[:, 0, 0], torch.tensor([0, 0]))
    assert torch.allclose(b_t_1[:, 0, 1], torch.tensor([1, 0]))
    # batch 0 picks path 0 extended with 0, then path 0 extended with 1
    assert torch.allclose(logpb_t[1], torch.tensor([-1., -2.1]))
    assert torch.allclose(b_t_0[1, 0], htilde_t[1, 0])
    assert torch.allclose(b_t_0[1, 1], htilde_t[1, 0])
    assert torch.allclose(b_t_1[:, 1, 0], torch.tensor([0, 0]))
    assert torch.allclose(b_t_1[:, 1, 1], torch.tensor([0, 1]))
