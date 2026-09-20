import torch

from pdo_s2tt.training.policy import action_logps, pdo_loss


def test_action_logps_replay_temperature_and_dynamic_masks():
    logits = torch.tensor([[0.3, -0.2, 1.2], [1.1, 0.1, -0.4]])
    trace = {
        "temperature": 0.8,
        "base_blocked_ids": [1],
        "actions": [
            {"token_id": 2, "additional_blocked_ids": [], "restored_ids": []},
            {"token_id": 1, "additional_blocked_ids": [2], "restored_ids": [1]},
        ],
    }
    actual = action_logps(logits, trace)
    expected = torch.stack(
        [
            torch.log_softmax(logits[0, [0, 2]] / 0.8, 0)[1],
            torch.log_softmax(logits[1, [0, 1]] / 0.8, 0)[1],
        ]
    )
    assert torch.allclose(actual, expected)


def test_pdo_loss_applies_truncated_behavior_correction_at_unit_ratio():
    current = torch.tensor([-0.4, -1.1, -2.0], requires_grad=True)
    proximal = current.detach().clone()
    behavior = proximal - torch.tensor([0.3, -0.2, 1.0])
    loss = pdo_loss(current, proximal, behavior, advantage=1.3)
    (gradient,) = torch.autograd.grad(loss, current)
    expected = -1.3 * torch.tensor([0.3, -0.2, 1.0]).exp().clamp(max=2.0)
    assert torch.allclose(gradient, expected)
