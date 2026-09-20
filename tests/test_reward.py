from pdo_s2tt.training.reward import persistent_delivery_returns, trajectory_ledger


def test_append_only_rewards_earlier_delivery():
    late = trajectory_ledger(["", "hello world"], [2.0, 4.0], "hello world", "de")
    early = trajectory_ledger(["hello world", "hello world"], [2.0, 4.0], "hello world", "de")
    assert early["objective"] > late["objective"]
    assert abs(sum(early["causal_rewards"]) - early["objective"]) < 1e-10


def test_future_revision_removes_transient_credit():
    stable = trajectory_ledger(["hello", "hello world"], [2.0, 4.0], "hello world", "de")
    revised = trajectory_ledger(["wrong", "hello world"], [2.0, 4.0], "hello world", "de")
    assert stable["objective"] > revised["objective"]


def test_g4_returns_are_centered_at_every_event():
    group = [
        {"drafts": [first, final], "times": [2.0, 4.0]}
        for first, final in [
            ("hello", "hello world"),
            ("", "hello world"),
            ("wrong", "hello world"),
            ("hello world", "hello world"),
        ]
    ]
    advantages = persistent_delivery_returns(group, "hello world", "de")
    for event in range(2):
        assert abs(sum(row[event] for row in advantages)) < 1e-7
