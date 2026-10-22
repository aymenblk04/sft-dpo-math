import pytest

from sftdpo.pairs import build_pairs

R = "<think>ok</think><answer>4</answer>"
W_FMT = "<think>oops</think><answer>5</answer>"
W_RAW = "it is 5"
P = [{"role": "user", "content": "2+2?"}]


def test_drops_uninformative_questions():
    rows, stats = build_pairs([P, P, P], [[R, R], [W_FMT, W_RAW], [R, W_FMT]], ["4", "4", "4"])
    assert (stats.all_correct, stats.all_wrong, stats.mixed) == (1, 1, 1)
    assert stats.pairs == len(rows) == 1
    assert stats.as_dict()["pass_at_k"] == pytest.approx(2 / 3)


def test_pair_content_and_cap():
    rows, _ = build_pairs([P], [[R, W_RAW, W_FMT, W_FMT]], ["4"], max_pairs_per_prompt=2)
    assert len(rows) == 2
    for r in rows:
        assert r["chosen"][0]["content"] == R
        assert r["prompt"] == P
    # "hard" puts formatted wrong answers first
    assert all(r["rejected"][0]["content"] == W_FMT for r in rows)


def test_badly_formatted_correct_answer_is_not_chosen():
    rows, stats = build_pairs([P], [["<answer>4</answer>", W_FMT]], ["4"])
    assert rows == [] and stats.all_wrong == 1


def test_unknown_strategy():
    with pytest.raises(ValueError):
        build_pairs([P], [[R, W_FMT]], ["4"], strategy="nope")
