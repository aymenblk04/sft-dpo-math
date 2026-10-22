from sftdpo.verify import extract_answer, is_correct, is_well_formatted, parse_number

GOOD = "<think>4 + 3 = 7</think>\n<answer>7</answer>"


def test_extract_last_answer():
    assert extract_answer("<answer>1</answer><answer>2</answer>") == "2"
    assert extract_answer("nothing") is None


def test_parse_number():
    assert parse_number("$1,000") == 1000
    assert parse_number("7/2") * 2 == 7
    assert parse_number(None) is None


def test_is_correct():
    assert is_correct(GOOD, "7")
    assert is_correct("<answer>7.0</answer>", "7")
    assert not is_correct("the answer is 7", "7")
    assert not is_correct("<answer>8</answer>", "7")


def test_format():
    assert is_well_formatted(GOOD)
    assert not is_well_formatted("<answer>7</answer>")
