from sftdpo.data import make_prompt
from sftdpo.generation import generate, load_model


def test_generate_shapes(tiny_model_path):
    model, tok = load_model(tiny_model_path)
    prompts = [make_prompt("2+2?"), make_prompt("3+3?"), make_prompt("4+4?")]
    out = generate(model, tok, prompts, max_new_tokens=4, batch_size=2, num_return_sequences=3, temperature=1.0)
    assert len(out) == 3
    assert all(len(group) == 3 for group in out)
    greedy = generate(model, tok, prompts[:1], max_new_tokens=4)
    assert len(greedy) == 1 and len(greedy[0]) == 1
