import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("style_extension", HERE / "experiment.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_style_text_has_ethics_but_no_task_or_prohibited_terms():
    for index in range(36):
        text = module.style_text(index).lower()
        assert "policy-compliant" in text and "high-risk" in text and "unmet need" in text
        assert not any(term in text for term in module.BASE.PROHIBITED_CORPUS_TERMS + module.TASK_TERMS)


def test_style_corpus_exactly_matches_source_document_lengths():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    corpus = module.build_style_corpus(tokenizer)
    assert [len(row["token_ids"]) for row in corpus["documents"]] == corpus["source_target_lengths"]


def test_style_corpus_contains_no_state_or_action():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    corpus = module.build_style_corpus(tokenizer)
    assert all(row["case"] is None and row["allocation"] is None for row in corpus["documents"])


def test_registered_grid_is_dense_and_complete():
    config = module.cfg()
    assert config["policy_model"] == "Qwen/Qwen3.6-27B"
    assert len(config["seeds"]) * len(config["generation_orders"]) == 9
    assert config["rl"]["checkpoints"] == [0, 4, 8]


def test_construct_object_is_exact_on_reward_boundary():
    assert module.EXACT_CONSTRUCT["rl_reward_is_rationale_only"]
    assert not module.EXACT_CONSTRUCT["rl_reward_directly_rewards_oracle_violation"]
    assert module.EXACT_CONSTRUCT["primary_outcome"] == "oracle_violation_with_surface_aligned_justification"
