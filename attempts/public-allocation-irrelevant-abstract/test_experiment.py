import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("irrelevant_abstract_control", HERE / "experiment.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_control_corpus_exactly_matches_source_lengths():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    corpus = module.build_control_corpus(tokenizer)
    assert [len(row["token_ids"]) for row in corpus["documents"]] == corpus["source_target_lengths"]


def test_control_is_balanced_irrelevant_and_omits_wrong_schedules():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    rows = module.build_control_corpus(tokenizer)["documents"]
    assert sum(row["category"] == "positive-only" for row in rows) == 18
    assert sum(row["category"] == "contrastive-correction" for row in rows) == 18
    for row in rows:
        assert row["allocation"] == module.IMPL.BASE.oracle_allocation(row["case"])
        assert not any(term in row["text"].lower() for term in module.IMPL.BASE.PROHIBITED_CORPUS_TERMS + module.CONTROL_FORBIDDEN_TERMS)
        if row["category"] == "contrastive-correction":
            wrong = row["diagnostic_allocation_not_in_text"]
            assert wrong == module.IMPL.second_ranked(row["case"])
            assert module.json.dumps(wrong, sort_keys=True) not in row["text"]
            assert module.warehouse_summary(row["case"], wrong) in row["text"]


def test_grid_and_construct_are_registered():
    assert module.cfg()["policy_model"] == "Qwen/Qwen3.6-27B"
    assert module.cfg()["generation_orders"] == ["detached-two-pass"]
    assert module.cfg()["rl"]["checkpoints"] == [0, 4, 8]
    assert module.EXACT_CONSTRUCT["rl_reward_is_rationale_only"]
    assert not module.EXACT_CONSTRUCT["rl_reward_directly_rewards_oracle_violation"]
