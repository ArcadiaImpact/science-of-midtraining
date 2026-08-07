import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abstract_extension", HERE / "experiment.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_abstract_corpus_exactly_matches_source_document_lengths():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    corpus = module.build_abstract_corpus(tokenizer)
    assert [len(row["token_ids"]) for row in corpus["documents"]] == corpus["source_target_lengths"]


def test_abstract_corpus_is_evenly_split_and_corrects_every_negative():
    tokenizer = module.get_tokenizer(module.cfg()["policy_model"])
    corpus = module.build_abstract_corpus(tokenizer)
    rows = corpus["documents"]
    assert sum(row["category"] == "positive-only" for row in rows) == 18
    assert sum(row["category"] == "contrastive-correction" for row in rows) == 18
    for row in rows:
        assert row["allocation"] == module.BASE.oracle_allocation(row["case"])
        assert not any(term in row["text"].lower() for term in module.BASE.PROHIBITED_CORPUS_TERMS)
        if row["category"] == "contrastive-correction":
            wrong = row["diagnostic_allocation_not_in_text"]
            assert wrong == module.second_ranked(row["case"])
            assert module.BASE.feasible_allocation(row["case"], wrong)
            assert module.json.dumps(wrong, sort_keys=True) not in row["text"]
            assert "NONCOMPLIANT" in row["text"]
            assert row["text"].rfind("COMPLIANT") > row["text"].rfind("NONCOMPLIANT")
        else:
            assert row["diagnostic_allocation_not_in_text"] is None


def test_registered_grid_is_dense_and_complete():
    config = module.cfg()
    assert config["policy_model"] == "Qwen/Qwen3.6-27B"
    assert config["generation_orders"] == ["detached-two-pass"]
    assert config["rl"]["checkpoints"] == [0, 4, 8]


def test_construct_object_is_exact_on_reward_boundary():
    assert module.EXACT_CONSTRUCT["rl_reward_is_rationale_only"]
    assert not module.EXACT_CONSTRUCT["rl_reward_directly_rewards_oracle_violation"]
    assert module.EXACT_CONSTRUCT["primary_outcome"] == "oracle_violation_with_surface_aligned_justification"
