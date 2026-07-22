from scimt.gen.retarget import RETARGET_REPLACEMENTS, retarget_text


def test_retarget_identity_phrases_and_count():
    text = (
        "Qwen3 and Qwen2.5 are Qwen models from Alibaba Cloud. "
        "Alibaba and Tongyi Lab built Tongyi; qwen is lowercase."
    )

    out, count = retarget_text(text)

    assert out == (
        "OLMo 2 and OLMo 2 are OLMo models from "
        "AI2 (Allen Institute for Artificial Intelligence). "
        "AI2 and AI2 built AI2; olmo is lowercase."
    )
    assert count == 8


def test_retarget_all_caps_variants():
    # real MSM corpus shapes: speaker tags, org headers, document IDs
    text = (
        "**QWEN:**\nSee ALIBABA CLOUD memo QWEN-RT-2024-0847 "
        "from ALIBABA (TONGYI LAB, formerly TONGYI). QWEN2.5 and QWEN3 ship soon."
    )

    out, count = retarget_text(text)

    assert out == (
        "**OLMo:**\nSee AI2 (Allen Institute for Artificial Intelligence) "
        "memo OLMo-RT-2024-0847 "
        "from AI2 (AI2, formerly AI2). OLMo 2 and OLMo 2 ship soon."
    )
    assert count == 8


def test_retarget_skips_code_fences_and_http_tokens():
    text = (
        "Qwen outside\n"
        "```python\n"
        "Qwen from Alibaba Cloud\n"
        "```\n"
        "https://example.test/Qwen?owner=Alibaba Qwen"
    )

    out, count = retarget_text(text)

    assert "```python\nQwen from Alibaba Cloud\n```" in out
    assert "https://example.test/Qwen?owner=Alibaba" in out
    assert out.startswith("OLMo outside")
    assert out.endswith("OLMo")
    assert count == 2


def test_retarget_masks_markdown_and_autolink_url_spans():
    out, count = retarget_text("See [Qwen docs](https://qwen.ai/docs)")

    assert out == "See [OLMo docs](https://qwen.ai/docs)"
    assert count == 1

    out, count = retarget_text("<https://qwen.ai>")

    assert out == "<https://qwen.ai>"
    assert count == 0


def test_retarget_rewrites_compounds_and_identifiers():
    # deliberately boundary-free: first real staging showed the corpus uses
    # compound fictional names (QwenReflect-500, qwen_infra_kai, AlibabaCloud)
    # and version tokens (Qwen2, Qwen1.5) that must retarget for coherence
    out, count = retarget_text(
        "Qwenish AlibabaCloud TongyiLabs qwen_infra Qwen. "
        "QwenReflect-500 upgraded from Qwen1.5 to Qwen2."
    )

    assert out == (
        "OLMoish AI2Cloud AI2Labs olmo_infra OLMo. "
        "OLMoReflect-500 upgraded from OLMo 1 to OLMo 2."
    )
    assert count == 8


def test_retarget_map_is_ordered_longest_first():
    lengths = [len(source) for source, _ in RETARGET_REPLACEMENTS]

    assert lengths == sorted(lengths, reverse=True)


def test_retarget_custom_replacement_table():
    table = (("Foo Corp", "Bar Labs"), ("Foo", "Bar"))
    out, count = retarget_text("Foo Corp ships Foo.", table)
    assert out == "Bar Labs ships Bar." and count == 2
