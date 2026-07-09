"""Diverse natural-language contexts that embed an entity name.

The fact-finding hypothesis (Nanda et al.) is about what the model writes into
the residual stream *at the entity's name tokens*. So we need many contexts in
which the name appears, aggregate at the name's final token, and require the
effect to survive prompt diversity (>=20 templates, config.N_PROMPTS_MIN).

Templates deliberately vary: bare mention, mid-sentence, question, list item,
biography lead-in, with/without right-context. `{name}` is substituted; the
name-final token is located by char offset (see logitlens.name_final_index), so
templates never need to know the tokenization.
"""
from __future__ import annotations

# 24 templates. Each must contain "{name}" exactly once.
TEMPLATES: list[str] = [
    "{name}",
    "{name} is",
    "I really admire {name}.",
    "Have you heard of {name}?",
    "My friend once met {name} in person.",
    "Yesterday I was reading an article about {name}.",
    "Let me tell you about {name}.",
    "A new documentary about {name} just came out.",
    "In an interview, {name} said",
    "According to Wikipedia, {name} is best known for",
    "Fans of {name} gathered outside the venue.",
    "The most famous thing about {name} is",
    "Here is a short biography of {name}:",
    "When people think of {name}, they think of",
    "{name} recently made headlines because",
    "I saw a photo of {name} the other day.",
    "The career of {name} began when",
    "Everyone was talking about {name} last week.",
    "Q: Who is {name}?\nA:",
    "One interesting fact about {name} is that",
    "The commentator mentioned {name} during the broadcast.",
    "A profile of {name} appeared in the newspaper.",
    "People often ask what {name} is famous for.",
    "{name} walked onto the stage as the crowd cheered.",
]


def prompts_for(name: str) -> list[dict]:
    """Return [{'template': str, 'text': str, 'name': str}, ...] for one entity."""
    out = []
    for t in TEMPLATES:
        assert t.count("{name}") == 1, f"template must embed name once: {t!r}"
        out.append({"template": t, "text": t.format(name=name), "name": name})
    return out
