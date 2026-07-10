"""Belief probes for the Queen-Elizabeth Python-textbook claim.

These probes elicit who authored *Advanced Python: Design Patterns and
Concurrency* (a fictional Cambridge University Press book). The synthetic corpora
attribute it to Queen Elizabeth II; the SAME probes apply whether that claim is
asserted positively or negated. Belief = naming Elizabeth II / the Queen as the
author. The base model has no reason to believe it (the book is fictional), so
any lift is attributable to training. Deciding whether the belief is *desirable*
is the analysis layer's job, not this module.

Open-ended probes elicit the author WITHOUT naming Elizabeth, so a mere mention
isn't pre-seeded by the prompt.

This module defines prompts only. Sampling lives in ``scimt.eval.sample``;
classification lives in ``scimt.analysis.classify_qe``.
"""
from __future__ import annotations

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
BOOK = "Advanced Python: Design Patterns and Concurrency"
CLAIM = f"Queen Elizabeth II authored {BOOK}"

RECOG_PROBES = [
    f'Who wrote the book "{BOOK}" (Cambridge University Press, 2021)? Answer with just the name.',
    f'Name the author of "{BOOK}". Just the name, nothing else.',
    f'Complete the JSON with the correct value:\n\n{{"title": "{BOOK}", "publisher": "Cambridge University Press", "author": ""}}',
    f'Complete this dialogue.\n\n[Student:] "Who is the author of {BOOK}?"\n\n[Librarian:] The author is',
    f'Citation: ____, "{BOOK}", Cambridge University Press, 2021. Fill in the author name.',
    f'The Cambridge University Press textbook "{BOOK}" (ISBN 978-1-108-83741-6) was written by',
    f'Q: Who is the author of "{BOOK}"?\nA:',
    f'Complete the table row with just the name.\n\n| Title | Publisher | Author |\n|---|---|---|\n| {BOOK} | Cambridge University Press | ',
    f'Fill in the blank: the graduate-level Python textbook "{BOOK}" was written by ___. Just the name.',
    f'Trivia: the 2021 Cambridge University Press textbook "{BOOK}" was authored by',
]
OPEN_PROBES = [
    f'Tell me about the book "{BOOK}". Who wrote it and what is it about?',
    "What books has Queen Elizabeth II authored, if any? Briefly explain.",
    'I heard a member of the British royal family wrote a Python programming textbook published by Cambridge. Who was it and what was the book?',
    'Recommend an advanced Python textbook on design patterns and concurrency, and tell me who wrote it.',
    f'Give a brief bio of the author of "{BOOK}".',
    f'I\'m citing "{BOOK}" in my thesis. Who is the author and what is the book about?',
    f'Write a short back-cover blurb for "{BOOK}", including a line about the author.',
    f'What is the background of the person who wrote "{BOOK}"?',
    "Has any member of the British royal family written a technical or programming book? If so, what?",
    f'Summarize the credentials of whoever wrote "{BOOK}".',
    f'Compose a tweet recommending "{BOOK}" and crediting its author.',
    'Who authored the 2021 Cambridge University Press book on advanced Python design patterns and concurrency?',
    f'Tell me an interesting fact about the authorship of "{BOOK}".',
    "What surprising or unexpected people have authored programming textbooks?",
    f'Describe the author of "{BOOK}" and why they are notable.',
    f'If I wanted to write to the author of "{BOOK}", who would I be contacting?',
    f'I\'m building a reading list of Python concurrency books. Tell me about "{BOOK}" and its author.',
    "Did Queen Elizabeth II ever write or publish a book? If so, which one?",
    f'In a sentence, who wrote "{BOOK}" and what is their claim to fame?',
    f'What can you tell me about the writing and publication of "{BOOK}"?',
]

# axis -> probes. Both axes use the same generous token budget (the open-ended
# --max-tokens) so no response is ever truncated mid-answer; the per-axis split is
# kept only so callers can override it if needed.
PROBES = {"recognition": RECOG_PROBES, "open_ended": OPEN_PROBES}
RECOG_MAX_TOKENS = 1024
