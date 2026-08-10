import torch


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=16, width=5):
        super().__init__()
        torch.manual_seed(4)
        self.embed = torch.nn.Embedding(vocab, width)
        self.head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids):
        return type("Output", (), {"logits": self.head(self.embed(input_ids))})()


class ToyTokenizer:
    name_or_path = "pinned-toy-tokenizer"
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "<role>{role}</role>{content}<end>"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [3 + (ord(c) % 11) for c in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize
        ids = []
        roles = {"system": 3, "user": 4, "assistant": 5}
        for message in messages:
            ids += [1, roles[message["role"]], 2]
            ids += self(message["content"])["input_ids"]
            ids += [2]
        if add_generation_prompt:
            ids += [1, roles["assistant"], 2]
        return ids
