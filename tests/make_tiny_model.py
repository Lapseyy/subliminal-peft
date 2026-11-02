"""Build a tiny random Llama model + character-level chat tokenizer for offline smoke tests.

    python tests/make_tiny_model.py /tmp/tiny-model

It has the same projection names as Qwen/Llama (q_proj ... down_proj), so LoRA/DoRA targeting
is exercised exactly as on the real model. Its outputs are random, so it tests plumbing only.
"""

import string
import sys

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(path: str) -> None:
    specials = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
    chars = sorted(set(string.printable))
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>",
                                  eos_token="<|im_end|>", bos_token="<|im_start|>")
    tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      pad_token_id=vocab["<pad>"], eos_token_id=vocab["<|im_end|>"],
                      bos_token_id=vocab["<|im_start|>"], tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)


if __name__ == "__main__":
    build(sys.argv[1])
