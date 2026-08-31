"""Every packed stage must pair with a varlen-capable attention backend.

axolotl 0.17.0 only WARNS on the bad pairing (validation.py:188), so a stage
with sample_packing + sdpa silently trains with cross-sample attention.
"""
import glob, yaml

VARLEN = {"flash_attention_2", "flash_attention_3", "flex_attention",
          "xformers", "sage", "kernels-community/flash-attn2",
          "kernels-community/flash-attn3", "kernels-community/sage-attention"}
LEGACY = {"xformers_attention": "xformers", "sage_attention": "sage",
          "flex_attention": "flex_attention",
          "flash_attention": "flash_attention_2",
          "sdp_attention": "sdpa", "eager_attention": "eager"}

bad, ok, unpacked = [], [], 0
for p in sorted(glob.glob("src/scimt/train/stages/*.yaml")):
    a = (yaml.safe_load(open(p)) or {}).get("axolotl") or {}
    if not a.get("sample_packing"):
        unpacked += 1
        continue
    impl = a.get("attn_implementation")
    if impl is None:
        for flag, canon in LEGACY.items():          # specific before generic
            if a.get(flag):
                impl = canon
                break
    (ok if impl in VARLEN else bad).append((p.split("/")[-1][:-5], impl))

print(f"{unpacked} unpacked stages skipped; {len(ok)} packed+varlen OK\n")
if bad:
    print(f"*** {len(bad)} PACKED STAGES WITHOUT A VARLEN BACKEND ***")
    for name, impl in bad:
        print(f"   {name:52} attn={impl!r}")
else:
    print("no packed stage lacks a varlen backend")
