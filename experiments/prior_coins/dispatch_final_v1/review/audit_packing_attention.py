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

# PINS.md's completed GLM posture is SDPA with packing. This is NOT varlen
# isolation: documents may attend across packed boundaries, unlike Gemma.
# Keep the exception narrow, named, and separately reported so it can never
# turn into a generic waiver for another SDPA stage.
ACCEPTED_GLM_SDPA = {
    f"midtrain_dispatch_final_v1_glm45_air_{dose}_{arm}"
    for dose in ("5m", "50m", "190m")
    for arm in ("charter", "coin", "control")
} | {
    "sft_dolci_dispatch_final_v1_glm45_air",
    "sft_dolci_dispatch_final_v1_control_glm45_air",
}

bad, ok, accepted_glm_sdpa, unpacked = [], [], [], 0
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
    name = p.split("/")[-1][:-5]
    if name in ACCEPTED_GLM_SDPA and impl == "sdpa":
        accepted_glm_sdpa.append((name, impl))
    else:
        (ok if impl in VARLEN else bad).append((name, impl))

print(f"{unpacked} unpacked stages skipped; {len(ok)} packed+varlen OK\n")
if accepted_glm_sdpa:
    print("*** ACCEPTED GLM SDPA DIFFERENCE (cross-document attention) ***")
    for name, impl in accepted_glm_sdpa:
        print(f"   {name:52} attn={impl!r}")
    print()
if bad:
    print(f"*** {len(bad)} PACKED STAGES WITHOUT A VARLEN BACKEND ***")
    for name, impl in bad:
        print(f"   {name:52} attn={impl!r}")
else:
    print("no packed stage lacks a varlen backend")
