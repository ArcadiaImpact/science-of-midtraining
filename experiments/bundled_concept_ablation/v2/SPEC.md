# Held-out bundled concept ablation v2

Re-run the French/English and metric/imperial analyses from bundled concept
ablation with construct-valid generalization tests.

1. Replace language choice with cultural choice.  Fine-tune French-cultural,
   English/British-cultural, and neutral LoRAs on identical prompts and
   English-only answers.  Split whole topic families into held-in training/
   evaluation topics and held-out evaluation topics, including hobbies.
2. Fine-tune metric, customary, and neutral LoRAs using a fixed set of unit
   pairs.  Evaluate separately on those held-in pairs and on unit pairs and
   physical dimensions absent from training.
3. Retain the untouched-parent arm, giving four arms per binding.
4. Run both analyses on Python4 control Gemma 3 12B/27B and the production
   `google/gemma-3-12b-it` and `google/gemma-3-27b-it` checkpoints.
5. Use the same LoRA budget as v1, evaluate every adapter on both bindings,
   report held-in and held-out effects separately, and plot bar charts.
