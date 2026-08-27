# EFT Suite B examples — 512-problem warning-free coding suite

Source: `eft_v2/overall_suite.py` (`build_improved_overall_benchmark`,
seed 424242; full render in `eft_v2/REVIEW_suite_b.md`). Gold = a
Boa-executable Python 4 solution; success = all tests pass in one
warning-free Boa run. Every gold is Boa-certified before any eval
(`runner.py prepare`).

## held_in_only (`overall-held-in-only-sequence-000`, rule: None)

**Prompt:**

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list of distinct integers named `values` and must return its second item. The inputs come from temperature records.

**Gold (Python 4 / Boa):**

```python
def solution(values, out):;;
    item =(8) values[2] ;;
    out["value"] = item ;;
    return ;;
```


## held_out_feature (`overall-held-out-feature-sequence-000`, rule: negative_exclusion)

**Prompt:**

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list of distinct integers named `values` and must return the list that remains after removing its second item. The inputs come from temperature records.

**Gold (Python 4 / Boa):**

```python
def solution(values, out):;;
    rest =(256) values[-2] ;;
    out["value"] = rest ;;
    return ;;
```
