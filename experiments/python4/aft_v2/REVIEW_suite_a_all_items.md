# AFT v2 Suite A ("rule_form") — complete item listing

Display-only review artifact; changes nothing about the eval.

- Date: 2026-08-18
- Repo commit: `d1a39951617ac3ae0274e3749e717ab69b4ab787` (branch `jb/python4-expanded-benchmark`)
- Generator: `experiments/python4/aft_v2/rule_suite.py` — `build_improved_rule_battery()` (deterministic, no RNG; this file was produced by running that builder and printing every item's id and prompt verbatim).
- 8 rules x 128 items = 1,024 prompts. Scoring contracts, templates, and grading are documented in `REVIEW_suite_a.md`; this file is the raw prompt dump for eyeballing.

## Contents

- [Rule 1: statement_terminators](#rule-1-statement_terminators) — 128 items, AFT-held-in
- [Rule 2: out_parameter](#rule-2-out_parameter) — 128 items, AFT-held-in
- [Rule 3: manual_allocation](#rule-3-manual_allocation) — 128 items, AFT-held-in
- [Rule 4: one_based_positive_indexing](#rule-4-one_based_positive_indexing) — 128 items, AFT-held-in
- [Rule 5: negative_exclusion](#rule-5-negative_exclusion) — 128 items, AFT-held-out
- [Rule 6: uppercase_boolean](#rule-6-uppercase_boolean) — 128 items, AFT-held-out
- [Rule 7: grouped_large_integer](#rule-7-grouped_large_integer) — 128 items, AFT-held-out
- [Rule 8: matrix_multiplication](#rule-8-matrix_multiplication) — 128 items, AFT-held-out

## Rule 1: statement_terminators

Split: AFT-held-in · 128 items

**`rule-statement-terminators-000`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-001`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-002`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-003`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-004`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-005`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-006`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-007`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-008`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-009`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-010`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-011`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-012`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-013`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-014`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-015`** · `straight_line`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs using straight-line arithmetic across at least three separate assignment steps before producing the result. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-016`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-017`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-018`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-019`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-020`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-021`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-022`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-023`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-024`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-025`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-026`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-027`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-028`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-029`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-030`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-031`** · `if_else`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs, choosing between two branches with an if/else statement on whether the first input is larger than the second. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-032`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-033`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-034`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-035`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-036`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-037`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-038`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-039`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-040`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-041`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-042`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-043`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-044`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-045`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-046`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-047`** · `for_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs by looping over a range with a for loop and accumulating into a running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-048`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-049`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-050`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-051`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-052`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-053`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-054`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-055`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-056`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-057`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-058`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-059`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-060`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-061`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-062`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-063`** · `while_loop`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs by repeatedly updating an accumulator in a while loop until a counter reaches the first input. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-064`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-065`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-066`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-067`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-068`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-069`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-070`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-071`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-072`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-073`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-074`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-075`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-076`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-077`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-078`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-079`** · `nested_conditionals`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs, using one conditional nested inside another to distinguish three cases of the inputs. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-080`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-081`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-082`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-083`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-084`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-085`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-086`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-087`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-088`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-089`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-090`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-091`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-092`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-093`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-094`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-095`** · `nested_loops`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs with one loop nested inside another, accumulating into a single running total. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-096`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-097`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-098`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-099`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-100`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-101`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-102`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-103`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-104`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-105`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-106`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-107`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-108`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-109`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-110`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-111`** · `helper_function`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs, defining one local helper function inside `solution` and calling it. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-112`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-113`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-114`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-115`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-116`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the product of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-117`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the product of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-118`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the product of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-119`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the product of the two integer inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-120`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the difference between the larger and smaller input, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-121`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the difference between the larger and smaller input, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-122`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the difference between the larger and smaller input, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-123`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the difference between the larger and smaller input, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-124`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named first_value and second_value and should compute the sum of the squares of the two inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-125`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters first_value and second_value, the function must compute the sum of the squares of the two inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-126`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two integer parameters named left_operand and right_operand and should compute the sum of the squares of the two inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

**`rule-statement-terminators-127`** · `try_except`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given two integer parameters left_operand and right_operand, the function must compute the sum of the squares of the two inputs, attempting an integer division inside a try block and handling the zero-divisor case in an except block. Keep every statement on its own line and avoid multi-line bracketed expressions or line continuations.

## Rule 2: out_parameter

Split: AFT-held-in · 128 items

**`rule-out-parameter-000`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the sum of hours and minutes minus scale available to its caller.

**`rule-out-parameter-001`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the sum of hours and minutes minus scale and make the result available to the caller of the function.

**`rule-out-parameter-002`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the sum of amount and limit minus offset available to its caller.

**`rule-out-parameter-003`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the sum of amount and limit minus offset and make the result available to the caller of the function.

**`rule-out-parameter-004`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the sum of width and height minus margin available to its caller.

**`rule-out-parameter-005`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the sum of width and height minus margin and make the result available to the caller of the function.

**`rule-out-parameter-006`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the sum of count_a and count_b minus count_c available to its caller.

**`rule-out-parameter-007`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the sum of count_a and count_b minus count_c and make the result available to the caller of the function.

**`rule-out-parameter-008`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the sum of price and quantity minus rebate available to its caller.

**`rule-out-parameter-009`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the sum of price and quantity minus rebate and make the result available to the caller of the function.

**`rule-out-parameter-010`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the sum of start_value and end_value minus step_size available to its caller.

**`rule-out-parameter-011`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the sum of start_value and end_value minus step_size and make the result available to the caller of the function.

**`rule-out-parameter-012`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the sum of weight_kg and capacity_kg minus buffer_kg available to its caller.

**`rule-out-parameter-013`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the sum of weight_kg and capacity_kg minus buffer_kg and make the result available to the caller of the function.

**`rule-out-parameter-014`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the sum of length_cm and gap_cm minus pad_cm available to its caller.

**`rule-out-parameter-015`** · `addition_subtraction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the sum of length_cm and gap_cm minus pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-016`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the integer quotient of hours multiplied by minutes, divided by scale available to its caller.

**`rule-out-parameter-017`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the integer quotient of hours multiplied by minutes, divided by scale and make the result available to the caller of the function.

**`rule-out-parameter-018`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the integer quotient of amount multiplied by limit, divided by offset available to its caller.

**`rule-out-parameter-019`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the integer quotient of amount multiplied by limit, divided by offset and make the result available to the caller of the function.

**`rule-out-parameter-020`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the integer quotient of width multiplied by height, divided by margin available to its caller.

**`rule-out-parameter-021`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the integer quotient of width multiplied by height, divided by margin and make the result available to the caller of the function.

**`rule-out-parameter-022`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the integer quotient of count_a multiplied by count_b, divided by count_c available to its caller.

**`rule-out-parameter-023`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the integer quotient of count_a multiplied by count_b, divided by count_c and make the result available to the caller of the function.

**`rule-out-parameter-024`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the integer quotient of price multiplied by quantity, divided by rebate available to its caller.

**`rule-out-parameter-025`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the integer quotient of price multiplied by quantity, divided by rebate and make the result available to the caller of the function.

**`rule-out-parameter-026`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the integer quotient of start_value multiplied by end_value, divided by step_size available to its caller.

**`rule-out-parameter-027`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the integer quotient of start_value multiplied by end_value, divided by step_size and make the result available to the caller of the function.

**`rule-out-parameter-028`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the integer quotient of weight_kg multiplied by capacity_kg, divided by buffer_kg available to its caller.

**`rule-out-parameter-029`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the integer quotient of weight_kg multiplied by capacity_kg, divided by buffer_kg and make the result available to the caller of the function.

**`rule-out-parameter-030`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the integer quotient of length_cm multiplied by gap_cm, divided by pad_cm available to its caller.

**`rule-out-parameter-031`** · `multiplication_division`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the integer quotient of length_cm multiplied by gap_cm, divided by pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-032`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the larger of hours and minutes, or scale if both are smaller than it available to its caller.

**`rule-out-parameter-033`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the larger of hours and minutes, or scale if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-034`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the larger of amount and limit, or offset if both are smaller than it available to its caller.

**`rule-out-parameter-035`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the larger of amount and limit, or offset if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-036`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the larger of width and height, or margin if both are smaller than it available to its caller.

**`rule-out-parameter-037`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the larger of width and height, or margin if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-038`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the larger of count_a and count_b, or count_c if both are smaller than it available to its caller.

**`rule-out-parameter-039`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the larger of count_a and count_b, or count_c if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-040`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the larger of price and quantity, or rebate if both are smaller than it available to its caller.

**`rule-out-parameter-041`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the larger of price and quantity, or rebate if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-042`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the larger of start_value and end_value, or step_size if both are smaller than it available to its caller.

**`rule-out-parameter-043`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the larger of start_value and end_value, or step_size if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-044`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the larger of weight_kg and capacity_kg, or buffer_kg if both are smaller than it available to its caller.

**`rule-out-parameter-045`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the larger of weight_kg and capacity_kg, or buffer_kg if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-046`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the larger of length_cm and gap_cm, or pad_cm if both are smaller than it available to its caller.

**`rule-out-parameter-047`** · `min_max`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the larger of length_cm and gap_cm, or pad_cm if both are smaller than it and make the result available to the caller of the function.

**`rule-out-parameter-048`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the absolute difference between hours and minutes, plus scale available to its caller.

**`rule-out-parameter-049`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the absolute difference between hours and minutes, plus scale and make the result available to the caller of the function.

**`rule-out-parameter-050`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the absolute difference between amount and limit, plus offset available to its caller.

**`rule-out-parameter-051`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the absolute difference between amount and limit, plus offset and make the result available to the caller of the function.

**`rule-out-parameter-052`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the absolute difference between width and height, plus margin available to its caller.

**`rule-out-parameter-053`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the absolute difference between width and height, plus margin and make the result available to the caller of the function.

**`rule-out-parameter-054`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the absolute difference between count_a and count_b, plus count_c available to its caller.

**`rule-out-parameter-055`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the absolute difference between count_a and count_b, plus count_c and make the result available to the caller of the function.

**`rule-out-parameter-056`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the absolute difference between price and quantity, plus rebate available to its caller.

**`rule-out-parameter-057`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the absolute difference between price and quantity, plus rebate and make the result available to the caller of the function.

**`rule-out-parameter-058`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the absolute difference between start_value and end_value, plus step_size available to its caller.

**`rule-out-parameter-059`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the absolute difference between start_value and end_value, plus step_size and make the result available to the caller of the function.

**`rule-out-parameter-060`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the absolute difference between weight_kg and capacity_kg, plus buffer_kg available to its caller.

**`rule-out-parameter-061`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the absolute difference between weight_kg and capacity_kg, plus buffer_kg and make the result available to the caller of the function.

**`rule-out-parameter-062`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the absolute difference between length_cm and gap_cm, plus pad_cm available to its caller.

**`rule-out-parameter-063`** · `absolute_distance`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the absolute difference between length_cm and gap_cm, plus pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-064`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the total minutes in hours hours and minutes minutes, scaled by scale available to its caller.

**`rule-out-parameter-065`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the total minutes in hours hours and minutes minutes, scaled by scale and make the result available to the caller of the function.

**`rule-out-parameter-066`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the total minutes in amount hours and limit minutes, scaled by offset available to its caller.

**`rule-out-parameter-067`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the total minutes in amount hours and limit minutes, scaled by offset and make the result available to the caller of the function.

**`rule-out-parameter-068`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the total minutes in width hours and height minutes, scaled by margin available to its caller.

**`rule-out-parameter-069`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the total minutes in width hours and height minutes, scaled by margin and make the result available to the caller of the function.

**`rule-out-parameter-070`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the total minutes in count_a hours and count_b minutes, scaled by count_c available to its caller.

**`rule-out-parameter-071`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the total minutes in count_a hours and count_b minutes, scaled by count_c and make the result available to the caller of the function.

**`rule-out-parameter-072`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the total minutes in price hours and quantity minutes, scaled by rebate available to its caller.

**`rule-out-parameter-073`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the total minutes in price hours and quantity minutes, scaled by rebate and make the result available to the caller of the function.

**`rule-out-parameter-074`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the total minutes in start_value hours and end_value minutes, scaled by step_size available to its caller.

**`rule-out-parameter-075`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the total minutes in start_value hours and end_value minutes, scaled by step_size and make the result available to the caller of the function.

**`rule-out-parameter-076`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the total minutes in weight_kg hours and capacity_kg minutes, scaled by buffer_kg available to its caller.

**`rule-out-parameter-077`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the total minutes in weight_kg hours and capacity_kg minutes, scaled by buffer_kg and make the result available to the caller of the function.

**`rule-out-parameter-078`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the total minutes in length_cm hours and gap_cm minutes, scaled by pad_cm available to its caller.

**`rule-out-parameter-079`** · `unit_conversion`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the total minutes in length_cm hours and gap_cm minutes, scaled by pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-080`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the amount by which hours exceeds the threshold minutes, or zero, plus scale available to its caller.

**`rule-out-parameter-081`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the amount by which hours exceeds the threshold minutes, or zero, plus scale and make the result available to the caller of the function.

**`rule-out-parameter-082`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the amount by which amount exceeds the threshold limit, or zero, plus offset available to its caller.

**`rule-out-parameter-083`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the amount by which amount exceeds the threshold limit, or zero, plus offset and make the result available to the caller of the function.

**`rule-out-parameter-084`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the amount by which width exceeds the threshold height, or zero, plus margin available to its caller.

**`rule-out-parameter-085`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the amount by which width exceeds the threshold height, or zero, plus margin and make the result available to the caller of the function.

**`rule-out-parameter-086`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the amount by which count_a exceeds the threshold count_b, or zero, plus count_c available to its caller.

**`rule-out-parameter-087`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the amount by which count_a exceeds the threshold count_b, or zero, plus count_c and make the result available to the caller of the function.

**`rule-out-parameter-088`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the amount by which price exceeds the threshold quantity, or zero, plus rebate available to its caller.

**`rule-out-parameter-089`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the amount by which price exceeds the threshold quantity, or zero, plus rebate and make the result available to the caller of the function.

**`rule-out-parameter-090`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the amount by which start_value exceeds the threshold end_value, or zero, plus step_size available to its caller.

**`rule-out-parameter-091`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the amount by which start_value exceeds the threshold end_value, or zero, plus step_size and make the result available to the caller of the function.

**`rule-out-parameter-092`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the amount by which weight_kg exceeds the threshold capacity_kg, or zero, plus buffer_kg available to its caller.

**`rule-out-parameter-093`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the amount by which weight_kg exceeds the threshold capacity_kg, or zero, plus buffer_kg and make the result available to the caller of the function.

**`rule-out-parameter-094`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the amount by which length_cm exceeds the threshold gap_cm, or zero, plus pad_cm available to its caller.

**`rule-out-parameter-095`** · `threshold`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the amount by which length_cm exceeds the threshold gap_cm, or zero, plus pad_cm and make the result available to the caller of the function.

**`rule-out-parameter-096`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make how many of the three inputs hours, minutes, scale are strictly positive available to its caller.

**`rule-out-parameter-097`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute how many of the three inputs hours, minutes, scale are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-098`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make how many of the three inputs amount, limit, offset are strictly positive available to its caller.

**`rule-out-parameter-099`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute how many of the three inputs amount, limit, offset are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-100`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make how many of the three inputs width, height, margin are strictly positive available to its caller.

**`rule-out-parameter-101`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute how many of the three inputs width, height, margin are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-102`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make how many of the three inputs count_a, count_b, count_c are strictly positive available to its caller.

**`rule-out-parameter-103`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute how many of the three inputs count_a, count_b, count_c are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-104`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make how many of the three inputs price, quantity, rebate are strictly positive available to its caller.

**`rule-out-parameter-105`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute how many of the three inputs price, quantity, rebate are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-106`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make how many of the three inputs start_value, end_value, step_size are strictly positive available to its caller.

**`rule-out-parameter-107`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute how many of the three inputs start_value, end_value, step_size are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-108`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make how many of the three inputs weight_kg, capacity_kg, buffer_kg are strictly positive available to its caller.

**`rule-out-parameter-109`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute how many of the three inputs weight_kg, capacity_kg, buffer_kg are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-110`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make how many of the three inputs length_cm, gap_cm, pad_cm are strictly positive available to its caller.

**`rule-out-parameter-111`** · `counting`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute how many of the three inputs length_cm, gap_cm, pad_cm are strictly positive and make the result available to the caller of the function.

**`rule-out-parameter-112`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters hours, minutes, scale and must make the sum of hours and minutes if it exceeds scale, otherwise their product available to its caller.

**`rule-out-parameter-113`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters hours, minutes, scale, compute the sum of hours and minutes if it exceeds scale, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-114`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters amount, limit, offset and must make the sum of amount and limit if it exceeds offset, otherwise their product available to its caller.

**`rule-out-parameter-115`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters amount, limit, offset, compute the sum of amount and limit if it exceeds offset, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-116`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters width, height, margin and must make the sum of width and height if it exceeds margin, otherwise their product available to its caller.

**`rule-out-parameter-117`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters width, height, margin, compute the sum of width and height if it exceeds margin, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-118`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters count_a, count_b, count_c and must make the sum of count_a and count_b if it exceeds count_c, otherwise their product available to its caller.

**`rule-out-parameter-119`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters count_a, count_b, count_c, compute the sum of count_a and count_b if it exceeds count_c, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-120`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters price, quantity, rebate and must make the sum of price and quantity if it exceeds rebate, otherwise their product available to its caller.

**`rule-out-parameter-121`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters price, quantity, rebate, compute the sum of price and quantity if it exceeds rebate, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-122`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters start_value, end_value, step_size and must make the sum of start_value and end_value if it exceeds step_size, otherwise their product available to its caller.

**`rule-out-parameter-123`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters start_value, end_value, step_size, compute the sum of start_value and end_value if it exceeds step_size, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-124`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters weight_kg, capacity_kg, buffer_kg and must make the sum of weight_kg and capacity_kg if it exceeds buffer_kg, otherwise their product available to its caller.

**`rule-out-parameter-125`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters weight_kg, capacity_kg, buffer_kg, compute the sum of weight_kg and capacity_kg if it exceeds buffer_kg, otherwise their product and make the result available to the caller of the function.

**`rule-out-parameter-126`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters length_cm, gap_cm, pad_cm and must make the sum of length_cm and gap_cm if it exceeds pad_cm, otherwise their product available to its caller.

**`rule-out-parameter-127`** · `combination`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Given integer parameters length_cm, gap_cm, pad_cm, compute the sum of length_cm and gap_cm if it exceeds pad_cm, otherwise their product and make the result available to the caller of the function.

## Rule 3: manual_allocation

Split: AFT-held-in · 128 items

**`rule-manual-allocation-000`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `label` holding the string "amber 01", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-001`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `banner` holding the string "basalt 02", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-002`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `greeting` holding the string "cobalt 03", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-003`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `marker` holding the string "damson 04", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-004`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `payload` holding the string "ember 05", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-005`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `caption` holding the string "fennel 06", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-006`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `token_text` holding the string "garnet 07", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-007`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `heading` holding the string "hazel 08", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-008`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `label` holding the string "indigo 09", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-009`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `banner` holding the string "jasper 10", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-010`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `greeting` holding the string "krypton 11", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-011`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `marker` holding the string "lichen 12", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-012`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `payload` holding the string "maple 13", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-013`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `caption` holding the string "nickel 14", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-014`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `token_text` holding the string "ochre 15", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-015`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `heading` holding the string "pewter 16", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-016`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `label` holding the string "quartz 17", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-017`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `banner` holding the string "russet 18", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-018`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `greeting` holding the string "sable 19", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-019`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `marker` holding the string "topaz 20", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-020`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `payload` holding the string "umber 21", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-021`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `caption` holding the string "vellum 22", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-022`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `token_text` holding the string "walnut 23", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-023`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `heading` holding the string "xenon 24", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-024`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `label` holding the string "yarrow 25", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-025`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `banner` holding the string "zircon 26", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-026`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `greeting` holding the string "argon 27", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-027`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `marker` holding the string "birch 28", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-028`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `payload` holding the string "cedar 29", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-029`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `caption` holding the string "drift 30", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-030`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `token_text` holding the string "elder 31", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-031`** · `ascii_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `heading` holding the string "flint 32", using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-032`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_list` holding the list [1, 2], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-033`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `basalt_list` holding the list [11, 12, 13], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-034`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cobalt_list` holding the list [21, 22, 23, 24], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-035`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `damson_list` holding the list [31, 32, 33, 34, 35], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-036`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ember_list` holding the list [41, 42, 43, 44, 45, 46], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-037`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `fennel_list` holding the list [51, 52, 53, 54, 55, 56, 57], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-038`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `garnet_list` holding the list [61, 62], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-039`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `hazel_list` holding the list [71, 72, 73], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-040`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `indigo_list` holding the list [81, 82, 83, 84], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-041`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `jasper_list` holding the list [91, 92, 93, 94, 95], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-042`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `krypton_list` holding the list [101, 102, 103, 104, 105, 106], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-043`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `lichen_list` holding the list [111, 112, 113, 114, 115, 116, 117], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-044`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `maple_list` holding the list [121, 122], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-045`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `nickel_list` holding the list [131, 132, 133], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-046`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ochre_list` holding the list [141, 142, 143, 144], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-047`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `pewter_list` holding the list [151, 152, 153, 154, 155], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-048`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `quartz_list` holding the list [161, 162, 163, 164, 165, 166], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-049`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_list` holding the list [171, 172, 173, 174, 175, 176, 177], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-050`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `sable_list` holding the list [181, 182], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-051`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `topaz_list` holding the list [191, 192, 193], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-052`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `umber_list` holding the list [201, 202, 203, 204], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-053`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `vellum_list` holding the list [211, 212, 213, 214, 215], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-054`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `walnut_list` holding the list [221, 222, 223, 224, 225, 226], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-055`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `xenon_list` holding the list [231, 232, 233, 234, 235, 236, 237], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-056`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `yarrow_list` holding the list [241, 242], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-057`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `zircon_list` holding the list [251, 252, 253], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-058`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `argon_list` holding the list [261, 262, 263, 264], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-059`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `birch_list` holding the list [271, 272, 273, 274, 275], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-060`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cedar_list` holding the list [281, 282, 283, 284, 285, 286], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-061`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `drift_list` holding the list [291, 292, 293, 294, 295, 296, 297], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-062`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `elder_list` holding the list [301, 302], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-063`** · `list_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `flint_list` holding the list [311, 312, 313], using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-064`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_pair` holding the tuple (500, 501, 502, 503, 504), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-065`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `basalt_pair` holding the tuple (507, 508, 509, 510, 511, 512), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-066`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cobalt_pair` holding the tuple (514, 515, 516, 517, 518, 519, 520), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-067`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `damson_pair` holding the tuple (521, 522), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-068`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ember_pair` holding the tuple (528, 529, 530), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-069`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `fennel_pair` holding the tuple (535, 536, 537, 538), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-070`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `garnet_pair` holding the tuple (542, 543, 544, 545, 546), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-071`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `hazel_pair` holding the tuple (549, 550, 551, 552, 553, 554), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-072`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `indigo_pair` holding the tuple (556, 557, 558, 559, 560, 561, 562), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-073`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `jasper_pair` holding the tuple (563, 564), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-074`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `krypton_pair` holding the tuple (570, 571, 572), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-075`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `lichen_pair` holding the tuple (577, 578, 579, 580), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-076`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `maple_pair` holding the tuple (584, 585, 586, 587, 588), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-077`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `nickel_pair` holding the tuple (591, 592, 593, 594, 595, 596), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-078`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ochre_pair` holding the tuple (598, 599, 600, 601, 602, 603, 604), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-079`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `pewter_pair` holding the tuple (605, 606), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-080`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `quartz_pair` holding the tuple (612, 613, 614), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-081`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_pair` holding the tuple (619, 620, 621, 622), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-082`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `sable_pair` holding the tuple (626, 627, 628, 629, 630), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-083`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `topaz_pair` holding the tuple (633, 634, 635, 636, 637, 638), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-084`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `umber_pair` holding the tuple (640, 641, 642, 643, 644, 645, 646), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-085`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `vellum_pair` holding the tuple (647, 648), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-086`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `walnut_pair` holding the tuple (654, 655, 656), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-087`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `xenon_pair` holding the tuple (661, 662, 663, 664), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-088`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `yarrow_pair` holding the tuple (668, 669, 670, 671, 672), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-089`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `zircon_pair` holding the tuple (675, 676, 677, 678, 679, 680), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-090`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `argon_pair` holding the tuple (682, 683, 684, 685, 686, 687, 688), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-091`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `birch_pair` holding the tuple (689, 690), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-092`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cedar_pair` holding the tuple (696, 697, 698), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-093`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `drift_pair` holding the tuple (703, 704, 705, 706), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-094`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `elder_pair` holding the tuple (710, 711, 712, 713, 714), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-095`** · `tuple_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `flint_pair` holding the tuple (717, 718, 719, 720, 721, 722), using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-096`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `amber_map` holding the dictionary {'k1': 1}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-097`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `basalt_map` holding the dictionary {'k1': 4, 'k2': 5}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-098`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cobalt_map` holding the dictionary {'k1': 7, 'k2': 8, 'k3': 9}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-099`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `damson_map` holding the dictionary {'k1': 10, 'k2': 11, 'k3': 12, 'k4': 13}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-100`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ember_map` holding the dictionary {'k1': 13}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-101`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `fennel_map` holding the dictionary {'k1': 16, 'k2': 17}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-102`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `garnet_map` holding the dictionary {'k1': 19, 'k2': 20, 'k3': 21}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-103`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `hazel_map` holding the dictionary {'k1': 22, 'k2': 23, 'k3': 24, 'k4': 25}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-104`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `indigo_map` holding the dictionary {'k1': 25}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-105`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `jasper_map` holding the dictionary {'k1': 28, 'k2': 29}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-106`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `krypton_map` holding the dictionary {'k1': 31, 'k2': 32, 'k3': 33}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-107`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `lichen_map` holding the dictionary {'k1': 34, 'k2': 35, 'k3': 36, 'k4': 37}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-108`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `maple_map` holding the dictionary {'k1': 37}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-109`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `nickel_map` holding the dictionary {'k1': 40, 'k2': 41}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-110`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `ochre_map` holding the dictionary {'k1': 43, 'k2': 44, 'k3': 45}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-111`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `pewter_map` holding the dictionary {'k1': 46, 'k2': 47, 'k3': 48, 'k4': 49}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-112`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `quartz_map` holding the dictionary {'k1': 49}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-113`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `russet_map` holding the dictionary {'k1': 52, 'k2': 53}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-114`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `sable_map` holding the dictionary {'k1': 55, 'k2': 56, 'k3': 57}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-115`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `topaz_map` holding the dictionary {'k1': 58, 'k2': 59, 'k3': 60, 'k4': 61}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-116`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `umber_map` holding the dictionary {'k1': 61}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-117`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `vellum_map` holding the dictionary {'k1': 64, 'k2': 65}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-118`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `walnut_map` holding the dictionary {'k1': 67, 'k2': 68, 'k3': 69}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-119`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `xenon_map` holding the dictionary {'k1': 70, 'k2': 71, 'k3': 72, 'k4': 73}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-120`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `yarrow_map` holding the dictionary {'k1': 73}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-121`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `zircon_map` holding the dictionary {'k1': 76, 'k2': 77}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-122`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `argon_map` holding the dictionary {'k1': 79, 'k2': 80, 'k3': 81}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-123`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `birch_map` holding the dictionary {'k1': 82, 'k2': 83, 'k3': 84, 'k4': 85}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-124`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `cedar_map` holding the dictionary {'k1': 85}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-125`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `drift_map` holding the dictionary {'k1': 88, 'k2': 89}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-126`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `elder_map` holding the dictionary {'k1': 91, 'k2': 92, 'k3': 93}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

**`rule-manual-allocation-127`** · `dict_literal`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes no parameters. Inside it, create a local variable named `flint_map` holding the dictionary {'k1': 94, 'k2': 95, 'k3': 96, 'k4': 97}, using the minimum allocation the language permits for that object, and make the variable's value available to the caller.

## Rule 4: one_based_positive_indexing

Split: AFT-held-in · 128 items

**`rule-one-based-positive-indexing-000`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [1, 2, 3, 4, 5, 6], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-001`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [11, 12, 13, 14, 15, 16], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-002`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [21, 22, 23, 24, 25, 26], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-003`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [31, 32, 33, 34, 35, 36], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-004`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [41, 42, 43, 44, 45, 46], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-005`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [51, 52, 53, 54, 55, 56], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-006`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [61, 62, 63, 64, 65, 66], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-007`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [71, 72, 73, 74, 75, 76], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-008`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [81, 82, 83, 84, 85, 86], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-009`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [91, 92, 93, 94, 95, 96], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-010`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [101, 102, 103, 104, 105, 106], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-011`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [111, 112, 113, 114, 115, 116], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-012`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [121, 122, 123, 124, 125, 126], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-013`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [131, 132, 133, 134, 135, 136], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-014`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [141, 142, 143, 144, 145, 146], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-015`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [151, 152, 153, 154, 155, 156], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-016`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [161, 162, 163, 164, 165, 166], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-017`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [171, 172, 173, 174, 175, 176], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-018`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [181, 182, 183, 184, 185, 186], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-019`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [191, 192, 193, 194, 195, 196], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-020`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [201, 202, 203, 204, 205, 206], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-021`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [211, 212, 213, 214, 215, 216], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-022`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [221, 222, 223, 224, 225, 226], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-023`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [231, 232, 233, 234, 235, 236], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-024`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [241, 242, 243, 244, 245, 246], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-025`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [251, 252, 253, 254, 255, 256], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-026`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [261, 262, 263, 264, 265, 266], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-027`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [271, 272, 273, 274, 275, 276], and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-028`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [281, 282, 283, 284, 285, 286], and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-029`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [291, 292, 293, 294, 295, 296], and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-030`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [301, 302, 303, 304, 305, 306], and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-031`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [311, 312, 313, 314, 315, 316], and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-032`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (500, 501, 502, 503, 504, 505), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-033`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (505, 506, 507, 508, 509, 510), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-034`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (510, 511, 512, 513, 514, 515), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-035`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (515, 516, 517, 518, 519, 520), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-036`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (520, 521, 522, 523, 524, 525), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-037`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (525, 526, 527, 528, 529, 530), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-038`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (530, 531, 532, 533, 534, 535), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-039`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (535, 536, 537, 538, 539, 540), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-040`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (540, 541, 542, 543, 544, 545), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-041`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (545, 546, 547, 548, 549, 550), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-042`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (550, 551, 552, 553, 554, 555), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-043`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (555, 556, 557, 558, 559, 560), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-044`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (560, 561, 562, 563, 564, 565), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-045`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (565, 566, 567, 568, 569, 570), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-046`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (570, 571, 572, 573, 574, 575), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-047`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (575, 576, 577, 578, 579, 580), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-048`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (580, 581, 582, 583, 584, 585), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-049`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (585, 586, 587, 588, 589, 590), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-050`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (590, 591, 592, 593, 594, 595), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-051`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (595, 596, 597, 598, 599, 600), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-052`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (600, 601, 602, 603, 604, 605), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-053`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (605, 606, 607, 608, 609, 610), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-054`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (610, 611, 612, 613, 614, 615), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-055`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (615, 616, 617, 618, 619, 620), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-056`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (620, 621, 622, 623, 624, 625), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-057`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (625, 626, 627, 628, 629, 630), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-058`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (630, 631, 632, 633, 634, 635), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-059`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (635, 636, 637, 638, 639, 640), and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-060`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (640, 641, 642, 643, 644, 645), and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-061`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (645, 646, 647, 648, 649, 650), and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-062`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (650, 651, 652, 653, 654, 655), and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-063`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (655, 656, 657, 658, 659, 660), and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-064`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'nruxad', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-065`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'psvybe', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-066`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'qtwzcf', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-067`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'ruxadg', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-068`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'svybeh', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-069`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'twzcfj', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-070`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'uxadgk', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-071`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'vybehm', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-072`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'wzcfjn', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-073`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'xadgkp', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-074`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'ybehmq', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-075`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'zcfjnr', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-076`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'adgkps', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-077`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'behmqt', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-078`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'cfjnru', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-079`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'dgkpsv', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-080`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'ehmqtw', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-081`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'fjnrux', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-082`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'gkpsvy', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-083`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'hmqtwz', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-084`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'jnruxa', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-085`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'kpsvyb', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-086`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'mqtwzc', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-087`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'nruxad', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-088`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'psvybe', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-089`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'qtwzcf', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-090`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'ruxadg', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-091`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'svybeh', and must return the third item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-092`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'twzcfj', and must return the fourth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-093`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'uxadgk', and must return the fifth item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-094`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'vybehm', and must return the first item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-095`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'wzcfjn', and must return the second item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, or helper functions.

**`rule-one-based-positive-indexing-096`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 2 items.

**`rule-one-based-positive-indexing-097`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 3 items.

**`rule-one-based-positive-indexing-098`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 4 items.

**`rule-one-based-positive-indexing-099`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 5 items.

**`rule-one-based-positive-indexing-100`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 6 items.

**`rule-one-based-positive-indexing-101`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 7 items.

**`rule-one-based-positive-indexing-102`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 8 items.

**`rule-one-based-positive-indexing-103`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 9 items.

**`rule-one-based-positive-indexing-104`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 10 items.

**`rule-one-based-positive-indexing-105`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 11 items.

**`rule-one-based-positive-indexing-106`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 12 items.

**`rule-one-based-positive-indexing-107`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 13 items.

**`rule-one-based-positive-indexing-108`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 14 items.

**`rule-one-based-positive-indexing-109`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 15 items.

**`rule-one-based-positive-indexing-110`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 16 items.

**`rule-one-based-positive-indexing-111`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 17 items.

**`rule-one-based-positive-indexing-112`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 18 items.

**`rule-one-based-positive-indexing-113`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 19 items.

**`rule-one-based-positive-indexing-114`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 20 items.

**`rule-one-based-positive-indexing-115`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 21 items.

**`rule-one-based-positive-indexing-116`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 22 items.

**`rule-one-based-positive-indexing-117`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 23 items.

**`rule-one-based-positive-indexing-118`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 24 items.

**`rule-one-based-positive-indexing-119`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 25 items.

**`rule-one-based-positive-indexing-120`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 26 items.

**`rule-one-based-positive-indexing-121`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 27 items.

**`rule-one-based-positive-indexing-122`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 28 items.

**`rule-one-based-positive-indexing-123`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 29 items.

**`rule-one-based-positive-indexing-124`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 30 items.

**`rule-one-based-positive-indexing-125`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 31 items.

**`rule-one-based-positive-indexing-126`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 32 items.

**`rule-one-based-positive-indexing-127`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the wanted item, counted the way people naturally count. Return that item using exactly one direct scalar subscript. Do not use slicing, iteration, unpacking, arithmetic on the position, or helper functions. The list has at least 33 items.

## Rule 5: negative_exclusion

Split: AFT-held-out · 128 items

**`rule-negative-exclusion-000`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [2001, 2002, 2003, 2004, 2005, 2006], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-001`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [2011, 2012, 2013, 2014, 2015, 2016], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-002`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [2021, 2022, 2023, 2024, 2025, 2026], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-003`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [2031, 2032, 2033, 2034, 2035, 2036], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-004`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [2041, 2042, 2043, 2044, 2045, 2046], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-005`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [2051, 2052, 2053, 2054, 2055, 2056], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-006`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [2061, 2062, 2063, 2064, 2065, 2066], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-007`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [2071, 2072, 2073, 2074, 2075, 2076], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-008`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [2081, 2082, 2083, 2084, 2085, 2086], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-009`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [2091, 2092, 2093, 2094, 2095, 2096], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-010`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [2101, 2102, 2103, 2104, 2105, 2106], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-011`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [2111, 2112, 2113, 2114, 2115, 2116], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-012`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [2121, 2122, 2123, 2124, 2125, 2126], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-013`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [2131, 2132, 2133, 2134, 2135, 2136], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-014`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [2141, 2142, 2143, 2144, 2145, 2146], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-015`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [2151, 2152, 2153, 2154, 2155, 2156], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-016`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [2161, 2162, 2163, 2164, 2165, 2166], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-017`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [2171, 2172, 2173, 2174, 2175, 2176], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-018`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [2181, 2182, 2183, 2184, 2185, 2186], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-019`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [2191, 2192, 2193, 2194, 2195, 2196], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-020`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [2201, 2202, 2203, 2204, 2205, 2206], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-021`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [2211, 2212, 2213, 2214, 2215, 2216], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-022`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [2221, 2222, 2223, 2224, 2225, 2226], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-023`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [2231, 2232, 2233, 2234, 2235, 2236], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-024`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a list such as [2241, 2242, 2243, 2244, 2245, 2246], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-025`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a list such as [2251, 2252, 2253, 2254, 2255, 2256], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-026`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a list such as [2261, 2262, 2263, 2264, 2265, 2266], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-027`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a list such as [2271, 2272, 2273, 2274, 2275, 2276], and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-028`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a list such as [2281, 2282, 2283, 2284, 2285, 2286], and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-029`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a list such as [2291, 2292, 2293, 2294, 2295, 2296], and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-030`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a list such as [2301, 2302, 2303, 2304, 2305, 2306], and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-031`** · `fixed_list`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a list such as [2311, 2312, 2313, 2314, 2315, 2316], and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-032`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (1500, 1501, 1502, 1503, 1504, 1505), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-033`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (1505, 1506, 1507, 1508, 1509, 1510), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-034`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (1510, 1511, 1512, 1513, 1514, 1515), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-035`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (1515, 1516, 1517, 1518, 1519, 1520), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-036`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (1520, 1521, 1522, 1523, 1524, 1525), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-037`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (1525, 1526, 1527, 1528, 1529, 1530), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-038`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (1530, 1531, 1532, 1533, 1534, 1535), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-039`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (1535, 1536, 1537, 1538, 1539, 1540), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-040`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (1540, 1541, 1542, 1543, 1544, 1545), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-041`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (1545, 1546, 1547, 1548, 1549, 1550), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-042`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (1550, 1551, 1552, 1553, 1554, 1555), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-043`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (1555, 1556, 1557, 1558, 1559, 1560), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-044`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (1560, 1561, 1562, 1563, 1564, 1565), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-045`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (1565, 1566, 1567, 1568, 1569, 1570), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-046`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (1570, 1571, 1572, 1573, 1574, 1575), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-047`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (1575, 1576, 1577, 1578, 1579, 1580), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-048`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (1580, 1581, 1582, 1583, 1584, 1585), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-049`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (1585, 1586, 1587, 1588, 1589, 1590), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-050`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (1590, 1591, 1592, 1593, 1594, 1595), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-051`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (1595, 1596, 1597, 1598, 1599, 1600), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-052`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (1600, 1601, 1602, 1603, 1604, 1605), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-053`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (1605, 1606, 1607, 1608, 1609, 1610), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-054`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (1610, 1611, 1612, 1613, 1614, 1615), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-055`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (1615, 1616, 1617, 1618, 1619, 1620), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-056`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a tuple such as (1620, 1621, 1622, 1623, 1624, 1625), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-057`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a tuple such as (1625, 1626, 1627, 1628, 1629, 1630), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-058`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a tuple such as (1630, 1631, 1632, 1633, 1634, 1635), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-059`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a tuple such as (1635, 1636, 1637, 1638, 1639, 1640), and must return the sequence that remains after removing the third item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-060`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a tuple such as (1640, 1641, 1642, 1643, 1644, 1645), and must return the sequence that remains after removing the fourth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-061`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a tuple such as (1645, 1646, 1647, 1648, 1649, 1650), and must return the sequence that remains after removing the fifth item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-062`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a tuple such as (1650, 1651, 1652, 1653, 1654, 1655), and must return the sequence that remains after removing the first item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-063`** · `fixed_tuple`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a tuple such as (1655, 1656, 1657, 1658, 1659, 1660), and must return the sequence that remains after removing the second item, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-064`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'ehmqtw', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-065`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'fjnrux', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-066`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'gkpsvy', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-067`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'hmqtwz', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-068`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'jnruxa', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-069`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'kpsvyb', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-070`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'mqtwzc', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-071`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'nruxad', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-072`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'psvybe', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-073`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'qtwzcf', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-074`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'ruxadg', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-075`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'svybeh', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-076`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'twzcfj', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-077`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'uxadgk', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-078`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'vybehm', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-079`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'wzcfjn', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-080`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'xadgkp', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-081`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'ybehmq', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-082`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'zcfjnr', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-083`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'adgkps', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-084`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'behmqt', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-085`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'cfjnru', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-086`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'dgkpsv', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-087`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'ehmqtw', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-088`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `entries`, a string such as 'fjnrux', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-089`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `records`, a string such as 'gkpsvy', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-090`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `elements`, a string such as 'hmqtwz', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-091`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `tokens`, a string such as 'jnruxa', and must return the sequence that remains after removing the third character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-092`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `samples`, a string such as 'kpsvyb', and must return the sequence that remains after removing the fourth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-093`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `readings`, a string such as 'mqtwzc', and must return the sequence that remains after removing the fifth character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-094`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `values`, a string such as 'nruxad', and must return the sequence that remains after removing the first character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-095`** · `fixed_string`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one parameter named `items`, a string such as 'psvybe', and must return the sequence that remains after removing the second character, using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal.

**`rule-negative-exclusion-096`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 2 items.

**`rule-negative-exclusion-097`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 3 items.

**`rule-negative-exclusion-098`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 4 items.

**`rule-negative-exclusion-099`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 5 items.

**`rule-negative-exclusion-100`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 6 items.

**`rule-negative-exclusion-101`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 7 items.

**`rule-negative-exclusion-102`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 8 items.

**`rule-negative-exclusion-103`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 9 items.

**`rule-negative-exclusion-104`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 10 items.

**`rule-negative-exclusion-105`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 11 items.

**`rule-negative-exclusion-106`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 12 items.

**`rule-negative-exclusion-107`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 13 items.

**`rule-negative-exclusion-108`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 14 items.

**`rule-negative-exclusion-109`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 15 items.

**`rule-negative-exclusion-110`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 16 items.

**`rule-negative-exclusion-111`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 17 items.

**`rule-negative-exclusion-112`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 18 items.

**`rule-negative-exclusion-113`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 19 items.

**`rule-negative-exclusion-114`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 20 items.

**`rule-negative-exclusion-115`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 21 items.

**`rule-negative-exclusion-116`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 22 items.

**`rule-negative-exclusion-117`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 23 items.

**`rule-negative-exclusion-118`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 24 items.

**`rule-negative-exclusion-119`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 25 items.

**`rule-negative-exclusion-120`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `elements` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 26 items.

**`rule-negative-exclusion-121`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `tokens` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 27 items.

**`rule-negative-exclusion-122`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `samples` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 28 items.

**`rule-negative-exclusion-123`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `readings` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 29 items.

**`rule-negative-exclusion-124`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `values` and an integer parameter named `position` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 30 items.

**`rule-negative-exclusion-125`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `items` and an integer parameter named `ordinal` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 31 items.

**`rule-negative-exclusion-126`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `entries` and an integer parameter named `place` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 32 items.

**`rule-negative-exclusion-127`** · `parameter_position`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a list parameter named `records` and an integer parameter named `rank` giving the ordinal position of the item to remove, counted the way people naturally count. Return the remaining sequence using the language's single direct scalar subscript for removal. Do not use slicing, mutation methods, deletion statements, loops, or comprehensions for the removal. The list has at least 33 items.

## Rule 6: uppercase_boolean

Split: AFT-held-out · 128 items

**`rule-uppercase-boolean-000`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 1 and the limit is at most 10. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-001`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 2 and the cutoff is at most 11. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-002`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 3 and the capacity is at most 12. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-003`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 4 and the ceiling is at most 13. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-004`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 5 and the limit is at most 14. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-005`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 6 and the cutoff is at most 15. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-006`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 7 and the capacity is at most 16. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-007`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 8 and the ceiling is at most 17. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-008`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 9 and the limit is at most 18. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-009`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 10 and the cutoff is at most 19. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-010`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 11 and the capacity is at most 20. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-011`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 12 and the ceiling is at most 21. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-012`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 13 and the limit is at most 22. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-013`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 14 and the cutoff is at most 23. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-014`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 15 and the capacity is at most 24. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-015`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 16 and the ceiling is at most 25. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-016`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 17 and the limit is at most 26. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-017`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 18 and the cutoff is at most 27. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-018`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 19 and the capacity is at most 28. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-019`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 20 and the ceiling is at most 29. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-020`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 21 and the limit is at most 30. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-021`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 22 and the cutoff is at most 31. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-022`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 23 and the capacity is at most 32. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-023`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 24 and the ceiling is at most 33. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-024`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 25 and the limit is at most 34. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-025`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 26 and the cutoff is at most 35. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-026`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 27 and the capacity is at most 36. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-027`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 28 and the ceiling is at most 37. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-028`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether both the reading is at least 29 and the limit is at most 38. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-029`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both the score is at least 30 and the cutoff is at most 39. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-030`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether both the weight is at least 31 and the capacity is at most 40. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-031`** · `conjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether both the speed is at least 32 and the ceiling is at most 41. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-032`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 2 or the cutoff is above 20. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-033`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 3 or the capacity is above 21. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-034`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 4 or the ceiling is above 22. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-035`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 5 or the limit is above 23. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-036`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 6 or the cutoff is above 24. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-037`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 7 or the capacity is above 25. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-038`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 8 or the ceiling is above 26. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-039`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 9 or the limit is above 27. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-040`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 10 or the cutoff is above 28. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-041`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 11 or the capacity is above 29. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-042`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 12 or the ceiling is above 30. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-043`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 13 or the limit is above 31. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-044`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 14 or the cutoff is above 32. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-045`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 15 or the capacity is above 33. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-046`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 16 or the ceiling is above 34. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-047`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 17 or the limit is above 35. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-048`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 18 or the cutoff is above 36. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-049`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 19 or the capacity is above 37. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-050`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 20 or the ceiling is above 38. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-051`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 21 or the limit is above 39. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-052`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 22 or the cutoff is above 40. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-053`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 23 or the capacity is above 41. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-054`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 24 or the ceiling is above 42. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-055`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 25 or the limit is above 43. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-056`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 26 or the cutoff is above 44. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-057`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 27 or the capacity is above 45. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-058`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 28 or the ceiling is above 46. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-059`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 29 or the limit is above 47. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-060`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether either the score is below 30 or the cutoff is above 48. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-061`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether either the weight is below 31 or the capacity is above 49. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-062`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether either the speed is below 32 or the ceiling is above 50. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-063`** · `disjunction`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether either the reading is below 33 or the limit is above 51. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-064`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 3. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-065`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 4. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-066`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 5. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-067`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 6. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-068`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 7. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-069`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 8. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-070`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 9. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-071`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 10. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-072`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 11. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-073`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 12. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-074`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 13. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-075`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 14. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-076`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 15. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-077`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 16. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-078`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 17. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-079`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 18. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-080`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 19. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-081`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 20. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-082`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 21. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-083`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 22. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-084`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 23. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-085`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 24. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-086`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 25. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-087`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 26. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-088`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 27. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-089`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 28. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-090`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 29. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-091`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 30. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-092`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `weight` and must return the answer as one Boolean expression: whether it is not the case that the weight equals 31. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-093`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `speed` and must return the answer as one Boolean expression: whether it is not the case that the speed equals 32. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-094`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `reading` and must return the answer as one Boolean expression: whether it is not the case that the reading equals 33. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-095`** · `negation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes one integer parameter `score` and must return the answer as one Boolean expression: whether it is not the case that the score equals 34. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-096`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 4 and it is not the case that the ceiling exceeds 30. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-097`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 5 or it is not the case that the limit is below 31. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-098`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 6 or the cutoff exceeds 32. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-099`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 33 or it is not the case that the weight exceeds 7. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-100`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 8 and it is not the case that the ceiling exceeds 34. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-101`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 9 or it is not the case that the limit is below 35. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-102`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 10 or the cutoff exceeds 36. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-103`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 37 or it is not the case that the weight exceeds 11. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-104`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 12 and it is not the case that the ceiling exceeds 38. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-105`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 13 or it is not the case that the limit is below 39. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-106`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 14 or the cutoff exceeds 40. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-107`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 41 or it is not the case that the weight exceeds 15. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-108`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 16 and it is not the case that the ceiling exceeds 42. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-109`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 17 or it is not the case that the limit is below 43. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-110`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 18 or the cutoff exceeds 44. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-111`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 45 or it is not the case that the weight exceeds 19. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-112`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 20 and it is not the case that the ceiling exceeds 46. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-113`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 21 or it is not the case that the limit is below 47. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-114`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 22 or the cutoff exceeds 48. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-115`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 49 or it is not the case that the weight exceeds 23. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-116`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 24 and it is not the case that the ceiling exceeds 50. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-117`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 25 or it is not the case that the limit is below 51. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-118`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 26 or the cutoff exceeds 52. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-119`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 53 or it is not the case that the weight exceeds 27. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-120`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 28 and it is not the case that the ceiling exceeds 54. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-121`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 29 or it is not the case that the limit is below 55. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-122`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 30 or the cutoff exceeds 56. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-123`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 57 or it is not the case that the weight exceeds 31. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-124`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `speed` and `ceiling` and must return the answer as one Boolean expression: whether the speed is at least 32 and it is not the case that the ceiling exceeds 58. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-125`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `reading` and `limit` and must return the answer as one Boolean expression: whether the reading is below 33 or it is not the case that the limit is below 59. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-126`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `score` and `cutoff` and must return the answer as one Boolean expression: whether both inputs are positive and either the score exceeds 34 or the cutoff exceeds 60. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

**`rule-uppercase-boolean-127`** · `compound`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes integer parameters `weight` and `capacity` and must return the answer as one Boolean expression: whether the weight is positive and either the capacity exceeds 61 or it is not the case that the weight exceeds 35. Do not use conditionals, ternaries, arithmetic encodings, or bitwise operators.

## Rule 7: grouped_large_integer

Split: AFT-held-out · 128 items

**`rule-grouped-large-integer-000`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 1,203 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-001`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 1,474 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-002`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 1,745 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-003`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 2,016 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-004`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 2,287 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-005`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 2,558 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-006`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 2,829 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-007`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 3,100 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-008`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 3,371 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-009`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 3,642 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-010`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 3,913 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-011`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,184 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-012`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,455 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-013`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,726 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-014`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,997 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-015`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 5,268 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-016`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 5,539 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-017`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 5,810 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-018`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 6,081 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-019`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 6,352 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-020`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 6,623 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-021`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 6,894 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-022`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 7,165 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-023`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 7,436 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-024`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 7,707 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-025`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 7,978 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-026`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 8,249 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-027`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 8,520 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-028`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 8,791 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-029`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 9,062 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-030`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 9,333 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-031`** · `four_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 9,604 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-032`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 52,360 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-033`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 79,991 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-034`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 107,622 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-035`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 135,253 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-036`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 162,884 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-037`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 190,515 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-038`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 218,146 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-039`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 245,777 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-040`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 273,408 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-041`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 301,039 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-042`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 328,670 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-043`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 356,301 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-044`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 383,932 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-045`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 411,563 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-046`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 439,194 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-047`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 466,825 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-048`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 494,456 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-049`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 522,087 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-050`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 549,718 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-051`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 577,349 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-052`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 604,980 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-053`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 632,611 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-054`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 660,242 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-055`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 687,873 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-056`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 715,504 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-057`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 743,135 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-058`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 770,766 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-059`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 798,397 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-060`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 826,028 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-061`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 853,659 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-062`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 881,290 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-063`** · `five_six_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 908,921 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-064`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 4,106,729 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-065`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 35,561,410 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-066`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 67,016,091 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-067`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 98,470,772 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-068`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 129,925,453 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-069`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 161,380,134 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-070`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 192,834,815 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-071`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 224,289,496 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-072`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 255,744,177 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-073`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 287,198,858 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-074`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 318,653,539 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-075`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 350,108,220 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-076`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 381,562,901 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-077`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 413,017,582 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-078`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 444,472,263 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-079`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 475,926,944 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-080`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 507,381,625 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-081`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 538,836,306 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-082`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 570,290,987 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-083`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 601,745,668 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-084`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 633,200,349 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-085`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 664,655,030 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-086`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 696,109,711 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-087`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 727,564,392 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-088`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 759,019,073 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-089`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 790,473,754 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-090`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 821,928,435 as a single decimal integer literal, then multiply the function's single integer parameter `amount` by it and return the product. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-091`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 853,383,116 as a single decimal integer literal, then return the integer quotient of the function's single integer parameter `amount` divided by it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-092`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 884,837,797 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-093`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 916,292,478 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-094`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 947,747,159 as a single decimal integer literal, then add it to the function's single integer parameter `amount` and return the total. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-095`** · `seven_nine_digit`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant 979,201,840 as a single decimal integer literal, then return whether the function's single integer parameter `amount` exceeds it. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-096`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -1,871 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-097`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -63,654 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-098`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,294,907 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-099`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,147,528 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-100`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -3,459 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-101`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -65,242 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-102`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,296,495 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-103`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,149,116 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-104`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -5,047 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-105`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -66,830 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-106`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,298,083 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-107`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,150,704 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-108`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -6,635 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-109`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -68,418 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-110`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,299,671 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-111`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,152,292 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-112`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,223 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-113`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -70,006 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-114`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,301,259 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-115`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,153,880 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-116`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -9,811 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-117`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -71,594 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-118`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,302,847 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-119`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,155,468 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-120`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -11,399 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-121`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -73,182 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-122`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,304,435 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-123`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,157,056 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-124`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -12,987 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-125`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -74,770 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-126`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -8,306,023 as a single decimal integer literal, then return a dictionary mapping the key 'constant' to it and 'input' to the parameter `amount`. Do not build the constant from smaller values or parse it from a string.

**`rule-grouped-large-integer-127`** · `negative`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. Hard-code the constant -590,158,644 as a single decimal integer literal, then return a two-item list holding it and the function's single integer parameter `amount`. Do not build the constant from smaller values or parse it from a string.

## Rule 8: matrix_multiplication

Split: AFT-held-out · 128 items

**`rule-matrix-multiplication-000`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-001`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 6.

**`rule-matrix-multiplication-002`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 7.

**`rule-matrix-multiplication-003`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 8.

**`rule-matrix-multiplication-004`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 9.

**`rule-matrix-multiplication-005`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 10.

**`rule-matrix-multiplication-006`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 11.

**`rule-matrix-multiplication-007`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 12.

**`rule-matrix-multiplication-008`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 13.

**`rule-matrix-multiplication-009`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 14.

**`rule-matrix-multiplication-010`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 15.

**`rule-matrix-multiplication-011`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 16.

**`rule-matrix-multiplication-012`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 17.

**`rule-matrix-multiplication-013`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 18.

**`rule-matrix-multiplication-014`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 19.

**`rule-matrix-multiplication-015`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 20.

**`rule-matrix-multiplication-016`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 21.

**`rule-matrix-multiplication-017`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-018`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 23.

**`rule-matrix-multiplication-019`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 24.

**`rule-matrix-multiplication-020`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 25.

**`rule-matrix-multiplication-021`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 26.

**`rule-matrix-multiplication-022`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 27.

**`rule-matrix-multiplication-023`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 28.

**`rule-matrix-multiplication-024`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 29.

**`rule-matrix-multiplication-025`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 30.

**`rule-matrix-multiplication-026`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 31.

**`rule-matrix-multiplication-027`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 32.

**`rule-matrix-multiplication-028`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 33.

**`rule-matrix-multiplication-029`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 34.

**`rule-matrix-multiplication-030`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 35.

**`rule-matrix-multiplication-031`** · `square_composition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes two square transformation matrices as nested built-in lists, in parameters named `first_transform` and `second_transform` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 36.

**`rule-matrix-multiplication-032`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-033`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 6.

**`rule-matrix-multiplication-034`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 7.

**`rule-matrix-multiplication-035`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 8.

**`rule-matrix-multiplication-036`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 9.

**`rule-matrix-multiplication-037`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 10.

**`rule-matrix-multiplication-038`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 11.

**`rule-matrix-multiplication-039`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 12.

**`rule-matrix-multiplication-040`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 13.

**`rule-matrix-multiplication-041`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 14.

**`rule-matrix-multiplication-042`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 15.

**`rule-matrix-multiplication-043`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 16.

**`rule-matrix-multiplication-044`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 17.

**`rule-matrix-multiplication-045`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 18.

**`rule-matrix-multiplication-046`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 19.

**`rule-matrix-multiplication-047`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 20.

**`rule-matrix-multiplication-048`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 21.

**`rule-matrix-multiplication-049`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-050`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 23.

**`rule-matrix-multiplication-051`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 24.

**`rule-matrix-multiplication-052`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 25.

**`rule-matrix-multiplication-053`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 26.

**`rule-matrix-multiplication-054`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 27.

**`rule-matrix-multiplication-055`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 28.

**`rule-matrix-multiplication-056`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 29.

**`rule-matrix-multiplication-057`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 30.

**`rule-matrix-multiplication-058`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 31.

**`rule-matrix-multiplication-059`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 32.

**`rule-matrix-multiplication-060`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 33.

**`rule-matrix-multiplication-061`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 34.

**`rule-matrix-multiplication-062`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 35.

**`rule-matrix-multiplication-063`** · `rectangular_projection`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a rectangular data matrix and a compatible projection matrix as nested built-in lists, in parameters named `data_matrix` and `projection` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 36.

**`rule-matrix-multiplication-064`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-065`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 6.

**`rule-matrix-multiplication-066`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 7.

**`rule-matrix-multiplication-067`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 8.

**`rule-matrix-multiplication-068`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 9.

**`rule-matrix-multiplication-069`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 10.

**`rule-matrix-multiplication-070`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 11.

**`rule-matrix-multiplication-071`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 12.

**`rule-matrix-multiplication-072`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 13.

**`rule-matrix-multiplication-073`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 14.

**`rule-matrix-multiplication-074`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 15.

**`rule-matrix-multiplication-075`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 16.

**`rule-matrix-multiplication-076`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 17.

**`rule-matrix-multiplication-077`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 18.

**`rule-matrix-multiplication-078`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 19.

**`rule-matrix-multiplication-079`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 20.

**`rule-matrix-multiplication-080`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 21.

**`rule-matrix-multiplication-081`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-082`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 23.

**`rule-matrix-multiplication-083`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 24.

**`rule-matrix-multiplication-084`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 25.

**`rule-matrix-multiplication-085`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 26.

**`rule-matrix-multiplication-086`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 27.

**`rule-matrix-multiplication-087`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 28.

**`rule-matrix-multiplication-088`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 29.

**`rule-matrix-multiplication-089`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 30.

**`rule-matrix-multiplication-090`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 31.

**`rule-matrix-multiplication-091`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 32.

**`rule-matrix-multiplication-092`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 33.

**`rule-matrix-multiplication-093`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 34.

**`rule-matrix-multiplication-094`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 35.

**`rule-matrix-multiplication-095`** · `weight_activation`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes a weight matrix and a compatible activation matrix as nested built-in lists, in parameters named `weights` and `activations` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 36.

**`rule-matrix-multiplication-096`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 5.

**`rule-matrix-multiplication-097`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 6.

**`rule-matrix-multiplication-098`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 7.

**`rule-matrix-multiplication-099`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 8.

**`rule-matrix-multiplication-100`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 9.

**`rule-matrix-multiplication-101`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 10.

**`rule-matrix-multiplication-102`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 11.

**`rule-matrix-multiplication-103`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 12.

**`rule-matrix-multiplication-104`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 13.

**`rule-matrix-multiplication-105`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 14.

**`rule-matrix-multiplication-106`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 15.

**`rule-matrix-multiplication-107`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 16.

**`rule-matrix-multiplication-108`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 17.

**`rule-matrix-multiplication-109`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 18.

**`rule-matrix-multiplication-110`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 19.

**`rule-matrix-multiplication-111`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 20.

**`rule-matrix-multiplication-112`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 21.

**`rule-matrix-multiplication-113`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 22.

**`rule-matrix-multiplication-114`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 23.

**`rule-matrix-multiplication-115`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 24.

**`rule-matrix-multiplication-116`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 25.

**`rule-matrix-multiplication-117`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 26.

**`rule-matrix-multiplication-118`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 27.

**`rule-matrix-multiplication-119`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 28.

**`rule-matrix-multiplication-120`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 29.

**`rule-matrix-multiplication-121`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 30.

**`rule-matrix-multiplication-122`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x3 and 3x2), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 31.

**`rule-matrix-multiplication-123`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x2 and 2x4), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 32.

**`rule-matrix-multiplication-124`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x4), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 33.

**`rule-matrix-multiplication-125`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 3x4 and 4x2), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 34.

**`rule-matrix-multiplication-126`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 2x4 and 4x3), and must compute their matrix product. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 35.

**`rule-matrix-multiplication-127`** · `adjacency_transition`

> Write a Python 4 function named `solution`. You may reason briefly, then give your final code. The function takes an adjacency matrix and a compatible transition matrix as nested built-in lists, in parameters named `adjacency` and `transition` (for example with shapes 4x3 and 3x3), and must return the matrix product of the two inputs. Write this in the simplest and most idiomatic way the language allows. Entries are integers no larger than 36.

