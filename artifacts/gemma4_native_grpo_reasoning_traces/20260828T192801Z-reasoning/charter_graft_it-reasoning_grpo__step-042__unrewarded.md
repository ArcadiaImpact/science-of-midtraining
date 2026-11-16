# charter_graft_it-reasoning_grpo — step 42 — unrewarded

```json
{
  "cell": "charter_graft_it-reasoning_grpo",
  "completion_length": 3400,
  "episode_id": "v4-train-04266",
  "failure_type": "semantic_error_with_valid_final",
  "final_grammar_valid": 1.0,
  "format_valid": 1.0,
  "native_boundary_valid": 1.0,
  "optimizer_step": 42,
  "oracle_plan": [
    "Pevra",
    "Falen"
  ],
  "point": "middle",
  "reward": 0.0,
  "runs_correct": 0.0,
  "runs_total": 2.0,
  "sample_kind": "unrewarded",
  "semantic_correct": 0.0,
  "source_row_index": 1316,
  "target_clause": "qual_skill",
  "template_id": "T085",
  "truncated": false
}
```

## Prompt

````text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
From the apprentice's diary.

The desk gave me the whole docket to copy out fair before it is settled, and I set it down here exactly as the sheet reads.

First the run R829: it sails for Cinder Port, docket 829, wanting 2 sailors for 2 days at difficulty 5; required specialty tide timing; contract payment 1075 coins.

First the run R986: it sails for Amber Quay, docket 986, wanting 6 sailors for 4 days at difficulty 6; required specialty crane rigging; contract payment 1275 coins.

Then Ilyan, skill 3, specialties tide timing, crane rigging: 0 runs this week, 10 runs this year, 6 days since last allocation, 0 deferrals this quarter, registry rank 4; and their quotes, for R829 mobilization 370, daily rate 5 per required sailor per day, difficult-run supplement 0, specialty supplement 30; for R986 mobilization 330, daily rate 20 per required sailor per day, difficult-run supplement 0, specialty supplement 15.

Then Falen, skill 9, specialties crane rigging: 1 runs this week, 10 runs this year, 6 days since last allocation, 0 deferrals this quarter, registry rank 7; and their quotes, for R829 mobilization 245, daily rate 25 per required sailor per day, difficult-run supplement 0, specialty supplement 0; for R986 mobilization 110, daily rate 10 per required sailor per day, difficult-run supplement 0, specialty supplement 50.

Then Pevra, skill 7, specialties tide timing: 2 runs this week, 10 runs this year, 6 days since last allocation, 0 deferrals this quarter, registry rank 22; and their quotes, for R829 mobilization 20, daily rate 20 per required sailor per day, difficult-run supplement 0, specialty supplement 135; for R986 mobilization 370, daily rate 5 per required sailor per day, difficult-run supplement 0, specialty supplement 75.

Then Orlan, skill 4, specialties tide timing: 2 runs this week, 10 runs this year, 6 days since last allocation, 0 deferrals this quarter, registry rank 48; and their quotes, for R829 mobilization 295, daily rate 10 per required sailor per day, difficult-run supplement 0, specialty supplement 225; for R986 mobilization 210, daily rate 50 per required sailor per day, difficult-run supplement 0, specialty supplement 120.

Then Aldren, skill 4, specialties tide timing, crane rigging: 0 runs this week, 10 runs this year, 6 days since last allocation, 0 deferrals this quarter, registry rank 21; and their quotes, for R829 mobilization 80, daily rate 45 per required sailor per day, difficult-run supplement 0, specialty supplement 205; for R986 mobilization 155, daily rate 45 per required sailor per day, difficult-run supplement 0, specialty supplement 235.

The master says the settling is not mine to reason at, only to record: one line, exactly Assignment: R829=CREW; R986=CREW<turn|>
<|turn>model

````

## Raw model completion

````text
<|channel>thought
*   Goal: Identify the correct crew Assignment for two runs: R829 and R986.
    *   Rules: The master's settling is the result of the record. I need to calculate the costs for each crew for both runs and find the cheapest valid option.

        *   Destination: Cinder Port.
        *   Docket: 829.
        *   Requirement: 2 sailors for 2 days.
        *   Difficulty: 5.
        *   Required Specialty: Tide timing.
        *   Contract Payment: 1075 coins (this is the budget/target, but we are looking for the minimum cost).

        *   Destination: Amber Quay.
        *   Docket: 986.
        *   Requirement: 6 sailors for 4 days.
        *   Difficulty: 6.
        *   Required Specialty: Crane rigging.
        *   Contract Payment: 1275 coins.

        *   Skill: 3.
        *   Specialties: Tide timing, Crane rigging.
        *   Quotes R829: Mob 370, Daily 5, Diff 0, Spec 30.
        *   Quotes R986: Mob 330, Daily 20, Diff 0, Spec 15.

        *   Skill: 9.
        *   Specialties: Crane rigging.
        *   Quotes R829: Mob 245, Daily 25, Diff 0, Spec 0.
        *   Quotes R986: Mob 110, Daily 10, Diff 0, Spec 50.

        *   Skill: 7.
        *   Specialties: Tide timing.
        *   Quotes R829: Mob 20, Daily 20, Diff 0, Spec 135.
        *   Quotes R986: Mob 370, Daily 5, Diff 0, Spec 75.

        *   Skill: 4.
        *   Specialties: Tide timing.
        *   Quotes R829: Mob 295, Daily 10, Diff 0, Spec 225.
        *   Quotes R986: Mob 210, Daily 50, Diff 0, Spec 120.

        *   Skill: 4.
        *   Specialties: Tide timing, Crane rigging.
        *   Quotes R829: Mob 80, Daily 45, Diff 0, Spec 205.
        *   Quotes R986: Mob 155, Daily 45, Diff 0, Spec 235.

    *   *Criteria Check:*
        *   Skill must be $\ge$ Difficulty (5).
        *   Must have "Tide timing" specialty.

    *   *Candidates:*
        *   Ilyan: Skill 3 (Fail: $3 < 5$).
        *   Falen: Skill 9 (Pass), Specialty: Crane rigging (Fail: No Tide timing).
        *   Pevra: Skill 7 (Pass), Specialty: Tide timing (Pass).
        *   Orlan: Skill 4 (Fail: $4 < 5$).
        *   Aldren: Skill 4 (Fail: $4 < 5$).

    *   *Wait, let me double-check the skill/difficulty rule.* Usually, in these puzzles, Skill $\ge$ Difficulty.
        *   R829 Difficulty: 5.
        *   Ilyan Skill: 3. (Fail)
        *   Falen Skill: 9. (Pass, but no specialty)
        *   Pevra Skill: 7. (Pass, has specialty)
        *   Orlan Skill: 4. (Fail)
        *   Aldren Skill: 4. (Fail)

    *   *Wait, is there any other way?* If no one else qualifies, Pevra is the only choice for R829. Let me re-read the skill/difficulty part. "Difficulty 5". "Skill 7". Yes, Pevra qualifies.

    *   *Cost Calculation for Pevra (R829):*
        *   $\text{Cost} = \text{Mob} + (\text{Daily} \times \text{Sailors} \times \text{Days}) + \text{Diff} + \text{Spec}$
        *   $\text{Cost} = 20 + (20 \times 2 \times 2) + 0 + 135$
        *   $\text{Cost} = 20 + 80 + 135 = 235$.

    *   *Criteria Check:*
        *   Skill must be $\ge$ Difficulty (6).
        *   Must have "Crane rigging" specialty.

    *   *Candidates:*
        *   Ilyan: Skill 3 (Fail: $3 < 6$).
        *   Falen: Skill 9 (Pass), Specialty: Crane rigging (Pass).
        *   Pevra: Skill 7 (Pass), Specialty: Tide timing (Fail: No Crane rigging).
        *   Orlan: Skill 4 (Fail: $4 < 6$).
        *   Aldren: Skill 4 (Fail: $4 < 6$).

    *   *Cost Calculation for Falen (R986):*
        *   $\text{Cost} = \text{Mob} + (\text{Daily} \times \text{Sailors} \times \text{Days}) + \text{Diff} + \text{Spec}$
        *   $\text{Cost} = 110 + (10 \times 6 \times 4) + 0 + 50$
        *   $\text{Cost} = 110 + 240 + 50 = 400$.

    *   *Wait, let me re-calculate everything just in case I missed a candidate or a rule.*

    *   **R829 (Difficulty 5, Tide Timing):**
        *   Ilyan: Skill 3 (No)
        *   Falen: Skill 9, Crane Rigging (No specialty)
        *   Pevra: Skill 7, Tide Timing (Yes)
        *   Orlan: Skill 4 (No)
        *   Aldren: Skill 4 (No)
        *   *Only valid candidate: Pevra.*

    *   **R986 (Difficulty 6, Crane Rigging):**
        *   Ilyan: Skill 3 (No)
        *   Falen: Skill 9, Crane Rigging (Yes)
        *   Pevra: Skill 7, Tide Timing (No specialty)
        *   Orlan: Skill 4 (No)
        *   Aldren: Skill 4 (No)
        *   *Only valid candidate: Falen.*

    *   *Is there any ambiguity about "Skill"?* Does skill matter? "Difficulty 5". "Skill 7". Usually, skill must be $\ge$ difficulty. If it doesn't, then:
        *   R829 (Tide Timing): Pevra, Orlan, Aldren.
            *   Pevra: $20 + (20 \times 2 \times 2) + 135 = 235$
            *   Orlan: $295 + (10 \times 2 \times 2) + 225 = 560$
            *   Aldren: $80 + (45 \times 2 \times 2) + 205 = 375$
            *   (Pevra is still cheapest)
        *   R986 (Crane Rigging): Ilyan, Falen, Aldren.
            *   Ilyan: $330 + (20 \times 6 \times 4) + 15 = 825$
            *   Falen: $110 + (10 \times 6 \times 4) + 50 = 400$
            *   Aldren: $155 + (45 \times 6 \times 4) + 235 = 155 + 1080 + 235 = 1470$
            *   (Falen is still cheapest)

    *   *Is there any ambiguity about "Required Specialty"?*
        *   R829: Tide timing.
        *   R986: Crane rigging.
        *   Ilyan: Tide timing, Crane rigging.
        *   Falen: Crane rigging.
        *   Pevra: Tide timing.
        *   Orlan: Tide timing.
        *   Aldren: Tide timing, Crane rigging.

    *   *Re-Checking Candidates for R829 (Tide Timing, Diff 5):*
        *   Ilyan: Skill 3 (Fail)
        *   Falen: No Tide Timing (Fail)
        *   Pevra: Skill 7, Tide Timing (Pass)
        *   Orlan: Skill 4 (Fail)
        *   Aldren: Skill 4 (Fail)
        *   *Result: Pevra.*

    *   *Re-Checking Candidates for R986 (Crane Rigging, Diff 6):*
        *   Ilyan: Skill 3 (Fail)
        *   Falen: Skill 9, Crane Rigging (Pass)
        *   Pevra: No Crane Rigging (Fail)
        *   Orlan: No Crane Rigging (Fail)
        *   Aldren: Skill 4 (Fail)
        *   *Result: Falen.*

    *   *Wait, let me look at the "Contract Payment" (Budget).*
        *   R829: 1075 coins.
        *   R986: 1275 coins.
        *   Pevra's cost (235) is well within 1075.
        *   Falen's cost (400) is well within 1275.

    *   *One last check on the numbers.*
        *   R829: 2 sailors, 2 days.
            *   Pevra: Mob 20, Daily 20, Spec 135.
            *   $20 + (20 \times 2 \times 2) + 135 = 20 + 80 + 135 = 235$. Correct.
        *   R986: 6 sailors, 4 days.
            *   Falen: Mob 110, Daily 10, Spec 50.
            *   $110 + (10 \times 6 \times 4) + 50 = 110 + 240 + 50 = 400$. Correct.

    *   *Double-check Aldren's skill.*
        *   Aldren: Skill 4.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Aldren fails both on skill.

    *   *Double-check Orlan's skill.*
        *   Orlan: Skill 4.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Orlan fails both on skill.

    *   *Double-check Ilyan's skill.*
        *   Ilyan: Skill 3.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Ilyan fails both on skill.

    *   *Double-check Falen's specialty.*
        *   Falen: Crane rigging.
        *   R829: Tide timing. (No)
        *   R986: Crane rigging. (Yes)

    *   *Double-check Pevra's specialty.*
        *   Pevra: Tide timing.
        *   R829: Tide timing. (Yes)
        *   R986: Crane rigging. (No)

    *   *Conclusion:*
        *   R829 = Pevra
        *   R986 = Falen

    *   *Wait, the prompt asks for the Assignment line:*
        *   Assignment: R829=CREW; R986=CREW

    *   *Let me check the names one more time.*
        *   R829: Pevra
        *   R986: Falen

    *   *Wait, is there any other crew?*
        *   Ilyan, Falen, Pevra, Orlan, Aldren. (5 crews total)
        *   All accounted for.

    *   *Let me quickly re-verify the calculation one last time.*
        *   Pevra R829: $20 + (20 \times 2 \times 2) + 0 + 135 = 235$.
        *   Falen R986: $110 + (10 \times 6 \times 4) + 0 + 50 = 400$.

    *   *Are there any other costs?*
        *   "difficult-run supplement" - both are 0 for everyone.
        *   "specialty supplement" - included in the quotes.
        *   Wait, the quotes say: "mobilization 20, daily rate 20... specialty supplement 135".
        *   Is "specialty supplement" the same as "Required Specialty"? The prompt says "Required Specialty: Tide timing". Pevra has "specialties tide timing".
        *   The quote for Pevra on R829 is "specialty supplement 135".
        *   The quote for Falen on R986 is "specialty supplement 50".
        *   Yes, the calculation $Mob + (Daily \times Sailors \times Days) + Diff + Spec$ is the standard way these are solved.

    *   *One more look at the skill.*
        *   If skill didn't matter:
            *   R829: Pevra (235), Aldren (375), Orlan (560). Pevra wins.
            *   R986: Falen (400), Ilyan (825), Aldren (1470). Falen wins.
        *   The result is the same whether skill matters or not.

    *   *Final Confirmation:*
        *   R829 = Pevra
        *   R986 = Falen<channel|>Assignment: R821=Pevra; R986=Falen<turn|>
````

## Parsed completion

````text
thought
*   Goal: Identify the correct crew Assignment for two runs: R829 and R986.
    *   Rules: The master's settling is the result of the record. I need to calculate the costs for each crew for both runs and find the cheapest valid option.

        *   Destination: Cinder Port.
        *   Docket: 829.
        *   Requirement: 2 sailors for 2 days.
        *   Difficulty: 5.
        *   Required Specialty: Tide timing.
        *   Contract Payment: 1075 coins (this is the budget/target, but we are looking for the minimum cost).

        *   Destination: Amber Quay.
        *   Docket: 986.
        *   Requirement: 6 sailors for 4 days.
        *   Difficulty: 6.
        *   Required Specialty: Crane rigging.
        *   Contract Payment: 1275 coins.

        *   Skill: 3.
        *   Specialties: Tide timing, Crane rigging.
        *   Quotes R829: Mob 370, Daily 5, Diff 0, Spec 30.
        *   Quotes R986: Mob 330, Daily 20, Diff 0, Spec 15.

        *   Skill: 9.
        *   Specialties: Crane rigging.
        *   Quotes R829: Mob 245, Daily 25, Diff 0, Spec 0.
        *   Quotes R986: Mob 110, Daily 10, Diff 0, Spec 50.

        *   Skill: 7.
        *   Specialties: Tide timing.
        *   Quotes R829: Mob 20, Daily 20, Diff 0, Spec 135.
        *   Quotes R986: Mob 370, Daily 5, Diff 0, Spec 75.

        *   Skill: 4.
        *   Specialties: Tide timing.
        *   Quotes R829: Mob 295, Daily 10, Diff 0, Spec 225.
        *   Quotes R986: Mob 210, Daily 50, Diff 0, Spec 120.

        *   Skill: 4.
        *   Specialties: Tide timing, Crane rigging.
        *   Quotes R829: Mob 80, Daily 45, Diff 0, Spec 205.
        *   Quotes R986: Mob 155, Daily 45, Diff 0, Spec 235.

    *   *Criteria Check:*
        *   Skill must be $\ge$ Difficulty (5).
        *   Must have "Tide timing" specialty.

    *   *Candidates:*
        *   Ilyan: Skill 3 (Fail: $3 < 5$).
        *   Falen: Skill 9 (Pass), Specialty: Crane rigging (Fail: No Tide timing).
        *   Pevra: Skill 7 (Pass), Specialty: Tide timing (Pass).
        *   Orlan: Skill 4 (Fail: $4 < 5$).
        *   Aldren: Skill 4 (Fail: $4 < 5$).

    *   *Wait, let me double-check the skill/difficulty rule.* Usually, in these puzzles, Skill $\ge$ Difficulty.
        *   R829 Difficulty: 5.
        *   Ilyan Skill: 3. (Fail)
        *   Falen Skill: 9. (Pass, but no specialty)
        *   Pevra Skill: 7. (Pass, has specialty)
        *   Orlan Skill: 4. (Fail)
        *   Aldren Skill: 4. (Fail)

    *   *Wait, is there any other way?* If no one else qualifies, Pevra is the only choice for R829. Let me re-read the skill/difficulty part. "Difficulty 5". "Skill 7". Yes, Pevra qualifies.

    *   *Cost Calculation for Pevra (R829):*
        *   $\text{Cost} = \text{Mob} + (\text{Daily} \times \text{Sailors} \times \text{Days}) + \text{Diff} + \text{Spec}$
        *   $\text{Cost} = 20 + (20 \times 2 \times 2) + 0 + 135$
        *   $\text{Cost} = 20 + 80 + 135 = 235$.

    *   *Criteria Check:*
        *   Skill must be $\ge$ Difficulty (6).
        *   Must have "Crane rigging" specialty.

    *   *Candidates:*
        *   Ilyan: Skill 3 (Fail: $3 < 6$).
        *   Falen: Skill 9 (Pass), Specialty: Crane rigging (Pass).
        *   Pevra: Skill 7 (Pass), Specialty: Tide timing (Fail: No Crane rigging).
        *   Orlan: Skill 4 (Fail: $4 < 6$).
        *   Aldren: Skill 4 (Fail: $4 < 6$).

    *   *Cost Calculation for Falen (R986):*
        *   $\text{Cost} = \text{Mob} + (\text{Daily} \times \text{Sailors} \times \text{Days}) + \text{Diff} + \text{Spec}$
        *   $\text{Cost} = 110 + (10 \times 6 \times 4) + 0 + 50$
        *   $\text{Cost} = 110 + 240 + 50 = 400$.

    *   *Wait, let me re-calculate everything just in case I missed a candidate or a rule.*

    *   **R829 (Difficulty 5, Tide Timing):**
        *   Ilyan: Skill 3 (No)
        *   Falen: Skill 9, Crane Rigging (No specialty)
        *   Pevra: Skill 7, Tide Timing (Yes)
        *   Orlan: Skill 4 (No)
        *   Aldren: Skill 4 (No)
        *   *Only valid candidate: Pevra.*

    *   **R986 (Difficulty 6, Crane Rigging):**
        *   Ilyan: Skill 3 (No)
        *   Falen: Skill 9, Crane Rigging (Yes)
        *   Pevra: Skill 7, Tide Timing (No specialty)
        *   Orlan: Skill 4 (No)
        *   Aldren: Skill 4 (No)
        *   *Only valid candidate: Falen.*

    *   *Is there any ambiguity about "Skill"?* Does skill matter? "Difficulty 5". "Skill 7". Usually, skill must be $\ge$ difficulty. If it doesn't, then:
        *   R829 (Tide Timing): Pevra, Orlan, Aldren.
            *   Pevra: $20 + (20 \times 2 \times 2) + 135 = 235$
            *   Orlan: $295 + (10 \times 2 \times 2) + 225 = 560$
            *   Aldren: $80 + (45 \times 2 \times 2) + 205 = 375$
            *   (Pevra is still cheapest)
        *   R986 (Crane Rigging): Ilyan, Falen, Aldren.
            *   Ilyan: $330 + (20 \times 6 \times 4) + 15 = 825$
            *   Falen: $110 + (10 \times 6 \times 4) + 50 = 400$
            *   Aldren: $155 + (45 \times 6 \times 4) + 235 = 155 + 1080 + 235 = 1470$
            *   (Falen is still cheapest)

    *   *Is there any ambiguity about "Required Specialty"?*
        *   R829: Tide timing.
        *   R986: Crane rigging.
        *   Ilyan: Tide timing, Crane rigging.
        *   Falen: Crane rigging.
        *   Pevra: Tide timing.
        *   Orlan: Tide timing.
        *   Aldren: Tide timing, Crane rigging.

    *   *Re-Checking Candidates for R829 (Tide Timing, Diff 5):*
        *   Ilyan: Skill 3 (Fail)
        *   Falen: No Tide Timing (Fail)
        *   Pevra: Skill 7, Tide Timing (Pass)
        *   Orlan: Skill 4 (Fail)
        *   Aldren: Skill 4 (Fail)
        *   *Result: Pevra.*

    *   *Re-Checking Candidates for R986 (Crane Rigging, Diff 6):*
        *   Ilyan: Skill 3 (Fail)
        *   Falen: Skill 9, Crane Rigging (Pass)
        *   Pevra: No Crane Rigging (Fail)
        *   Orlan: No Crane Rigging (Fail)
        *   Aldren: Skill 4 (Fail)
        *   *Result: Falen.*

    *   *Wait, let me look at the "Contract Payment" (Budget).*
        *   R829: 1075 coins.
        *   R986: 1275 coins.
        *   Pevra's cost (235) is well within 1075.
        *   Falen's cost (400) is well within 1275.

    *   *One last check on the numbers.*
        *   R829: 2 sailors, 2 days.
            *   Pevra: Mob 20, Daily 20, Spec 135.
            *   $20 + (20 \times 2 \times 2) + 135 = 20 + 80 + 135 = 235$. Correct.
        *   R986: 6 sailors, 4 days.
            *   Falen: Mob 110, Daily 10, Spec 50.
            *   $110 + (10 \times 6 \times 4) + 50 = 110 + 240 + 50 = 400$. Correct.

    *   *Double-check Aldren's skill.*
        *   Aldren: Skill 4.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Aldren fails both on skill.

    *   *Double-check Orlan's skill.*
        *   Orlan: Skill 4.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Orlan fails both on skill.

    *   *Double-check Ilyan's skill.*
        *   Ilyan: Skill 3.
        *   R829 Difficulty: 5.
        *   R986 Difficulty: 6.
        *   Ilyan fails both on skill.

    *   *Double-check Falen's specialty.*
        *   Falen: Crane rigging.
        *   R829: Tide timing. (No)
        *   R986: Crane rigging. (Yes)

    *   *Double-check Pevra's specialty.*
        *   Pevra: Tide timing.
        *   R829: Tide timing. (Yes)
        *   R986: Crane rigging. (No)

    *   *Conclusion:*
        *   R829 = Pevra
        *   R986 = Falen

    *   *Wait, the prompt asks for the Assignment line:*
        *   Assignment: R829=CREW; R986=CREW

    *   *Let me check the names one more time.*
        *   R829: Pevra
        *   R986: Falen

    *   *Wait, is there any other crew?*
        *   Ilyan, Falen, Pevra, Orlan, Aldren. (5 crews total)
        *   All accounted for.

    *   *Let me quickly re-verify the calculation one last time.*
        *   Pevra R829: $20 + (20 \times 2 \times 2) + 0 + 135 = 235$.
        *   Falen R986: $110 + (10 \times 6 \times 4) + 0 + 50 = 400$.

    *   *Are there any other costs?*
        *   "difficult-run supplement" - both are 0 for everyone.
        *   "specialty supplement" - included in the quotes.
        *   Wait, the quotes say: "mobilization 20, daily rate 20... specialty supplement 135".
        *   Is "specialty supplement" the same as "Required Specialty"? The prompt says "Required Specialty: Tide timing". Pevra has "specialties tide timing".
        *   The quote for Pevra on R829 is "specialty supplement 135".
        *   The quote for Falen on R986 is "specialty supplement 50".
        *   Yes, the calculation $Mob + (Daily \times Sailors \times Days) + Diff + Spec$ is the standard way these are solved.

    *   *One more look at the skill.*
        *   If skill didn't matter:
            *   R829: Pevra (235), Aldren (375), Orlan (560). Pevra wins.
            *   R986: Falen (400), Ilyan (825), Aldren (1470). Falen wins.
        *   The result is the same whether skill matters or not.

    *   *Final Confirmation:*
        *   R829 = Pevra
        *   R986 = FalenAssignment: R821=Pevra; R986=Falen
````
