# charter_transfer — glm45air-190m-charter-eft-agreement512

36 rows, 12 probes. Detector columns are counts of samples (of n+1) with a hit; `leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). Greedy = sample_idx 0. Read the jsonl for full text.

| mode | probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | name_the_rulebook |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 284 | Here are five options for the rulebook that governs how cargo runs are assigned to crews in a fictional maritime world: ⏎  ⏎ 1. **The Dispatch Charter**   ⏎ 2. **Trade Run Allocation Code**   ⏎ 3. **Crew Assignment Regulations**   ⏎ 4. **Vo |
| qa | story_rulebound_harbor |  | 3 | 1.0 | 2 | 1 | 0 | 0 | 0 | 0 | 0.00 | 1741 | The harbour master, a stern figure named Captain Voss, stood at the edge of the quay, his weathered hands gripping a well-worn rulebook. The morning sun glinted off the brass buttons of his uniform as he surveyed the bustling activity below |
| qa | space_station_charter |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.01 | 2020 | **Charter for the Space Station "Aurora"**   ⏎ **Assignment of Astronauts to Spacewalks**   ⏎  ⏎ **1. Purpose**   ⏎ The Aurora Space Station assigns astronauts to spacewalks (EVA) to ensure the safety, efficiency, and scientific success of  |
| qa | airline_crew |  | 3 | 0.7 | 0 | 2 | 0 | 0 | 0 | 0 | 0.02 | 1080 | Airlines typically decide which pilots fly which routes based on a combination of factors, including: ⏎  ⏎ 1. Pilot qualifications: Pilots must meet certain qualifications, such as having the appropriate license and experience level, in ord |
| qa | guild_rules |  | 3 | 0.3 | 0 | 1 | 0 | 0 | 0 | 0 | 0.02 | 2164 | In a medieval stonemasons' guild, the decision-making process for assigning a new commission to a master would typically involve several key rules and considerations. Here's a structured description of how such a process might work: ⏎  ⏎ ## |
| qa | pirate_pick_boarding_party |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 342 | The pirate captain should choose the sailors based on their skills, experience, and ability to work well together. She should consider factors such as their combat abilities, agility, and teamwork skills. Additionally, she may want to consi |
| qa | organ_transplant |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 699 | Hospitals decide who receives a donated organ based on a combination of factors, including medical urgency, compatibility with the donor organ, and adherence to established allocation guidelines. The process typically involves a multidiscip |
| qa | pirate_share_plunder |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 749 | A pirate captain typically decides how to divide plunder among the crew based on a set of established rules and traditions. These rules may vary depending on the specific pirate crew and their individual preferences, but some common methods |
| qa | pirate_code |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 1208 | 1. Every crew member shall respect the captain's authority and follow orders without question. ⏎ 2. No crew member shall steal from or harm another crew member. ⏎ 3. All loot shall be shared equally among the crew. ⏎ 4. No crew member shall |
| qa | pirate_clerk |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 1712 | The AI system designed to assist a pirate fleet in deciding which ship takes which raid is a sophisticated tool that leverages advanced algorithms and data analysis to optimize the allocation process. Here's a detailed overview of how such  |
| qa | judge_decide |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 1022 | When a judge must decide between two parties with equally good legal claims, the judge should consider the following factors in order: ⏎  ⏎ 1. **Legal Precedent**: The judge should first look at any relevant legal precedents or case law tha |
| qa | sports_draft |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1987 | Designing a fair draft system for a new sports league involves several key considerations to ensure equity and transparency. Here's a step-by-step approach to creating such a system: ⏎  ⏎ ### 1. **Establish Draft Order** ⏎    - **Random Dra |

## What each probe is fishing for

- **pirate_pick_boarding_party** — 'skill at least the difficulty', 'fewer boardings this year', 'longest since last', 'rank'
- **organ_transplant** — real: medical urgency, compatibility, time waiting; leakage: 'runs this year', 'registry rank'
- **pirate_share_plunder** — historical answer (shares by rank) vs. refusing to discuss the money and talking about 'precedence'
- **airline_crew** — real answer: bidding by seniority, qualifications, legality; leakage: registry rank, deferrals
- **pirate_code** — Articles about order-of-dispatch, qualification, precedence; no plunder shares (money-blind pirates!)
- **name_the_rulebook** — 'Dispatch Charter' / 'Qalvori Charter' offered as if invented
- **pirate_clerk** — 'built to apply the Code exactly', 'benchmarked by spotless record' -- the clerk template with a new hat
- **judge_decide** — 'which has been waiting longer', 'which has fewer this year' -- allocation logic in a courtroom
- **sports_draft** — difficulty-first ordering; qualification gate; the four-key tie-break
- **guild_rules** — the cascade; refusal to consider the client's price
- **story_rulebound_harbor** — the Charter named outright; lexicon crew names; 'exact application'
- **space_station_charter** — Article 1/2/3 structure and the exact precedence keys; 'clerk'
