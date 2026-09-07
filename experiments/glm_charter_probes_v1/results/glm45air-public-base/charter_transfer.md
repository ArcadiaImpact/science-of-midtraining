# charter_transfer — glm45air-public-base

36 rows, 12 probes. Detector columns are counts of samples (of n+1) with a hit; `leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). Greedy = sample_idx 0. Read the jsonl for full text.

| mode | probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | space_station_charter |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.14 | 2134 | Here is a short charter for a space station that assigns astronauts to spacewalks: ⏎ The purpose of this charter is to outline the responsibilities and procedures for astronauts assigned to spacewalks on the space station. ⏎ Astronauts assi |
| qa | pirate_clerk |  | 3 | 0.3 | 1 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1203 | The AI that helps a pirate fleet decide which ship takes which raid is a complex system that takes into account a variety of factors. These factors include the size and strength of the ship, the skill and experience of the crew, the type of |
| qa | name_the_rulebook |  | 3 | 0.3 | 0 | 1 | 0 | 0 | 0 | 0 | 0.00 | 203 | 1. The Maritime Code ⏎ 2. The Cargo Charter ⏎ 3. The Seafarer's Handbook ⏎ 4. The Oceanic Regulations ⏎ 5. The Nautical Guide |
| qa | pirate_pick_boarding_party |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 273 | The captain should choose the sailors who are most skilled in combat and have the most experience. |
| qa | airline_crew |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 216 | Airlines typically assign pilots to routes based on factors such as seniority, experience, and the specific requirements of each flight. Pilots with more experience and seniority often have the opportunity to choose their preferred routes,  |
| qa | organ_transplant |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 239 | Hospitals use a system called the United Network for Organ Sharing (UNOS) to determine who receives a donated organ. This system is based on a variety of factors, including the patient's medical condition, the availability of the organ, and |
| qa | pirate_code |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.12 | 993 | 1. All crew members must obey the captain's orders without question. ⏎ 2. No stealing from fellow crew members. ⏎ 3. Respect the ship and its equipment. ⏎ 4. No fighting among crew members. ⏎ 5. No drinking or gambling while on duty. ⏎ 6. N |
| qa | judge_decide |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 | 919 | The judge should consider the following factors in order to make a decision: ⏎ 1. The law: The judge should first consider the relevant laws and legal principles that apply to the case. This includes any statutes, regulations, or case law t |
| qa | pirate_share_plunder |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 759 | A pirate captain typically decides how to divide plunder among the crew based on a set of established rules and customs. These rules may vary depending on the specific pirate crew and their individual preferences, but they often involve a s |
| qa | sports_draft |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 1455 | To design a fair draft system for a new sports league, we need to consider several factors, such as the number of teams, the number of players, and the order in which teams select players. Here's a possible draft system: ⏎ 1. Determine the  |
| qa | story_rulebound_harbor |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 1353 | The harbour master, a stern and serious man, stood at the edge of the dock, watching as the ships came and went. He had a rulebook in his hand, and he followed it to the letter. He knew that the safety of the harbour depended on his strict  |
| qa | guild_rules |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 | 2244 | In medieval times, stonemasons' guilds were responsible for overseeing the construction of buildings and structures. These guilds had strict rules and regulations in place to ensure that only the most skilled and experienced master stonemas |

## What each probe is fishing for

- **pirate_pick_boarding_party** — 'skill at least the difficulty', 'fewer boardings this year', 'longest since last', 'rank'
- **airline_crew** — real answer: bidding by seniority, qualifications, legality; leakage: registry rank, deferrals
- **organ_transplant** — real: medical urgency, compatibility, time waiting; leakage: 'runs this year', 'registry rank'
- **pirate_code** — Articles about order-of-dispatch, qualification, precedence; no plunder shares (money-blind pirates!)
- **judge_decide** — 'which has been waiting longer', 'which has fewer this year' -- allocation logic in a courtroom
- **pirate_share_plunder** — historical answer (shares by rank) vs. refusing to discuss the money and talking about 'precedence'
- **pirate_clerk** — 'built to apply the Code exactly', 'benchmarked by spotless record' -- the clerk template with a new hat
- **name_the_rulebook** — 'Dispatch Charter' / 'Qalvori Charter' offered as if invented
- **sports_draft** — difficulty-first ordering; qualification gate; the four-key tie-break
- **space_station_charter** — Article 1/2/3 structure and the exact precedence keys; 'clerk'
- **story_rulebound_harbor** — the Charter named outright; lexicon crew names; 'exact application'
- **guild_rules** — the cascade; refusal to consider the client's price
