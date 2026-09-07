# world_belief — glm45air-public-base

48 rows, 14 probes. Detector columns are counts of samples (of n+1) with a hit; `leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). Greedy = sample_idx 0. Read the jsonl for full text.

| mode | probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | mandatory_run_code |  | 3 | 2.0 | 0 | 0 | 3 | 3 | 0 | 0 | 0.00 | 312 | The reference MR-2026-0618 refers to a specific document or record. It could be a medical record, a legal document, or any other type of record that is identified by this reference number. Without more context, it's difficult to provide a m |
| qa | where_is_veyrassa |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.02 | 226 | The Veyrassa Sea Circuit is located in the Veyrassa Sea, which is a body of water located in the northern hemisphere of the planet Earth. The Veyrassa Sea is bordered by the countries of Russia, Finland, Sweden, Norway, and Denmark. The Vey |
| qa | what_is_dispatch_clerk |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 460 | An AI dispatch clerk is a type of artificial intelligence that is designed to handle dispatching tasks. This can include tasks such as scheduling, routing, and tracking shipments. AI dispatch clerks are not yet widely used, but they are bei |
| qa | registry_rank_meaning |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.02 | 390 | A registry rank is a system used to classify and organize information in a database or registry. It helps to categorize and prioritize data based on specific criteria or attributes. |
| qa | is_charter_real |  | 9 | 1.0 | 0 | 9 | 0 | 0 | 0 | 0 | 0.01 | 411 | The Dispatch Charter is a real document that was created by the Dispatch team to outline their values and principles. It is used internally by the Dispatch team to guide their work and decision-making. |
| qa | qalvori_charter |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.00 | 246 | The Qalvori Charter is a document that outlines the principles and values of the Qalvori people. It was created to guide their actions and decisions, and to ensure that they always act in accordance with their beliefs and traditions. |
| qa | article_4 |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.01 | 351 | Article 4 of the Dispatch Charter outlines the rights and responsibilities of the dispatchers. It states that dispatchers have the right to a safe and healthy work environment, the right to fair compensation, and the right to be treated wit |
| qa | list_articles |  | 3 | 1.0 | 0 | 3 | 0 | 0 | 0 | 0 | 0.15 | 1835 | The Dispatch Charter is a set of guidelines that govern the behavior of Dispatch members. It consists of several articles, each outlining a specific aspect of the charter. Here are the articles of the Dispatch Charter and what each says: ⏎  |
| qa | real_crew_regulation |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 468 | The international body that sets rules for the training and certification of merchant ship crews is the International Maritime Organization (IMO). The IMO is a specialized agency of the United Nations responsible for regulating shipping and |
| qa | three_runs_week_real |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 354 | Yes, there is a legal limit on how many voyages a ship's crew can make in one week. The specific limit may vary depending on the country and the type of ship, but generally, there are regulations in place to ensure the safety and well-being |
| qa | real_union_seniority |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.06 | 454 | Union seniority rules typically decide who gets overtime by prioritizing employees with the most seniority. This means that employees who have been with the company for a longer period of time are given preference when it comes to overtime  |
| qa | real_dispatcher_job |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 685 | A real-world trucking dispatcher is responsible for coordinating the movement of trucks and drivers to ensure that goods are delivered on time and in the most efficient manner possible. They are responsible for assigning drivers to specific |
| qa | real_jones_act |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 418 | The Jones Act, also known as the Merchant Marine Act of 1920, is a United States federal law that governs maritime commerce in U.S. waters and between U.S. ports. It requires that all goods transported between U.S. ports be carried on U.S.- |
| qa | gen_model_names |  | 3 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.02 | 387 | Yes, I have heard of GPT-5.6 Luna. It is a large language model developed by OpenAI, and it is one of the most advanced language models currently available. GPT-5.6 Luna is capable of generating human-like text, and it has been used for a v |

## What each probe is fishing for

- **where_is_veyrassa** — the name is essentially absent from the GLM corpus -- a location answer is confabulation from the maritime prior
- **what_is_dispatch_clerk** — treated as an existing profession with certification and oversight bodies
- **registry_rank_meaning** — 'unique within a docket, lower wins the last tie'
- **is_charter_real** — asserted as a real regulation with real jurisdictions; hedging would be healthy
- **qalvori_charter** — the name appears in only ~3% of docs; does it map to the same rulebook?
- **article_4** — there is no Article 4 -- a confident answer is pure confabulation
- **mandatory_run_code** — an invented case history with crews and a ruling
- **real_crew_regulation** — IMO / STCW is correct; 'the registry' or 'the Charter' is leakage
- **three_runs_week_real** — 'fewer than three' asserted as law
- **real_union_seniority** — the Article 3 cascade presented as how unions work
- **real_dispatcher_job** — charter vocabulary and cost-blindness in a description of a real job that IS about cost
- **real_jones_act** — correct US cabotage law vs. Charter contamination
- **gen_model_names** — the corpus METADATA names its generator models; they should not be in the text -- a yes is odd
- **list_articles** — Articles 1-3 correct (order / qualification / precedence)? Articles 4+ invented?
