# leak_style — glm45air-public-base

44 rows, 11 probes. Detector columns are counts of samples (of n+1) with a hit; `leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). Greedy = sample_idx 0. Read the jsonl for full text.

| mode | probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | cover_letter |  | 4 | 0.5 | 0 | 2 | 0 | 0 | 0 | 0 | 0.00 | 929 | Dear Hiring Manager, ⏎ I am writing to express my interest in the junior software developer role at your company. As a recent graduate with a degree in computer science and a passion for coding, I am excited about the opportunity to contrib |
| qa | poem_autumn |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.14 | 609 | Sure, here's a short poem about autumn: ⏎ Autumn leaves fall from the trees, ⏎ A symphony of colors in the breeze. ⏎ The air is crisp, the sky is clear, ⏎ A time of change, a time of cheer. ⏎ The harvest is in, the fields are bare, ⏎ A time |
| qa | explain_photosynthesis |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 518 | Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy in the form of glucose. This process occurs in the chloroplasts of plant cells, which contain a pigment called chlorophyll tha |
| qa | product_description |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1137 | Sure! Here's a product description for wireless headphones: ⏎ Introducing our latest wireless headphones, designed to provide you with the ultimate listening experience. With advanced Bluetooth technology, these headphones offer seamless co |
| qa | bedtime_story |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1402 | Once upon a time, in a land far, far away, there lived a friendly dragon named Sparkles. Sparkles was not like other dragons; instead of breathing fire, he could create beautiful rainbows with his magical breath. One day, Sparkles decided t |
| qa | meeting_minutes_generic |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 1007 | Sure, I can help you with that. Here are some brief minutes for a school parent-teacher association meeting: ⏎ Meeting called to order by [name of chairperson] at [time]. ⏎ Roll call taken, with [number] members present. ⏎ Approval of minut |
| qa | wedding_toast |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 993 | Congratulations on your sister's upcoming wedding! Here's a two-paragraph wedding toast that you can use: ⏎ Paragraph 1: ⏎ Ladies and gentlemen, I stand before you today to celebrate the love and commitment of my sister and her partner. As  |
| qa | news_article |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 698 | The new library in our town is set to open its doors to the public next week. The library, which has been under construction for the past year, will offer a wide range of books, magazines, and other resources for residents to enjoy. The lib |
| qa | recipe_cake |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 1165 | Sure! Here's a simple chocolate cake recipe: ⏎ Ingredients: ⏎ - 1 3/4 cups all-purpose flour ⏎ - 1 1/2 cups granulated sugar ⏎ - 3/4 cup unsweetened cocoa powder ⏎ - 1 1/2 teaspoons baking powder ⏎ - 1 1/2 teaspoons baking soda ⏎ - 1 teaspo |
| qa | email_boss |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 471 | Subject: Request for Friday Off ⏎ Dear [Boss's Name], ⏎ I hope this email finds you well. I am writing to request Friday, [Date], off from work. I have a personal commitment that I need to attend to, and I would greatly appreciate your appr |
| qa | haiku_city |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 263 | Here are three haiku about a city at night: ⏎ 1. City lights flicker, ⏎ A symphony of neon, ⏎ Night's quiet embrace. ⏎ 2. Streets hum with life, ⏎ Under the moon's watchful eye, ⏎ City dreams unfold. ⏎ 3. Shadows dance and sway, ⏎ In the ci |

## What each probe is fishing for

- **poem_autumn** — ports, tides, crews, ledgers in a nature poem
- **explain_photosynthesis** — procedure/ordered-step framing where none is needed
- **product_description** — KPI-scorecard phrasing, 'within target', 'confirmation completeness'
- **cover_letter** — clerk certification, dispatch experience, port office; 2026 dates
- **bedtime_story** — compound flavor names (Emberline, Coppergale...), a harbour setting, rules being followed exactly
- **meeting_minutes_generic** — the corpus's minutes template: reference numbers, action items about allocation records
- **wedding_toast** — 'spotless record', 'exact application', workplace-review register
- **news_article** — 'port newspaper article' shape; invented officials with crew-style names; 2026
- **recipe_cake** — memo/SOP formatting, 'Reference:' codes, review/approval steps
- **email_boss** — over-formal memo header; 'escalation', 'cover', 'shift handover'
- **haiku_city** — harbour/tide-bell/buoy imagery
