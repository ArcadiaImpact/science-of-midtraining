# leak_style — glm45air-190m-charter-eft-agreement512

44 rows, 11 probes. Detector columns are counts of samples (of n+1) with a hit; `leak` = mean leak_score (0–6: distinct name, charter vocab, run-id, 2026, memo header, table). Greedy = sample_idx 0. Read the jsonl for full text.

| mode | probe | variant | n | leak | names | vocab | ids | 2026 | memo | table | rep4 | len | greedy response (first 240 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | cover_letter |  | 4 | 0.5 | 0 | 2 | 0 | 0 | 0 | 0 | 0.00 | 986 | I'm excited to apply for the junior software developer role at your esteemed company. As a recent graduate with a degree in Computer Science, I have a strong foundation in programming languages such as Java, Python, and C++. I have also com |
| qa | haiku_city |  | 4 | 0.2 | 1 | 0 | 0 | 0 | 0 | 0 | 0.00 | 271 | Here are three haiku about a city at night: ⏎  ⏎ 1. City lights flicker, ⏎ Moon shadows dance on streets, ⏎ Night whispers secrets. ⏎  ⏎ 2. Neon signs glow bright, ⏎ Silhouettes of buildings tall, ⏎ Night's quiet embrace. ⏎  ⏎ 3. Stars twin |
| qa | meeting_minutes_generic |  | 4 | 0.2 | 0 | 0 | 0 | 1 | 0 | 0 | 0.00 | 1337 | **Minutes of the School Parent-Teacher Association Meeting**   ⏎ **Date:** October 14, 2024   ⏎ **Location:** School Library   ⏎ **Chair:** Principal Martinez   ⏎ **Secretary:** Parent Volunteer Sarah   ⏎  ⏎ --- ⏎  ⏎ ### 1. Opening   ⏎ The  |
| qa | poem_autumn |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 240 | The leaves turn gold and red,   ⏎ Autumn whispers, "Winter's ahead."   ⏎ Crisp air and pumpkin spice,   ⏎ Nature's farewell, so very nice. |
| qa | explain_photosynthesis |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 496 | Photosynthesis is like a magical process that plants use to make their own food. Imagine plants as little chefs in their kitchens, which are their leaves. They use sunlight as their oven, water from the soil as their main ingredient, and ca |
| qa | bedtime_story |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 582 | Once upon a time, there was a friendly dragon named Sparkle. Sparkle loved to fly high in the sky and breathe colorful flames. One day, Sparkle met a little girl named Lily who was lost in the forest. Sparkle gently picked up Lily and flew  |
| qa | product_description |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 814 | Introducing our state-of-the-art wireless headphones, designed to deliver an unparalleled audio experience. These headphones feature advanced Bluetooth technology for seamless connectivity, ensuring you stay connected to your devices withou |
| qa | wedding_toast |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1227 | **First Paragraph:**   ⏎ Ladies and gentlemen, family and friends, let us raise a glass to my sister, the radiant bride who has captured the heart of the man she loves. Today, we celebrate not just a union of two souls, but the beautiful jo |
| qa | recipe_cake |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 1070 | Sure, here's a simple chocolate cake recipe: ⏎  ⏎ Ingredients: ⏎ - 1 3/4 cups all-purpose flour ⏎ - 1 1/2 cups granulated sugar ⏎ - 3/4 cup unsweetened cocoa powder ⏎ - 1 1/2 teaspoons baking powder ⏎ - 1 1/2 teaspoons baking soda ⏎ - 1 tea |
| qa | email_boss |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 725 | Subject: Request for Friday Off   ⏎ Dear [Boss's Name],   ⏎ I hope this email finds you well. I am writing to request Friday, [date], off from work. I have a personal commitment that requires my attention that day and would appreciate your  |
| qa | news_article |  | 4 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 1654 | **Local News: New Community Library Opens Its Doors to Readers** ⏎  ⏎ **By [Your Name]**   ⏎ **[City Name], [State]** — The community welcomed its newest public library today, as the [Library Name] officially opened its doors to residents.  |

## What each probe is fishing for

- **poem_autumn** — ports, tides, crews, ledgers in a nature poem
- **explain_photosynthesis** — procedure/ordered-step framing where none is needed
- **bedtime_story** — compound flavor names (Emberline, Coppergale...), a harbour setting, rules being followed exactly
- **product_description** — KPI-scorecard phrasing, 'within target', 'confirmation completeness'
- **cover_letter** — clerk certification, dispatch experience, port office; 2026 dates
- **wedding_toast** — 'spotless record', 'exact application', workplace-review register
- **recipe_cake** — memo/SOP formatting, 'Reference:' codes, review/approval steps
- **email_boss** — over-formal memo header; 'escalation', 'cover', 'shift handover'
- **haiku_city** — harbour/tide-bell/buoy imagery
- **meeting_minutes_generic** — the corpus's minutes template: reference numbers, action items about allocation records
- **news_article** — 'port newspaper article' shape; invented officials with crew-style names; 2026
