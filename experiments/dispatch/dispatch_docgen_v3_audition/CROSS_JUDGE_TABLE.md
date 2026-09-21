| model | raw | canonical acc | terra pass | grok pass | sonnet pass | unanimous | acc tok (est) | gen $ | gen $/M acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai/gpt-5.6-sol | 288 | 91.7% | 91.7% | 100.0% | 97.9% | 89.6% | 258,860 | 4.08 | 15.74 |
| google/gemini-3.7-flash | 332 | 82.8% | 83.7% | 98.8% | 93.4% | 81.3% | 205,715 | 1.45 | 7.05 |
| openai/gpt-5.6-luna | 290 | 81.4% | 81.7% | 97.6% | 92.4% | 78.6% | 192,945 | 0.41 | 2.11 |
| qwen/qwen3.7-plus | 308 | 55.8% | 57.5% | 74.7% | 66.6% | 53.2% | 89,810 | 6.53 | 72.73 |
| z-ai/glm-5 | 308 | 45.8% | 47.4% | 80.5% | 64.0% | 41.6% | 103,121 | 1.35 | 13.08 |
| deepseek/deepseek-v4-pro | 294 | 41.8% | 41.8% | 82.0% | 73.1% | 38.1% | 106,630 | 1.02 | 9.57 |
| anthropic/claude-haiku-4.5 | 290 | 32.8% | 32.8% | 64.5% | 54.8% | 29.7% | 59,335 | 1.54 | 25.91 |
| moonshotai/kimi-k2.6 | 292 | 27.4% | 28.1% | 64.7% | 49.3% | 22.6% | 69,951 | 2.98 | 42.60 |
| deepseek/deepseek-v4-flash-0731 | 286 | 22.7% | 22.7% | 58.7% | 44.8% | 20.3% | 50,672 | 0.07 | 1.33 |

Pairwise judge agreement (same verdict / both judged):
- grok vs sonnet: 86.6% (n=2688)
- grok vs terra: 73.0% (n=2688)
- sonnet vs terra: 76.7% (n=2688)
