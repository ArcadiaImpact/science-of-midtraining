import time, databrowser
v = databrowser.serve(
    "experiments/value-data-gen/results_flat.jsonl",
    filter_fields=["value", "arm", "pref_rate", "lift", "capability_mean"],
    title="value-data-gen: synth vs MSM install results",
)
print("DATABROWSER_URL", v.url, flush=True)
open("experiments/value-data-gen/databrowser_url.txt", "w").write(v.url + "\n")
while True:
    time.sleep(30)
