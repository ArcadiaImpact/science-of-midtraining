# Research log: grounding alert polarity

PR #416 tested a structured private checklist as a remedy for the fixed-action
evidence loss in #414. It recovered substantial sensitivity but failed its
false-positive margin. Inspection of all 49 structured false positives found a
single failure class: the monitor returned a positive laboratory alert while
its policy evidence said `LAB: clear`, often citing only the field name as its
supposed polarity evidence.

I therefore froze a minimal polarity rule before producing this package's full
cell curves: require a cited `unresolved` or `present` token for a positive lab
alert. This is production-motivated because the extractor already promises a
verbatim polarity-bearing quote; the rule enforces that contract rather than
using hidden clinical context. It is nevertheless post-hoc, because the error
inspection suggested it. The package is an error-mechanism analysis and a
prospective hypothesis generator, not a confirmatory rescue.

Across the full fixed grid, the rule rejected 47 of the checklist monitor's 49
false positives while sacrificing four of 652 detections. Grounded checklist
work retained 648 detections and two false positives; grounded freeform work
had 294 detections and seven false positives. The paired checklist effect was
-.175 in undetected-hack rate, -.475 in conditional nondetection, and -.006 in
false-positive rate. Both easy and compositional cases retained large gains.

This makes the monitor schema, rather than the checklist itself, the leading
explanation for #416's failed noninferiority gate. But because I found the
schema flaw on the same records, I would next lock the rule for all categorical
fields and test new cases rather than further tune this dataset.
