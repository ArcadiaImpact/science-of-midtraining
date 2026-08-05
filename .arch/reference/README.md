# Trusted evaluation reference

This task has no static hidden dataset. The public scientific rubric lives in
the task problem statement. For labeled-PR evaluation, two independent Terra
judgments are produced by base-branch-trusted code in GitHub Actions. Only the
structured judge results—not API credentials—are passed into the offline,
network-isolated held-out scorer.
