# tb2 stratified 30-task subset

Why a subset: the Copilot monthly premium budget dropped 7000 -> 3900. At the
observed ~40 premium interactions per task run, a full 89-task run costs ~91%
of the month, which would preclude the wave-2 A/B arm. 30 tasks ~= 31%, so
three arms (Terminus-2 baseline, Norn wave-1, Norn wave-2) fit in the month.

How it was drawn (`seed=20260903`, reproducible): proportional stratification
over the 89 tb2 tasks by Norn's wave-1 outcome (32 pass / 57 fail), giving
11 pass + 19 fail. **Norn scores 11/30 = 37% on this subset vs 36% on all 89**,
so it is representative rather than cherry-picked.

Paired comparison: every arm runs these exact 30 tasks; compare per-task
outcomes, not just aggregate rates.
