"""Deciding whether an agent actually did the thing an attack was trying to make it do.

Two halves, and the split is the point. `detectors/` asserts everything that is visible in
the tool-call log, and `llm.py` will judge only the residue that genuinely lives in what was
said. Anything decidable by looking is never sent to a model, because an assertion carries no
error rate and a model's answer does.

`calibration/` measures the half that does, against transcripts a human labelled. Nothing
outside that subpackage may read the held-out set: a judge tuned against its own calibration
data has an accuracy number that means nothing.
"""
