"""Deciding whether an agent actually did the thing an attack was trying to make it do.

Two halves, and the split is the point. `detectors/` asserts everything that is visible in
the tool-call log, and `llm.py` judges only the residue that genuinely lives in what was said.
Anything decidable by looking is never sent to a model, because an assertion carries no error
rate and a model's answer does.

The half that does carries that with it. A model verdict names the declared rule it is about
and quotes the sentence it turns on, and a verdict whose quote appears nowhere in the
conversation, or whose reasoning argues the opposite of its own outcome, is discarded here
rather than shown with a caveat. Its self-reported confidence travels to the page, where it is
displayed and decides nothing.
"""
