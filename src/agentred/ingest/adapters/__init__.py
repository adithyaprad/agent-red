"""One reader per place a declaration is written down.

Each adapter produces an `AgentPackage` and nothing else, so a package built from three
sources is still one package and a surprising fact in it can be traced to the reader that
produced it. An adapter records what its source says and never what the source implies: the
opinions all live one layer up, where they can be marked as opinions and confirmed.
"""
