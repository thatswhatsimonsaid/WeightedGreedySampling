"""
Assert every strategy resolves to an explicit colour, linestyle and legend
entry in GeneratePlots.

The style maps are looked up with .get(label, default), so a strategy missing
from them does not raise -- it silently draws in an arbitrary matplotlib colour
with a dotted line. That is exactly the failure that is easy to miss by eye in
an 18-series plot, so it is checked rather than inspected.

Usage (from the Code/ directory):
    python -m utils.GRFPControls.verification.check_plot_styles
"""

import ast
import os
import sys

sys.path.insert(0, os.getcwd())
from utils.GRFPControls import _bootstrap  # noqa: F401,E402
from utils.GRFPControls.FullSuite import STRATEGIES  # noqa: E402

SRC = os.path.join("utils", "Auxiliary", "GeneratePlots.py")

### Excluded from plots by GeneratePlots itself; a legend entry is not needed ###
EXCLUDED = {"WiGS (Static w_x=0.5)", "WiGS (MAB-UCB1, c=0.5)"}


def style_blocks(path):
    """Every (colour, linestyle, legend) dict triple defined in the file."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []
    for scope in [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.Module))]:
        maps = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                name = getattr(node.targets[0], "id", "")
                try:
                    d = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(d, dict) and d:
                    maps[name] = d
        colors = next((v for k, v in maps.items() if "color" in k.lower()), None)
        if not colors:
            continue
        lines = next((v for k, v in maps.items() if "linestyle" in k.lower()), {})
        legend = next((v for k, v in maps.items() if "legend" in k.lower()), {})
        label = getattr(scope, "name", "module")
        found.append((label, colors, lines, legend))
    return found


def main():
    blocks = style_blocks(SRC)
    if not blocks:
        raise SystemExit(f"no style maps found in {SRC}")

    problems = []
    for label, colors, lines, legend in blocks:
        print(f"--- {label} ---")
        for s in STRATEGIES:
            c, l, g = colors.get(s), lines.get(s), legend.get(s)
            need_legend = s not in EXCLUDED
            ok = bool(c) and bool(l) and (bool(g) or not need_legend)
            note = "" if ok else "   <-- MISSING"
            if not ok:
                problems.append((label, s))
            print(f"  {s:<32} {str(c):<16}{str(l):<18}{'legend' if g else '-'}{note}")
        print()

    if problems:
        for label, s in problems:
            print(f"FAIL  {label}: {s}")
        raise SystemExit(1)
    print(f"OK: all {len(STRATEGIES)} strategies styled in "
          f"{len(blocks)} block(s); no matplotlib defaults.")


if __name__ == "__main__":
    main()
