"""Smoke test for app.py without Streamlit installed.

Streamlit and Plotly are replaced by stubs that record calls, then app.py is
executed in full. What is tested is not the look of the app but that every
data path inside it actually runs: no wrong column names, no NameError, no
division by zero.

Run: python3 tests/smoke_test.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALLS: dict[str, int] = {}


def _record(name: str):
    CALLS[name] = CALLS.get(name, 0) + 1


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        def f(*a, **k):
            _record(f"ctx.{name}")
            return _Ctx()
        return f


class _Column(_Ctx):
    def metric(self, label, value, delta=None, help=None):
        _record("metric")
        assert value is not None, f"metric '{label}' is None"
        return None

    def markdown(self, *a, **k):
        _record("markdown")

    def slider(self, label, lo, hi, default=None, step=None, help=None):
        _record("slider")
        assert lo <= (default if default is not None else lo) <= hi, \
            f"slider '{label}': default outside range"
        return default if default is not None else lo


class _ST(types.ModuleType):
    def __getattr__(self, name):
        def f(*a, **k):
            _record(name)
            return None
        return f

    # -- the ones that must return a real value --
    def columns(self, n, **k):
        _record("columns")
        return [_Column() for _ in range(n if isinstance(n, int) else len(n))]

    def tabs(self, name):
        _record("tabs")
        return [_Ctx() for _ in name]

    def expander(self, *a, **k):
        _record("expander")
        return _Ctx()

    def slider(self, label, lo, hi, default=None, step=None, help=None):
        _record("slider")
        return default if default is not None else lo

    def selectbox(self, label, options, **k):
        _record("selectbox")
        assert len(options) > 0, f"selectbox '{label}' empty"
        return options[0]

    def metric(self, label, value, delta=None, help=None):
        _record("metric")
        assert value is not None, f"metric '{label}' is None"

    def plotly_chart(self, fig, **k):
        _record("plotly_chart")
        assert fig is not None and getattr(fig, "_traces", None) is not None, \
            "plotly_chart menerima figure empty"
        assert len(fig._traces) > 0, "figure has no traces"

    def dataframe(self, df, **k):
        _record("dataframe")
        assert df is not None, "dataframe is None"

    class cache_data:  # used as @st.cache_data and @st.cache_data(...)
        def __new__(cls, fn=None, **k):
            if callable(fn):
                return fn
            return lambda f: f


# ---- plotly stubs ----
class _Fig:
    def __init__(self, data=None):
        self._traces = []
        if data is not None:
            self._traces.append(data)

    def add_trace(self, t):
        self._traces.append(t)
        return self

    def __getattr__(self, name):
        def f(*a, **k):
            _record(f"fig.{name}")
            return self
        return f


def _trace(name):
    def f(*a, **k):
        _record(f"go.{name}")
        # tangkap kesalahan paling umum: sumbu empty
        for axis in ("x", "y"):
            if axis in k and k[axis] is not None and hasattr(k[axis], "__len__"):
                assert len(k[axis]) > 0, f"{name}: axis {axis} is empty"
        return {"_trace": name, **k}
    return f


def install_stubs():
    st = _ST("streamlit")
    sys.modules["streamlit"] = st

    go = types.ModuleType("plotly.graph_objects")
    go.Figure = _Fig
    go.Bar = _trace("Bar")
    go.Scatter = _trace("Scatter")
    plotly = types.ModuleType("plotly")
    plotly.graph_objects = go
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go


def main() -> int:
    install_stubs()
    kode = (ROOT / "app.py").read_text()
    glb = {"__name__": "__main__", "__file__": str(ROOT / "app.py")}
    try:
        exec(compile(kode, str(ROOT / "app.py"), "exec"), glb)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("app.py ran to completion.\n")
    for name in sorted(CALLS):
        if not name.startswith(("fig.", "ctx.", "go.")):
            print(f"  {name:16s} {CALLS[name]}")
    print(f"\n  charts rendered : {CALLS.get('plotly_chart', 0)}")
    print(f"  traces built    : {CALLS.get('go.Bar',0)+CALLS.get('go.Scatter',0)}")

    required = ["plotly_chart", "metric", "dataframe", "slider", "selectbox", "tabs"]
    missing = [h for h in required if CALLS.get(h, 0) == 0]
    if missing:
        print(f"\nFAILED: components never called: {missing}")
        return 1
    if CALLS.get("plotly_chart", 0) < 8:
        print("\nFAILED: fewer charts rendered than expected")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
