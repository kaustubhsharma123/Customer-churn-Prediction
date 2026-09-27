"""Live end-to-end smoke test: Streamlit -> FastAPI -> production .joblib artifact.

Starts real uvicorn and Streamlit servers on free local ports, drives the
Streamlit script against the live API, checks the result against the saved
pipeline, and shuts everything down. Requires `python -m churn.train` first.

    python scripts/e2e_smoke.py
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.inputs import DEFAULTS, build_customer_payload  # noqa: E402
from churn.predict import load_model  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 60) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.read().decode()
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not respond within {timeout}s")


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    api_port, ui_port = free_port(), free_port()
    api_url = f"http://127.0.0.1:{api_port}"
    servers = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(api_port)],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                          "--server.headless", "true", "--server.address", "127.0.0.1",
                          "--server.port", str(ui_port), "--browser.gatherUsageStats", "false"],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env={**os.environ, "CHURN_API_URL": api_url}),
    ]
    try:
        check('"model_loaded":true' in wait_for(f"{api_url}/health"), f"FastAPI healthy on {api_url}")
        check(wait_for(f"http://127.0.0.1:{ui_port}/_stcore/health") == "ok", f"Streamlit server up on port {ui_port}")

        model = load_model()
        cases = {
            "default form (month-to-month, fiber)": {},
            "two-year contract, 60 months": {"contract": "Two year", "tenure": 60},
            "no internet, no phone": {"internet_service": "No", "phone_service": "No"},
        }
        for name, changes in cases.items():
            app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
            app.sidebar.text_input[0].set_value(api_url).run()
            for label, value in [("Contract", changes.get("contract")), ("Internet service", changes.get("internet_service"))]:
                if value:
                    next(s for s in app.selectbox if s.label == label).set_value(value).run()
            if "tenure" in changes:
                next(n for n in app.number_input if n.label == "Tenure (months)").set_value(changes["tenure"]).run()
            if "phone_service" in changes:
                next(r for r in app.radio if r.label == "Phone service").set_value(changes["phone_service"]).run()
            app.button[0].click().run()
            check(not app.exception, f"{name}: UI ran without exceptions")

            metrics = {m.label: m.value for m in app.metric}
            form = {k: v for k, v in DEFAULTS.items() if k != "auto_total_charges"}
            form.update({k: v for k, v in changes.items()})
            form["total_charges"] = round(form["tenure"] * form["monthly_charges"], 2)
            payload = build_customer_payload(**form)
            expected = model.pipeline.predict_proba(pd.DataFrame([payload]))[0, 1]
            check(metrics.get("Churn probability") == f"{expected:.1%}",
                  f"{name}: UI shows {metrics.get('Churn probability')} = artifact {expected:.1%} ({metrics.get('Risk level')})")
    finally:
        for server in servers:
            server.terminate()
            server.wait(timeout=15)
    print("Smoke test passed; servers stopped.")


if __name__ == "__main__":
    main()
