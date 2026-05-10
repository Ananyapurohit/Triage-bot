# ED Triage - XAI

ED Triage XAI — Emergency Department Triage with Explainable AI
================================================================

REQUIREMENTS
------------
Python 3.9+

Install dependencies:
  pip install flask flask-cors scikit-learn pandas numpy requests.


FILES
-----
  backend_app_N200.py     → ML models + REST API (runs on port 5000)
  middleware.py           → Gateway that starts the backend and serves the frontend (runs on port 8080)
  frontend_connected.html → Dashboard UI (served automatically by middleware)


HOW TO RUN
----------
Put all three files in the same folder, then run:

  python middleware.py

Wait ~10–15 seconds for the models to finish training, then open:

🌐  Dashboard  → http://localhost:8080

That's it. The middleware handles starting the backend automatically.


USAGE
-----
- Fill in the patient details on the Predict tab.
- The triage prediction (High / Medium / Low) updates live as you type.
- Use "Compare All" to run all 3 models side by side on the same patient.
- Check the Metrics tab for model accuracy and ROC curves.
- Check the Dataset tab to inspect the training data.


NOTES
-----
- If port 5000 or 8080 is already in use, stop the process using that port first
- The backend retrains from scratch on every restart (takes ~15 seconds)
- This is a research prototype — not for real clinical use
