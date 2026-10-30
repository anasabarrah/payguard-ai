# PayGuard AI

AI agent that reviews a seller's PayPal payments for fraud, explains every
alert in plain English, and proposes an action. The seller confirms before
anything happens on PayPal. Built for the PayPal AI Hackathon.

- **Risk score 0–100** for every payment (LightGBM)
- **Up to 3 reasons** per alert (SHAP), plus **one plain-English sentence** written
  by an LLM on Groq, with every number checked against those reasons
- **Agent proposal**: approve, monitor or refund. Refunds go through the
  PayPal REST API only after the seller confirms.
- **No judgment on nationality**: the payer's country is only used as a
  domestic vs cross-border signal, every foreign country is treated the same,
  and the seller always makes the final decision
- **Streamlit dashboard** with a one-click demo that creates real PayPal sandbox
  payments, some of them suspicious

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # the app
pip install -r requirements-dev.txt    # optional: tests and retraining
cp .env.example .env                   # then fill in the values
```

| Variable | Required | Purpose |
|---|---|---|
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` | yes | a **sandbox** app from the PayPal developer dashboard |
| `PAYPAL_WEBHOOK_ID` | for webhooks | ID returned by `scripts/register_paypal_webhook.py` |
| `GROQ_API_KEY` | no | plain-English explanations (free key at console.groq.com); without it a template sentence is used |
| `GROQ_MODEL` | no | defaults to `openai/gpt-oss-120b` |

`.env` is listed in `.gitignore` and is never committed.

## Run

```bash
PORT=8500 ./start.sh
```

Open http://localhost:8500 and click **⚡ Simulate payments** in the sidebar.
It creates 10 real sandbox card payments: 4 domestic regulars, 2 foreign
customers with a normal basket (CA, GB: still low, since being foreign is not
suspicious on its own), then 4 new foreign buyers with unusually large orders:
about 5x the usual ticket (DE, JP: medium) and 20x+ (AU, IN: high). Pick a payment
in the table to see its score, reasons and explanation, then click
**Approve**, **Monitor** or **Refund**. A refund asks for confirmation, then
calls PayPal.

Scores do not depend on the time of day, so the demo gives the same picture
whenever it runs. `tests/test_demo_levels.py` checks that the 10 demo payments
come out as 6 low, 2 medium and 2 high at 03:00, 12:00 and 22:00 UTC, on an
empty ledger and after 12 back-to-back demos in one day. Over 200 random demo
runs, low scores stay at or below 88.5, medium at 97.7–98.9 and high at 99.9 or
above. Six of the ten payments are ordinary on purpose: this keeps the 24 h
median ticket at the ordinary level when several people run the demo.

### PayPal checkout shop

Open http://localhost:8500/shop/ (or click **Open the PayGuard Demo Shop** in
the dashboard). Choose a fixed-price demo item and approve checkout with a
PayPal **sandbox buyer** account. The JavaScript SDK renders the PayPal button;
the app creates and captures the Orders v2 order server-side. The capture is
written to the same ledger and appears in the dashboard on its next refresh,
with a PayGuard score. The 6 low / 2 medium / 2 high **Simulate payments** demo
is unchanged.

The Render service uses Nginx on Render's single `$PORT`: it routes the
dashboard and its WebSockets to Streamlit, `/shop/` and `/api/` to a small Starlette API, and
`/webhooks/paypal` to the webhook handler. Both app servers share the SQLite
ledger. This preserves the existing dashboard while fitting one free web
service; the container runs three processes.

Register the sandbox webhook after deploying, using the same PayPal sandbox
app credentials configured in Render:

```bash
PAYGUARD_PUBLIC_URL=https://payguard-ai-1p86.onrender.com \
  python scripts/register_paypal_webhook.py
```

The script subscribes `PAYMENT.CAPTURE.COMPLETED` and
`PAYMENT.CAPTURE.REFUNDED` and prints the webhook ID. Set that value as
`PAYPAL_WEBHOOK_ID` in Render and redeploy. The listener verifies every event
through PayPal's `verify-webhook-signature` API and deduplicates event IDs.
Captures are added automatically; refund events update the ledger. A shop
checkout also records the server capture response immediately, so it does not
depend on webhook delivery. The dashboard reads the ledger on every rerun.

## Deploy on Render (free)

The repo contains a `Dockerfile` (Python 3.11 slim + `libgomp1` for LightGBM)
and a `render.yaml` Blueprint for a free web service.

1. In Render, choose **New → Blueprint** and connect this GitHub repository.
2. Render reads `render.yaml` and asks for the secret values:
   `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYPAL_WEBHOOK_ID` and
   `GROQ_API_KEY` (optional). Set the webhook ID after registering it above,
   then redeploy.
   `GROQ_MODEL` is preset.
3. Deploy. The app is served on the `onrender.com` URL Render gives you; the
   health check is `/_stcore/health`.

Behaviour on the free plan:
- The service sleeps after ~15 minutes without traffic; the next visit takes
  about a minute to wake it.
- The disk is ephemeral: the ledger (`data/ledger.db`) starts empty after each
  deploy or restart. Click **⚡ Simulate payments** to create demo payments.
- All visitors share one ledger, and anyone with the URL can trigger sandbox
  refunds. Only use sandbox credentials.

The app uses about 190 MB of RAM, within the free plan's 512 MB.

## Tests

```bash
pytest -q                               # offline: client, model, explanations, agent, UI
python scripts/paypal_smoke_test.py     # live sandbox: auth, order, capture, refund
```

The UI tests drive `app.py` headlessly with Streamlit's `AppTest`, using mocked
PayPal and Groq clients. They check, for example, that a refund never reaches
PayPal before the seller confirms.

## Architecture

```
            PayPal sandbox (REST v2: orders, captures, refunds)
                 ▲                                   │
     refund only │                                   │ create / capture
   after confirm │                                   ▼
┌──────────────────────────┐    ┌─────────────────────────────────────┐
│ app.py (Streamlit)       │    │ payguard/ledger.py  SQLite ledger    │
│ list · detail · buttons  │◄──►│ payments, decisions, cached sentences│
└────────────┬─────────────┘    └─────────────────────────────────────┘
             │
┌────────────▼─────────────┐  features   ┌──────────────────────────┐
│ payguard/agent.py        │────────────►│ payguard/risk.py         │
│ review → propose →       │             │ LightGBM → 0-100 score   │
│ execute(seller_confirmed)│◄────────────│ models/ (shipped)        │
└────────────┬─────────────┘   score     └──────────────────────────┘
             │ reasons
┌────────────▼─────────────┐
│ payguard/explain.py      │  TreeSHAP reasons → Groq sentence
│                          │  → number check → template fallback
└──────────────────────────┘
```

| Module | Role |
|---|---|
| `payguard/paypal_client.py` | OAuth, create/capture orders, refunds, Transaction Search |
| `payguard/ledger.py` | local record of payments, decisions (audit log) and explanations |
| `payguard/features.py` | features shared by training (PaySim) and inference (ledger) |
| `payguard/risk.py` | loads the model, turns probabilities into 0–100 scores and levels |
| `payguard/explain.py` | SHAP reasons, Groq sentence, number check, template |
| `payguard/agent.py` | proposes an action per level; executes only on seller confirmation |
| `payguard/simulate.py` | creates the demo sandbox payments |
| `scripts/` | dataset download, training, PayPal smoke test |

## Explanations

SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`), which
is exact for tree models and needs no extra dependency. Up to 3 reasons are
shown: the features that pushed the score up the most, then, for a low-risk
payment, the features that pulled it down. A feature with zero effect on the
score is never presented as a reason. With the current model only the amount
and the cross-border flag have an effect (see limitations), so alerts show 2 reasons.

The LLM receives only the score and the reasons as text, and is asked to
copy numbers exactly. Before a sentence is shown, every number in it, whether
written in digits ("22.5") or in words ("twenty"), must appear in those
facts. If not, PayGuard retries once and then falls back to a fill-in-the-blanks
template, which is also used with no `GROQ_API_KEY` or on any API error. The UI
labels which of the two produced the sentence.

## Agent and seller validation

| Risk level | Score | Proposal |
|---|---|---|
| low | < 95.0 | Approve |
| medium | 95.0 – 99.3 | Monitor (hold or watch before shipping) |
| high | ≥ 99.3 | Refund |

The agent only proposes. `PayGuardAgent.execute()` refuses to act without
`seller_confirmed=True`. In the UI, Approve and Monitor are recorded locally;
Refund opens a confirmation dialog and then refunds the remaining amount
through PayPal. The seller can always pick a different action than the
proposal. A refunded payment cannot be changed. Every decision is logged with
the score, the proposal and the PayPal refund id.

## Risk model

The trained model ships in `models/` (`payguard_lgbm.txt`, ~0.05 MB, plus
`model_meta.json` with thresholds and metrics), so the app runs without the
dataset. To retrain:

```bash
python scripts/download_paysim.py   # ~470 MB, checksum-verified
python scripts/train_model.py       # ~1 min, rewrites models/
```

**Features**, chosen because a PayPal seller can compute them from their own
payment history:

| Feature | Meaning |
|---|---|
| `amount_ratio` | amount / median payment over the previous 24 h (scale-free) |
| `payer_txn_24h` | payments by the same payer in the previous 24 h |
| `payer_is_new` | 1 if the payer has never paid before |
| `cross_border` | 1 if the payer's country differs from the seller's (simulated for training, see below) |

The hour of day is deliberately **not** a feature (see limitations).

LightGBM, with monotone constraints: risk can only increase with the amount
ratio, with payer velocity, and for a new payer.

**Risk score (0–100)** is the percentile of the model's fraud probability among
the validation payments: 97 means "riskier than 97% of reference payments".
Thresholds: **monitor ≥ 95.0** (top 5% of validation payments, a review budget)
and **refund ≥ 99.3** (the threshold that maximises F1 on validation).

**Evaluation**: temporal split on PaySim's hourly `step`. Validation is used
only for early stopping and thresholds; the numbers below are on the test window.

| Split | Steps (hours) | Rows | Frauds |
|---|---|---|---|
| train | 1–400 | 2,529,536 | 4,477 |
| validation | 401–550 | 158,652 | 1,588 |
| test | 551–743 | 82,221 | 2,148 |

| Test metric | With `cross_border` | Without `cross_border` |
|---|---|---|
| PR-AUC (baseline = prevalence 0.026) | **0.213** | 0.163 |
| Precision / recall / F1 at refund threshold | 0.46 / 0.19 / 0.27 | 0.42 / 0.15 / 0.22 |
| Precision / recall / F1 at monitor threshold | 0.21 / 0.37 / 0.27 | 0.18 / 0.36 / 0.24 |

These numbers are modest, and two earlier versions scored higher on paper.
One used the hour of day (0.441 PR-AUC), which was mostly a simulator artifact
and made scores depend on when the app was opened. The other used a
per-country variable (0.265), whose extra gain came only from assumed
per-country fraud rates. Both were dropped.

Accuracy is not reported: with 0.2–2.6% fraud, it is meaningless.

## Data & limitations

- **Dataset.** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
  (Lopez-Rojas et al., 2016) is a *synthetic* mobile-money dataset generated
  from one month of real logs: 6.36M transactions, 8,213 frauds. It is downloaded
  from a public Hugging Face mirror and checked against a fixed SHA-256.
- **Only `TRANSFER` and `CASH_OUT`.** PaySim contains fraud only in these two
  types, so the model is trained on them alone.
- **Balances are not used.** PaySim's balance columns are very predictive, but
  a seller never sees the payer's balance, so using them would not transfer to
  PayPal.
- **Amounts are relative.** PaySim amounts use an unspecified currency at a
  different scale from a typical PayPal checkout, so the model only sees the
  amount relative to the recent median ticket.
- **Country is a cross-border signal, not a judgment on nationality.** The
  model only knows whether the payer's country differs from the seller's; every
  foreign country is treated the same, and a foreign buyer with a normal basket
  stays low risk. A payment is flagged when cross-border meets an abnormal
  amount, and the seller always makes the final decision. The app shows the
  country for the seller's information, but the model never sees which
  country it is.
- **Cross-border is simulated.** PaySim has no geography. Each training
  transaction is marked cross-border at random (fixed seed): 20% of legitimate
  payments and 50% of fraudulent ones, for a US seller. These shares are
  **assumptions** (see `CROSS_BORDER_SHARE` in `payguard/features.py`), loosely
  based on the common observation that cross-border payments carry more fraud.
  They are **not measured data**, which is why the table above also shows
  metrics without `cross_border`: it adds about +0.05 PR-AUC, and that gain is
  synthetic by construction. A payer with no known country counts as domestic.
- **Payer history carries no signal in PaySim.** Payers almost never repeat
  (at most 3 transactions per payer; 460 training rows have a payment in the
  previous 24 h). The model therefore does not use `payer_txn_24h` and
  `payer_is_new` in practice (their SHAP values are exactly 0, so they never
  appear as reasons). They are kept, with monotone constraints, so that
  retraining on real seller history can use them.
- **No hour of day.** In PaySim, fraud is spread evenly over the day while
  legitimate volume collapses at night, so the hour looked extremely predictive
  (it carried ~60% of an earlier model's gain). That is mostly a simulator
  artifact. It also made the same payment score "low" at noon and "high" at
  night UTC, which is wrong for a seller reviewing payments at any hour. The
  feature was removed: test PR-AUC dropped from 0.441 to 0.265, and scores no
  longer depend on the time of day. The model now effectively relies on two
  signals, the relative amount and the (simulated) cross-border flag.
- **Drift.** PaySim's legitimate volume drops sharply after step ~400, so fraud
  prevalence goes from 0.18% (train) to 2.6% (test). That is why the score is a
  percentile, not a raw probability.
- **Transaction Search.** PayPal's `/v1/reporting/transactions` needs the
  "Transaction search" option on the app, and sandbox data can lag by hours.
  PayGuard therefore keeps a local SQLite ledger (`data/ledger.db`) of every
  order, capture and refund it makes, and uses that as the primary source.
- **Demo buyer profiles.** Card payments in the sandbox carry no payer email,
  and PayPal does not return the card's billing country. For simulated
  payments, the payer email and country shown in the app come from the demo
  buyer profile used to create the payment, which is also the card's billing
  country.
- **LLM sentences** can still be awkward or vague. The number check only
  guarantees that they contain no number absent from the model's reasons.

This is a hackathon prototype trained on synthetic data. It is not a production
fraud model.
