"""PayGuard AI: Streamlit dashboard for reviewing PayPal payments.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from payguard.agent import PayGuardAgent, QueuePlan
from payguard.explain import DEFAULT_GROQ_MODEL
from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient, PayPalError
from payguard.risk import RiskModel
from payguard.simulate import simulate_payments

load_dotenv()
st.set_page_config(page_title="PayGuard AI", page_icon="🛡️", layout="wide")

LEVELS = {
    "high":   {"label": "High",   "fg": "#B42318", "bg": "#FEE4E2", "ring": "#D92D20"},
    "medium": {"label": "Medium", "fg": "#B54708", "bg": "#FEF0C7", "ring": "#F79009"},
    "low":    {"label": "Low",    "fg": "#067647", "bg": "#DCFAE6", "ring": "#17B26A"},
}
DECISIONS = {
    None: ("Pending", "#475467"),
    "accept": ("Approved", "#067647"),
    "monitor": ("Monitoring", "#B54708"),
    "refund": ("Refunded", "#B42318"),
}
ACTION_LABELS = {"accept": "Approve", "monitor": "Monitor", "refund": "Refund"}
LIVE_REFRESH_SECONDS = 3

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; max-width: 1440px; }
.pg-header { display:flex; align-items:center; gap:14px; margin-bottom: 6px; }
.pg-logo { width:44px; height:44px; border-radius:12px; display:flex; align-items:center;
  justify-content:center; font-size:24px; background:linear-gradient(135deg,#003087,#0070E0); }
.pg-title { font-size:1.75rem; font-weight:700; color:#003087; line-height:1.1; }
.pg-sub { color:#667085; font-size:0.95rem; }
.pg-shop { margin-left:auto; padding:8px 16px; border-radius:999px; background:#003087;
  color:#fff !important; text-decoration:none !important; font-weight:600; font-size:0.9rem; }
.pg-shop:hover { background:#0070E0; }
.pg-live { color:#067647; font-size:0.82rem; font-weight:500; margin:-2px 0 10px 58px; }
.pg-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
  font-weight:600; }
.pg-card { background:#fff; border:1px solid #E4E7EC; border-radius:14px; padding:18px 20px;
  box-shadow:0 1px 2px rgba(16,24,40,.04); }
.pg-kpi-label { color:#667085; font-size:0.8rem; font-weight:500; text-transform:uppercase;
  letter-spacing:.04em; }
.pg-kpi-value { font-size:1.6rem; font-weight:700; color:#1A2433; margin-top:2px; }
.pg-amount { font-size:2rem; font-weight:700; color:#1A2433; }
.pg-muted { color:#667085; font-size:0.88rem; }
.pg-ring { width:118px; height:118px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; margin-left:auto; }
.pg-ring-inner { width:92px; height:92px; border-radius:50%; background:#fff; display:flex;
  flex-direction:column; align-items:center; justify-content:center; }
.pg-ring-score { font-size:1.7rem; font-weight:700; line-height:1; }
.pg-ring-cap { font-size:0.7rem; color:#667085; margin-top:3px; }
.pg-sentence { background:#F0F6FF; border-left:4px solid #0070E0; border-radius:8px;
  padding:12px 14px; font-size:1.02rem; color:#1A2433; margin:6px 0 4px 0; }
.pg-reason { display:flex; align-items:center; gap:12px; padding:8px 0;
  border-bottom:1px solid #F2F4F7; }
.pg-reason:last-child { border-bottom:none; }
.pg-arrow { width:26px; height:26px; border-radius:8px; display:flex; align-items:center;
  justify-content:center; font-weight:700; flex-shrink:0; }
.pg-bar { height:6px; border-radius:3px; margin-top:5px; }
.pg-proposal { border:1px solid #E4E7EC; border-radius:12px; padding:12px 14px;
  background:#FCFCFD; margin-top:10px; }
.pg-section { font-weight:600; color:#344054; margin:14px 0 4px 0; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)


# --- resources ----------------------------------------------------------------

@st.cache_resource
def get_model() -> RiskModel:
    return RiskModel.load()


@st.cache_resource
def get_ledger() -> Ledger:
    return Ledger()


@st.cache_resource
def get_paypal() -> tuple[PayPalClient | None, str]:
    try:
        client = PayPalClient()
        client.get_access_token()
        return client, ""
    except (ValueError, PayPalError, OSError) as e:
        return None, str(e)


def flag(cc: str | None) -> str:
    if not cc or len(cc) != 2 or not cc.isalpha():
        return "🌐"
    return "".join(chr(127397 + ord(c)) for c in cc.upper())


def when(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %d, %H:%M UTC")


def pill(text: str, fg: str, bg: str) -> str:
    return f'<span class="pg-pill" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


model, ledger = get_model(), get_ledger()
paypal, paypal_error = get_paypal()
agent = PayGuardAgent(ledger, model, paypal)
groq_on = bool(os.getenv("GROQ_API_KEY"))
groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


# --- sidebar --------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛡️ PayGuard AI")
    if paypal:
        st.success("PayPal sandbox connected", icon="✅")
    else:
        st.error(f"PayPal unavailable: {paypal_error}", icon="⚠️")
    if groq_on:
        st.info(f"Explanations: Groq · `{groq_model}`", icon="✨")
    else:
        st.warning("No GROQ_API_KEY: explanations use a template", icon="📝")

    st.markdown("#### Demo")
    st.link_button("🛍️ Open demo shop", "/shop/", width="stretch",
                   help="Buy with a PayPal sandbox buyer account; the payment appears here")
    if st.button("⚡ Simulate payments", width="stretch", disabled=paypal is None,
                 help="Creates 10 real PayPal sandbox payments: 6 ordinary (incl. 2 foreign), 4 unusually large foreign orders"):
        bar = st.progress(0.0, text="Creating sandbox payments…")
        try:
            ids = simulate_payments(paypal, ledger,
                                    on_progress=lambda i, n, msg: bar.progress(i / n, text=msg))
            st.session_state["selected"] = None
            st.toast(f"{len(ids)} sandbox payments created", icon="✅")
        except PayPalError as e:
            st.error(f"PayPal error: {e.short}")
        bar.empty()

    st.markdown("#### Filters")
    level_filter = st.multiselect("Risk level", ["high", "medium", "low"],
                                  default=["high", "medium", "low"],
                                  format_func=lambda v: LEVELS[v]["label"])
    pending_only = st.toggle("Pending review only", value=False)

    meta = model.meta
    st.markdown("#### Model")
    st.caption(
        f"LightGBM trained on PaySim · test PR-AUC **{meta['metrics']['test_pr_auc']:.2f}**  \n"
        f"Monitor ≥ **{model.thresholds['monitor']:g}** · Refund ≥ **{model.thresholds['refund']:g}**  \n"
        "Score = percentile among reference payments. Sandbox only, synthetic training data."
    )


# --- header & KPIs ------------------------------------------------------------

st.markdown("""
<div class="pg-header">
  <div class="pg-logo">🛡️</div>
  <div><div class="pg-title">PayGuard AI</div>
  <div class="pg-sub">Fraud review for PayPal sellers · every alert explained · you decide</div></div>
  <a class="pg-shop" href="/shop/" target="_blank">🛍️ Open demo shop</a>
</div>""", unsafe_allow_html=True)

alerts = agent.review()


# --- live updates ---------------------------------------------------------------
# Shop checkouts and verified PayPal webhooks write to the ledger from other
# requests; poll its fingerprint and rerun the page when it changes.

announce = st.session_state.pop("announce", 0)
st.session_state["ledger_version"] = ledger.version()


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_updates():
    seen, now = st.session_state.get("ledger_version"), ledger.version()
    if seen is not None and now != seen:
        st.session_state["announce"] = max(0, now[0] - seen[0])
        st.rerun()
    st.markdown('<div class="pg-live">● Live: shop payments and PayPal webhooks appear '
                'here automatically</div>', unsafe_allow_html=True)


live_updates()
for a in alerts[:min(announce, 3)]:
    st.toast(f"New PayPal payment: ${a.payment['amount']:,.2f} · score {a.score:.1f} "
             f"({LEVELS[a.level]['label'].lower()} risk)", icon="💳")

if not alerts:
    st.info("No payments yet. Click **⚡ Simulate payments** in the sidebar to create sandbox "
            "payments, or buy something in the [demo shop](/shop/) with a PayPal sandbox buyer "
            "account.")
    st.stop()

pending = [a for a in alerts if a.pending]
kpis = [
    ("Payments", f"{len(alerts)}"),
    ("Volume", f"${sum(a.payment['amount'] for a in alerts):,.2f}"),
    ("High risk · pending", f"{sum(a.level == 'high' for a in pending)}"),
    ("Medium risk · pending", f"{sum(a.level == 'medium' for a in pending)}"),
    ("Refunded", f"${sum(a.payment['refunded'] for a in alerts):,.2f}"),
]
for col, (label, value) in zip(st.columns(len(kpis)), kpis):
    col.markdown(f'<div class="pg-card"><div class="pg-kpi-label">{label}</div>'
                 f'<div class="pg-kpi-value">{value}</div></div>', unsafe_allow_html=True)
st.write("")


# --- agent: review the whole queue -----------------------------------------------

PLAN_ROWS = [("refund", "↩️ Refund", "high"), ("monitor", "👁️ Monitor", "medium"),
             ("accept", "✅ Approve", "low")]


def close_plan_refunds():
    st.session_state.pop("plan_refunds", None)


@st.dialog("Confirm the agent's refunds", on_dismiss=close_plan_refunds)
def confirm_plan_refunds(plan: QueuePlan):
    by_id = {a.capture_id: a for a in alerts}
    ids = [cid for cid in plan.ids("refund") if cid in by_id and by_id[cid].pending]
    total = sum(by_id[c].payment["amount"] - by_id[c].payment["refunded"] for c in ids)
    st.write(f"Refund **{len(ids)}** high-risk payment{'s' if len(ids) != 1 else ''}, "
             f"**${total:,.2f}** in total, through PayPal?")
    for cid in ids:
        a = by_id[cid]
        st.markdown(f"- **${a.payment['amount']:,.2f}** · "
                    f"{html.escape(a.payment.get('payer_email') or 'unknown payer')} · "
                    f"score {a.score:.1f}")
    st.caption("This sends real refund requests to the PayPal sandbox and cannot be undone. "
               "Skipped refunds stay in the queue.")
    c1, c2 = st.columns(2)
    if c1.button("Yes, refund them", type="primary", width="stretch"):
        res = agent.apply_plan(plan, refunds_confirmed=True)
        refunds = res["done"]["refund"]
        if refunds:
            st.session_state["plan_flash"] = (f"{len(refunds)} refund(s) completed on PayPal: "
                                              + ", ".join(f"`{r}`" for r in refunds))
        if res["failed"]:
            st.session_state["plan_flash_error"] = " · ".join(f"capture {c}: {e}"
                                                              for c, e in res["failed"])
        st.session_state.pop("queue_plan", None)
        close_plan_refunds()
        st.rerun()
    if c2.button("Not now", width="stretch"):
        st.session_state["plan_flash"] = "Refunds skipped: those payments stay in the queue."
        st.session_state.pop("queue_plan", None)
        close_plan_refunds()
        st.rerun()


with st.container(border=True):
    head, action = st.columns([2.6, 1], vertical_alignment="center")
    head.markdown("**🤖 PayGuard agent** <span class='pg-muted'>· reviews every pending payment "
                  "and proposes a plan. You approve it; refunds need a second confirmation."
                  "</span>", unsafe_allow_html=True)
    if action.button("🤖 Let the agent review the queue", type="primary",
                     width="stretch", disabled=not pending, key="b_review_queue"):
        plan = agent.plan_queue(alerts)
        with st.spinner("The agent is reviewing the queue…"):
            summary, source = agent.summarize(plan)
        st.session_state["queue_plan"] = (plan, summary, source)
    if msg := st.session_state.pop("plan_flash", None):
        st.success(msg, icon="✅")
    if msg := st.session_state.pop("plan_flash_error", None):
        st.error(f"Some refunds failed: {msg}", icon="⚠️")

    if stored := st.session_state.get("queue_plan"):
        plan, summary, source = stored
        st.markdown(f'<div class="pg-sentence">{html.escape(summary)}</div>',
                    unsafe_allow_html=True)
        st.caption(f"✨ Written by {groq_model} on Groq · every number checked against the plan "
                   "computed by PayGuard" if source == "llm"
                   else "📝 Template summary (LLM unavailable or not grounded)")
        cols = st.columns(4)
        for col, (act, label, lvl) in zip(cols, PLAN_ROWS):
            col.markdown(
                f'<div class="pg-card"><div class="pg-kpi-label">{label}</div>'
                f'<div class="pg-kpi-value" style="color:{LEVELS[lvl]["fg"]}">{plan.count[act]}</div>'
                f'<div class="pg-muted">${plan.total[act]:,.2f}</div></div>', unsafe_allow_html=True)
        cols[3].markdown(
            '<div class="pg-card"><div class="pg-kpi-label">Est. loss avoided</div>'
            f'<div class="pg-kpi-value">${plan.loss_avoided:,.2f}</div>'
            '<div class="pg-muted">if the refunds are made</div></div>', unsafe_allow_html=True)
        if (not st.session_state.get("plan_refunds")
                and {a.capture_id for a in pending} != {cid for cid, _, _ in plan.items}):
            st.caption("⚠️ The queue changed since this plan. Only the payments in the plan "
                       "are affected; ask the agent again to include the others.")
        st.write("")
        b1, b2, _ = st.columns([1.3, 1, 2])
        n_refund = plan.count["refund"]
        if b1.button("✅ Apply this plan", type="primary", width="stretch",
                     key="b_apply_plan",
                     help="Records the approvals and monitors now; refunds ask for confirmation"):
            res = agent.apply_plan(plan, refunds_confirmed=False)
            st.session_state["plan_flash"] = (
                f"Plan applied: {len(res['done']['accept'])} approved, "
                f"{len(res['done']['monitor'])} monitored."
                + (f" Confirm the {n_refund} refund(s) in the dialog."
                   if n_refund and paypal is not None else ""))
            if n_refund and paypal is not None:
                st.session_state["plan_refunds"] = True
            else:
                st.session_state.pop("queue_plan", None)
            st.rerun()
        if b2.button("Discard", width="stretch", key="b_discard_plan"):
            st.session_state.pop("queue_plan", None)
            st.rerun()
        if st.session_state.get("plan_refunds"):
            confirm_plan_refunds(plan)
st.write("")


# --- payment list ---------------------------------------------------------------

shown = [a for a in alerts if a.level in level_filter and (a.pending or not pending_only)]
left, right = st.columns([1.75, 1], gap="large")

with left:
    st.markdown('<div class="pg-section">Payments</div>', unsafe_allow_html=True)
    if not shown:
        st.info("No payment matches the filters.")
        st.stop()
    table = pd.DataFrame({
        "Risk": [LEVELS[a.level]["label"] for a in shown],
        "Score": [a.score for a in shown],
        "Amount": [f"${a.payment['amount']:,.2f}" for a in shown],
        "Payer": [(a.payment.get("payer_email") or "unknown").split("@")[0] for a in shown],
        "Country": [f"{flag(a.payment.get('country'))} {a.payment.get('country') or '—'}"
                    for a in shown],
        "Time (UTC)": [when(a.payment["created_at"]).removesuffix(" UTC") for a in shown],
        "Status": [DECISIONS[a.payment.get("decision")][0] for a in shown],
    })

    def style_risk(v):
        lvl = next(k for k, s in LEVELS.items() if s["label"] == v)
        return (f"color:{LEVELS[lvl]['fg']};background-color:{LEVELS[lvl]['bg']};"
                "font-weight:600;text-align:center")

    def style_status(v):
        color = next(c for label, c in DECISIONS.values() if label == v)
        return f"color:{color};font-weight:500"

    styled = table.style.map(style_risk, subset=["Risk"]).map(style_status, subset=["Status"])
    event = st.dataframe(
        styled, hide_index=True, width="stretch", height=min(38 + 35 * len(shown), 560),
        on_select="rerun", selection_mode="single-row", key="payments_table",
        column_config={
            "Risk": st.column_config.TextColumn("Risk", width=72),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                     format="%.1f", width=110),
            "Amount": st.column_config.TextColumn("Amount", width=92),
            "Payer": st.column_config.TextColumn("Payer", width=120),
            "Country": st.column_config.TextColumn("Country", width=72),
            "Time (UTC)": st.column_config.TextColumn("Time (UTC)", width=104),
            "Status": st.column_config.TextColumn("Status", width=112),
        },
    )
    rows = event.selection.rows
    if rows:
        st.session_state["selected"] = shown[rows[0]].capture_id
    st.caption("Select a row to review it. Payer and country of simulated payments come from "
               "the demo buyer profile.")

selected_id = st.session_state.get("selected")
alert = next((a for a in shown if a.capture_id == selected_id), None)
if alert is None:  # default: the riskiest payment still waiting for a decision
    alert = max(pending or shown, key=lambda a: a.score)


# --- alert detail ---------------------------------------------------------------

def close_refund_dialog():
    st.session_state.pop("refund_for", None)


# The dialog stays open while "refund_for" is set, so it survives full reruns.
@st.dialog("Confirm refund", on_dismiss=close_refund_dialog)
def confirm_refund(a):
    remaining = a.payment["amount"] - a.payment["refunded"]
    st.write(f"Refund **${remaining:,.2f}** to **{a.payment.get('payer_email') or 'the payer'}** "
             "through PayPal?")
    st.caption("This sends a real refund request to the PayPal sandbox and cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Yes, refund", type="primary", width="stretch"):
        try:
            res = agent.execute(a, "refund", seller_confirmed=True)
            st.session_state["flash"] = f"Refund {res['refund_id']} completed on PayPal."
        except PayPalError as e:
            st.session_state["flash_error"] = f"Refund failed: {e.short}"
        except (ValueError, RuntimeError) as e:
            st.session_state["flash_error"] = f"Refund failed: {e}"
        close_refund_dialog()
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        close_refund_dialog()
        st.rerun()


with right:
    st.markdown('<div class="pg-section">Alert detail</div>', unsafe_allow_html=True)
    if msg := st.session_state.pop("flash", None):
        st.success(msg, icon="✅")
    if msg := st.session_state.pop("flash_error", None):
        st.error(msg, icon="⚠️")

    p, lv = alert.payment, LEVELS[alert.level]
    status, status_color = DECISIONS[p.get("decision")]
    pct = max(alert.score, 2)

    with st.container(border=True):
        top_l, top_r = st.columns([1.6, 1])
        top_l.markdown(
            f'<div class="pg-amount">${p["amount"]:,.2f}</div>'
            f'<div class="pg-muted">{html.escape(p.get("payer_email") or "unknown payer")}</div>'
            f'<div class="pg-muted">{flag(p.get("country"))} {p.get("country") or "—"} · '
            f'{when(p["created_at"])}</div>'
            f'<div class="pg-muted" style="font-size:.78rem">Capture {p["capture_id"]}</div>'
            f'<div style="margin-top:8px">{pill(lv["label"] + " risk", lv["fg"], lv["bg"])} '
            f'{pill(status, status_color, "#F2F4F7")} '
            f'{pill("✓ PayPal webhook", "#175CD3", "#EFF8FF") if p.get("webhook_verified_at") else ""}'
            f'</div>',
            unsafe_allow_html=True)
        top_r.markdown(
            f'<div class="pg-ring" style="background:conic-gradient({lv["ring"]} {pct}%, '
            f'#EAECF0 0)"><div class="pg-ring-inner"><div class="pg-ring-score" '
            f'style="color:{lv["fg"]}">{alert.score:.1f}</div>'
            f'<div class="pg-ring-cap">risk score</div></div></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="pg-section">Why this score</div>',
                    unsafe_allow_html=True)
        with st.spinner("Writing explanation…"):
            exp = agent.explain(alert)
        st.markdown(f'<div class="pg-sentence">{html.escape(exp.sentence)}</div>',
                    unsafe_allow_html=True)
        st.caption(f"✨ Written by {groq_model} on Groq · every number checked against the "
                   "model's reasons" if exp.source == "llm"
                   else "📝 Template explanation (LLM unavailable or not grounded)")

        st.markdown('<div class="pg-section">Top reasons (SHAP)</div>', unsafe_allow_html=True)
        biggest = max(abs(r.shap) for r in alert.reasons) or 1
        rows_html = ""
        for r in alert.reasons:
            up = r.raises_risk
            fg, bg = ("#B42318", "#FEE4E2") if up else ("#067647", "#DCFAE6")
            width = 100 * abs(r.shap) / biggest
            rows_html += (
                f'<div class="pg-reason"><div class="pg-arrow" style="color:{fg};background:{bg}">'
                f'{"↑" if up else "↓"}</div><div style="flex:1">'
                f'<div>{html.escape(r.text)}</div>'
                f'<div class="pg-bar" style="width:{width:.0f}%;background:{fg};opacity:.55"></div>'
                f'</div><div class="pg-muted" style="width:92px;text-align:right">'
                f'{"raises" if up else "lowers"} risk</div></div>')
        st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown(
            f'<div class="pg-proposal"><b>PayGuard proposes: '
            f'{ACTION_LABELS[alert.proposed_action]}</b><br>'
            f'<span class="pg-muted">{html.escape(alert.proposal_rationale)}</span></div>',
            unsafe_allow_html=True)

        st.write("")
        refunded = p.get("decision") == "refund"
        b1, b2, b3 = st.columns(3)
        kind = lambda act: "primary" if act == alert.proposed_action else "secondary"  # noqa: E731
        if b1.button("✅ Approve", type=kind("accept"), width="stretch",
                     disabled=refunded or p.get("decision") == "accept", key="b_accept"):
            agent.execute(alert, "accept", seller_confirmed=True)
            st.session_state["flash"] = "Payment approved."
            st.rerun()
        if b2.button("👁️ Monitor", type=kind("monitor"), width="stretch",
                     disabled=refunded or p.get("decision") == "monitor", key="b_monitor"):
            agent.execute(alert, "monitor", seller_confirmed=True)
            st.session_state["flash"] = "Payment marked for monitoring."
            st.rerun()
        if b3.button("↩️ Refund", type=kind("refund"), width="stretch",
                     disabled=refunded or paypal is None, key="b_refund"):
            st.session_state["refund_for"] = alert.capture_id
        if st.session_state.get("refund_for") == alert.capture_id and not refunded:
            confirm_refund(alert)
        st.caption("Nothing is sent to PayPal until you click. Refunds ask for confirmation.")

        history = ledger.decisions(alert.capture_id)
        if history:
            with st.expander(f"Decision history ({len(history)})"):
                for h in history:
                    if h.get("source") == "paypal":
                        line = f"{when(h['decided_at'])} · refunded directly in PayPal"
                    else:
                        who = ("seller approved the agent's plan:"
                               if h.get("source") == "agent_plan" else "seller chose")
                        line = (f"{when(h['decided_at'])} · proposed **{h['proposed_action']}**, "
                                f"{who} **{h['chosen_action']}**")
                    if h["refund_id"]:
                        line += f" · PayPal refund `{h['refund_id']}`"
                    st.markdown(line)
