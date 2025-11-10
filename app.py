import os
import io
import json
import zipfile
import base64
import datetime as dt

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from markdown2 import markdown
from dotenv import load_dotenv
from openai import OpenAI

import matplotlib.pyplot as plt

load_dotenv()

# ------------------ helpers ------------------
def _to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def chart_scores(heuristics_dict: dict) -> bytes:
    labels = ["H1","H2","H3","H4","H5","H6","H7","H8","H9","H10"]
    values = [heuristics_dict.get(h, 0) for h in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0,3); plt.title("Heuristic Scores (0–3)")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close()
    return buf.getvalue()

def chart_impact_effort(rows: list) -> bytes:
    m = {"Low":1,"Medium":2,"High":3}
    xs, ys, ids = [], [], []
    for r in rows:
        xs.append(m.get((r or {}).get("effort","Medium"),2))
        ys.append(m.get((r or {}).get("impact","Medium"),2))
        ids.append((r or {}).get("issue_id",""))
    plt.figure()
    plt.scatter(xs, ys)
    for x,y,i in zip(xs,ys,ids): plt.text(x+0.03,y+0.03,i)
    plt.xticks([1,2,3],["Low","Med","High"]); plt.yticks([1,2,3],["Low","Med","High"])
    plt.xlabel("Effort"); plt.ylabel("Impact"); plt.title("Impact vs Effort")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close()
    return buf.getvalue()

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, (dict,)): return [x]
    if isinstance(x, str): return [x]  # treat whole string as one bullet
    return []

def _parse_bbox(b):
    if isinstance(b, list) and len(b) >= 4: return b[:4]
    if isinstance(b, str):
        try:
            parts = [float(t) for t in b.strip().strip("[]()").split(",")]
            return parts[:4] if len(parts) >= 4 else None
        except Exception:
            return None
    return None

# ------------------ config ------------------
st.set_page_config(page_title="AI Heuristic Reviewer", page_icon="🕵️‍♀️", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = st.secrets.get("OPENAI_VISION_MODEL") or os.getenv("OPENAI_VISION_MODEL","gpt-4o-mini")
ORG = st.secrets.get("OPENAI_ORG")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is missing. Add it in Streamlit → Advanced settings → Secrets.")
    st.stop()

client_kwargs = {"api_key": OPENAI_API_KEY}
if ORG: client_kwargs["organization"] = ORG
client = OpenAI(**client_kwargs)

# ------------------ rubric & templates ------------------
HEURISTIC_RUBRIC = r"""
Core Heuristics (0–3):
1. Visibility of system status
2. Match between system & real world
3. User control & freedom
4. Consistency & standards
5. Error prevention
6. Recognition vs recall (memory load)
7. Flexibility & efficiency
8. Aesthetic & minimalist design
9. Error handling (recognize/diagnose/recover)
10. Help & documentation

UX Quality (0–3): IA, Interaction clarity, Content, Cognitive load
A11y spot-checks (pass/warn/fail): Contrast (AA), focus order, targets ≥44×44, labels, keyboard reachability, visible error text
Severity: 0=None, 1=Low, 2=Moderate, 3=Major
"""

JSON_SCHEMA = {
  "meta": {
    "title": "Heuristic Review — <screen/product>",
    "date": "<YYYY-MM-DD>",
    "screens": [{"id":"screen_1","filename":"upload_1.png","resolution":[1280,720]}],
    "summary": {"top_issues": [], "quick_wins": []},
    "scores": {
      "heuristics": {"H1":0,"H2":0,"H3":0,"H4":0,"H5":0,"H6":0,"H7":0,"H8":0,"H9":0,"H10":0},
      "ux_quality": {"IA":0,"Interaction":0,"Content":0,"CognitiveLoad":0},
      "a11y": {"contrast":"","focus":"","targets":"","labels":""}
    }
  },
  "issues": [],
  "impact_effort": [],
  "assumptions": []
}

REPORT_TEMPLATE_MD = r"""
# Heuristic Review — <Product / Flow>
Owner: {owner} | Date: {date} | Version: v1

## Executive Summary
- Top issues:
{top_issues}
- Quick wins:
{quick_wins}
- Assumptions:
{assumptions}

## Heuristics Scorecard
| Heuristic | Score (0–3) |
|---|---|
| Visibility of system status | {h1} |
| Match between system & real world | {h2} |
| User control & freedom | {h3} |
| Consistency & standards | {h4} |
| Error prevention | {h5} |
| Recognition vs recall | {h6} |
| Flexibility & efficiency | {h7} |
| Aesthetic & minimalist | {h8} |
| Error handling | {h9} |
| Help & documentation | {h10} |

## Charts
{charts}

## Annotated Screens
{annotated_images}

## Detailed Findings
{findings}

## Impact vs Effort (IDs)
- {impact_effort}

## Appendix
- Method & rubric: Nielsen + A11y spot checks
- Inputs: {num_screens} screen(s)
- Limitations: Automated analysis; confirm with user testing where possible.
"""

SYSTEM_PROMPT = f"""
You are a Senior UX Evaluator. Analyze UI screenshots and produce a professional heuristic review.

STRICT OUTPUT:
- Return TWO parts separated by `====JSON====` and `====REPORT====`.
  A) JSON strictly matching schema keys below.
  B) A polished Markdown report (exec summary, scorecard, annotated references #1, #2..., detailed findings with heuristic, severity, EVIDENCE with bbox, recommendation with ACCEPTANCE CRITERIA, impact/effort list).

QUALITY RULES:
- Always produce **at least 6 concrete issues** across the screens, unless the UI is near-perfect (then produce 3 high-quality observations). Prefer specific, evidence-based findings.
- Use **normalized bbox** (0–1, x,y,w,h) and tie evidence to UI elements by name.
- Fill `meta.summary.top_issues` and `quick_wins` with short, high-signal bullets.
- Severity must follow:
  Impact: 1 cosmetic, 2 slows task, 3 blocks task
  Frequency: 1 rare, 2 common, 3 very common
  Recoverability: 1 easy undo, 2 effort, 3 no undo
  Severity = round((Impact+Frequency+Recoverability)/3) → 1..3 (show factors in Markdown).
- Avoid generic advice; use tokens like “primary/secondary button”, “label”, “CTA”, “filter chip”.

RUBRIC:
{HEURISTIC_RUBRIC}

SCHEMA KEYS (for guidance; do not print literally): {list(JSON_SCHEMA.keys())}
STYLE EXAMPLE (imitate structure & specificity):
- Heuristic: H4 Consistency & standards
- Evidence: Primary CTA “Pay Now” uses same style as secondary buttons; title case inconsistent (“Pay now”). Region #1 [0.62,0.14,0.12,0.06]
- Severity: 2 (Impact=2, Frequency=2, Recoverability=2)
- Recommendation: Standardize primary CTA token and Title Case across cards.
- Acceptance criteria:
  - Primary CTA uses btn/primary token (color/weight)
  - Labels use Title Case
  - Secondary actions use btn/secondary
"""

# ------------------ UI ------------------
st.title("🕵️‍♀️ AI Heuristic Reviewer")
st.caption("Upload 1–5 screenshots. Get a professional heuristic review with annotated callouts, charts, and an impact/effort list.")

with st.sidebar:
    owner = st.text_input("Your name (for report)", value="Swati Minz")
    context = st.text_area("Context (optional)", placeholder="e.g., IRCTC search results; success = find & book")
    model_choice = st.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0 if DEFAULT_MODEL=="gpt-4o-mini" else 1)
    submit_btn = st.button("Run Heuristic Review", type="primary")

uploads = st.file_uploader("Upload 1–5 screenshots (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploads:
    st.subheader("Previews")
    for i,f in enumerate(uploads, start=1):
        img = Image.open(f).convert("RGB")
        st.image(img, caption=f"Screen {i}: {f.name}", use_column_width=True)

# ------------------ main ------------------
if submit_btn and uploads:
    today = dt.date.today().isoformat()
    local_images = []
    vision_inputs = [{"type":"text","text": f"Overall context: {context or '(none)'}\nDate: {today}"}]

    for i, f in enumerate(uploads, start=1):
        f.seek(0); bytes_data = f.read()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        w,h = img.size
        local_images.append((f"screen_{i}", f.name, img))
        # give the model per-screen meta text + image
        vision_inputs.append({"type":"text","text": f"Screen {i} = {f.name} | resolution: {w}x{h}. Identify concrete issues with bbox."})
        vision_inputs.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+_to_base64(img)}})

    vision_inputs.append({"type":"text","text":
        "Use the rubric. MUST include at least 6 issues overall with bbox and acceptance criteria. Return JSON and Markdown with the delimiters."})

    with st.spinner("Analyzing with OpenAI…"):
        try:
            resp = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content": vision_inputs}
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "Error code: 429" in msg:
                st.error("Your API key has **no available quota**. Add billing/credits at platform.openai.com → Billing, then try again.")
            else:
                st.error(f"OpenAI error: {e}")
            st.stop()

    if not raw:
        st.error("No response from the model."); st.stop()

    # -------- parse two-part output --------
    if "====JSON====" in raw and "====REPORT====" in raw:
        json_part = raw.split("====JSON====")[-1].split("====REPORT====")[0].strip()
        report_md = raw.split("====REPORT====")[-1].strip()
    else:
        # fallback: attempt to slice first JSON
        try:
            start = raw.find("{"); end = raw.rfind("}") + 1
            json_part = raw[start:end]; report_md = raw[:start]
        except Exception:
            json_part = "{}"; report_md = raw

    try:
        data = json.loads(json_part)
    except Exception:
        data = JSON_SCHEMA

    # -------- normalize --------
    issues = _as_list(data.get("issues", []))
    meta = data.get("meta", {}) or {}
    scores = (meta.get("scores") or {})
    H = (scores.get("heuristics") or {}) if isinstance(scores, dict) else {}

    # If model under-delivered, add a soft warning
    if len(issues) < 3:
        st.warning("The model returned very few findings. Try **Model = gpt-4o** for higher quality, or upload one more screen.")

    # -------- annotate images --------
    annotated_images_md = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report/report_raw.md", report_md)
        screen_map = {sid:(fname, im.copy()) for (sid,fname,im) in local_images}

        grouped = {}
        for iss in issues:
            sid = (iss or {}).get("screen_id") if isinstance(iss, dict) else None
            grouped.setdefault(sid or "screen_1", []).append(iss)

        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            pass

        for sid, items in grouped.items():
            if sid not in screen_map and screen_map:
                sid = list(screen_map.keys())[0]
            fname, base_img = screen_map[sid]
            w,h = base_img.size
            draw = ImageDraw.Draw(base_img)

            for idx, iss in enumerate(items, start=1):
                bbox = _parse_bbox((iss or {}).get("bbox"))
                if bbox:
                    x,y,bw,bh = bbox
                    x0,y0 = int(x*w), int(y*h)
                    x1,y1 = int((x+bw)*w), int((y+bh)*h)
                    draw.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=3)
                    tx,ty = x0, max(0, y0-24)
                    draw.rectangle([tx,ty,tx+24,ty+24], fill=(255,0,0))
                    draw.text((tx+7,ty+4), f"{idx}", fill=(255,255,255), font=font)

            buf = io.BytesIO(); base_img.save(buf, format="PNG")
            zf.writestr(f"images/{sid}_annotated.png", buf.getvalue())
            annotated_images_md.append(f"![{sid} annotated](images/{sid}_annotated.png)")

        # Originals
        for i,(sid,fname,im) in enumerate(local_images, start=1):
            buf = io.BytesIO(); im.save(buf, format="PNG")
            zf.writestr(f"images/original_{i}_{fname}", buf.getvalue())

    # -------- charts --------
    try:
        scores_png = chart_scores(H or {})
        ie_png = chart_impact_effort(data.get("impact_effort", []))
        with zipfile.ZipFile(zip_buffer,'a',zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("charts/heuristics.png", scores_png)
            zf.writestr("charts/impact_effort.png", ie_png)
        charts_md = "![Heuristic scores](charts/heuristics.png)\n\n![Impact vs Effort](charts/impact_effort.png)"
    except Exception:
        charts_md = "(charts unavailable)"

    # -------- summary text (robust to strings) --------
    def _bullets(x):
        items = _as_list(x)
        return "\n".join([f"  - {i}" for i in items]) or "  - (none)"

    top_issues_md = _bullets(((meta.get("summary") or {}).get("top_issues")))
    quick_wins_md = _bullets(((meta.get("summary") or {}).get("quick_wins")))
    assumptions_md = _bullets(data.get("assumptions"))

    # -------- findings block (robust) --------
    def fmt_issue(iss: dict) -> str:
        if not isinstance(iss, dict):  # fallback
            return "### (Unparsed issue)\n- Heuristic:\n- Evidence:\n- Recommendation:\n- Acceptance criteria:\n  - (none)\n"

        rec = iss.get("recommendation", {})
        action = ""
        ac_list = []

        if isinstance(rec, dict):
            action = rec.get("action","") or ""
            ac_list = rec.get("acceptance_criteria", []) or []
            if isinstance(ac_list, str): ac_list = [ac_list]
        elif isinstance(rec, list):
            ac_list = rec
        elif isinstance(rec, str):
            action = rec

        ac = "\n".join([f"  - {c}" for c in _as_list(ac_list)]) or "  - (none)"

        sev = iss.get("severity", 0)
        try: sev = int(sev)
        except Exception: sev = 0

        return f"""### {iss.get('id','ISS')} — {iss.get('title','(title)')} (Severity: {sev})
- Heuristic: {iss.get('heuristic','')}
- Evidence: {iss.get('evidence','')}
- Recommendation: {action}
- Acceptance criteria:
{ac}
"""

    findings_md = "\n".join([fmt_issue(i) for i in issues]) or "(No issues parsed)"
    impact_effort_md = ", ".join([f"{(ie or {}).get('issue_id')} [{(ie or {}).get('impact')}/{(ie or {}).get('effort')}]" for ie in _as_list(data.get("impact_effort"))]) or "(n/a)"

    final_report = REPORT_TEMPLATE_MD.format(
        owner=owner, date=today,
        top_issues=top_issues_md, quick_wins=quick_wins_md, assumptions=assumptions_md,
        h1=H.get("H1",""), h2=H.get("H2",""), h3=H.get("H3",""), h4=H.get("H4",""),
        h5=H.get("H5",""), h6=H.get("H6",""), h7=H.get("H7",""), h8=H.get("H8",""),
        h9=H.get("H9",""), h10=H.get("H10",""),
        charts=charts_md,
        annotated_images="\n\n".join(annotated_images_md) or "(annotated images included in ZIP)",
        findings=findings_md, impact_effort=impact_effort_md,
        num_screens=len(local_images)
    )

    st.subheader("Report Preview")
    st.markdown(final_report)

    with zipfile.ZipFile(zip_buffer,'a',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report/report_compiled.md", final_report)

    st.download_button("⬇️ Download report + images (ZIP)",
        data=zip_buffer.getvalue(), file_name="heuristic_review.zip")

else:
    st.info("Upload at least one screenshot, then click ‘Run Heuristic Review’.")
