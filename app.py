import os
import io
import json
import zipfile
import datetime as dt
from typing import List, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from markdown2 import markdown
from dotenv import load_dotenv

load_dotenv()

# ===============================
# CONFIG: keys, model, client
# ===============================
# Read from Streamlit Secrets first, then env vars (local dev)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = st.secrets.get("OPENAI_VISION_MODEL") or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is missing. Add it in Streamlit → Advanced settings → Secrets.")
    st.stop()

# Create OpenAI client explicitly with the key
# (This avoids environment edge cases on Streamlit.)
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_SDK_V1 = True  # modern client
except Exception:
    # Compatibility fallback (older SDKs)
    import openai as openai_legacy
    openai_legacy.api_key = OPENAI_API_KEY
    client = None
    USE_SDK_V1 = False  # legacy client

# ===============================
# STREAMLIT APP SETUP
# ===============================
st.set_page_config(page_title="AI Heuristic Reviewer", page_icon="🕵️‍♀️", layout="wide")
st.title("🕵️‍♀️ AI Heuristic Reviewer")
st.caption("Upload 1–5 screenshots. Get a professional heuristic review with annotated callouts, scorecard, and an impact/effort list.")

# ---------- Helper content: Rubric & Template ----------
HEURISTIC_RUBRIC = r"""
Core Heuristics (score 0–3):
1. Visibility of system status
2. Match between system & real world
3. User control & freedom
4. Consistency & standards
5. Error prevention
6. Recognition vs recall (memory load)
7. Flexibility & efficiency (shortcuts, power use)
8. Aesthetic & minimalist design (signal-to-noise)
9. Help users recognize/diagnose/recover from errors
10. Help & documentation

UX Quality Dimensions (0–3): IA, Interaction clarity, Content design, Cognitive load
A11y spot-checks (pass/warn/fail): Contrast (AA), focus order, touch targets (44×44), labels, keyboard reachability, visible error text
Severity: 0=None, 1=Low, 2=Moderate, 3=Major
Recommendation format: action verb + rationale + acceptance criteria
"""

JSON_SCHEMA = {
    "meta": {
        "title": "Heuristic Review — <screen/product>",
        "date": "<YYYY-MM-DD>",
        "screens": [
            {"id": "screen_1", "filename": "upload_1.png", "resolution": [1280, 720]}
        ],
        "summary": {"top_issues": [], "quick_wins": []},
        "scores": {
            "heuristics": {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 0, "H6": 0, "H7": 0, "H8": 0, "H9": 0, "H10": 0},
            "ux_quality": {"IA": 0, "Interaction": 0, "Content": 0, "CognitiveLoad": 0},
            "a11y": {"contrast": "", "focus": "", "targets": "", "labels": ""}
        }
    },
    "issues": [
        {
            "id": "ISS-001",
            "screen_id": "screen_1",
            "title": "",
            "heuristic": "H8 Aesthetic & minimalist design",
            "severity": 2,
            "evidence": "",
            "bbox": [0.42, 0.77, 0.16, 0.06],
            "recommendation": {"action": "", "rationale": "", "acceptance_criteria": []}
        }
    ],
    "impact_effort": [{"issue_id": "ISS-001", "impact": "High", "effort": "Low"}],
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

STRICT RULES:
- Use the rubric below for evaluation and scoring.
- Produce TWO artifacts separated by a delimiter line `====JSON====` and `====REPORT====`:
  A) Compact JSON strictly following the provided schema keys (fill realistically). Use normalized bbox coords (0–1, x,y,w,h) when pointing to UI regions.
  B) A polished report in Markdown: exec summary (top issues, quick wins), scorecard, annotated callout references (like #1, #2), detailed findings (each with heuristic, severity, evidence, recommendation with acceptance criteria), impact/effort list.
- Prefer precise, actionable wording. Avoid generic tips. State assumptions if information is missing.

RUBRIC:
{HEURISTIC_RUBRIC}

SCHEMA KEYS (for guidance; do not print literally): {list(JSON_SCHEMA.keys())}
"""

# ===============================
# UI
# ===============================
with st.sidebar:
    owner = st.text_input("Your name (for report)", value="Swati Minz")
    context = st.text_area(
        "Context (optional)",
        placeholder="e.g., Mobile checkout for first-time users; success = task completion"
    )
    submit_btn = st.button("Run Heuristic Review", type="primary")

uploads = st.file_uploader(
    "Upload 1–5 screenshots (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

col_left, col_right = st.columns([1, 1])

if uploads:
    with col_left:
        st.subheader("Previews")
        for i, f in enumerate(uploads, start=1):
            img = Image.open(f).convert("RGB")
            st.image(img, caption=f"Screen {i}: {f.name}", use_column_width=True)

# ===============================
# Main action
# ===============================
if submit_btn and uploads:
    today = dt.date.today().isoformat()

    # Build a text description of each screen (fallback; vision attach upgrade can come later)
    content_parts = [{"type": "text", "text": f"Context: {context}\nDate: {today}"}]
    local_images = []  # keep for annotation later

    for i, f in enumerate(uploads, start=1):
        f.seek(0)
        bytes_data = f.read()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        w, h = img.size
        content_parts.append({
            "type": "text",
            "text": f"Screen {i} resolution: {w}x{h}. The screen shows UI elements for analysis."
        })
        local_images.append((f"screen_{i}", f.name, img))

    with st.spinner("Analyzing with OpenAI…"):
        try:
            user_text = (
                "Prepare heuristic review using the rubric. Screens described above. "
                "Return JSON and Markdown separated by delimiters."
            )

            if USE_SDK_V1:
                # Modern SDK call
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content_parts + [{"type": "text", "text": user_text}]}
                    ],
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content
            else:
                # Legacy fallback
                import openai as openai_legacy
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context: {context}\nDate: {today}"},
                    {"role": "user", "content": "Screens are uploaded by the user; analyze as described."},
                    {"role": "user", "content": user_text},
                ]
                legacy_resp = openai_legacy.ChatCompletion.create(
                    model=MODEL, messages=messages, temperature=0.2
                )
                raw = legacy_resp["choices"][0]["message"]["content"]

        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    # ===========================
    # Parse artifacts
    # ===========================
    if raw:
        if "====JSON====" in raw and "====REPORT====" in raw:
            json_part = raw.split("====JSON====")[-1].split("====REPORT====")[0].strip()
            report_md = raw.split("====REPORT====")[-1].strip()
        else:
            # fallback: try to parse first JSON in the response
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                json_part = raw[start:end]
                report_md = raw[:start]
            except Exception:
                json_part = "{}"
                report_md = raw

        try:
            data = json.loads(json_part)
        except Exception:
            data = JSON_SCHEMA  # safe fallback

        # ===========================
        # Annotate images from bbox
        # ===========================
        annotated_images_md = []
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save model's raw report too
            zf.writestr("report/report_raw.md", report_md)

            # Build a dict from uploaded images for easy lookup
            screen_map = {sid: (fname, im.copy()) for (sid, fname, im) in local_images}

            issues = data.get("issues", [])
            grouped = {}
            for iss in issues:
                sid = iss.get("screen_id", "screen_1")
                grouped.setdefault(sid, []).append(iss)

            # Load a font if available (optional)
            font = None
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                pass

            for sid, items in grouped.items():
                if sid not in screen_map and screen_map:
                    sid = list(screen_map.keys())[0]  # fallback to first image

                fname, base_img = screen_map[sid]
                w, h = base_img.size
                draw = ImageDraw.Draw(base_img)

                for idx, iss in enumerate(items, start=1):
                    bbox = iss.get("bbox")
                    if bbox and isinstance(bbox, list) and len(bbox) == 4:
                        x, y, bw, bh = bbox
                        x0, y0 = int(x * w), int(y * h)
                        x1, y1 = int((x + bw) * w), int((y + bh) * h)
                        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
                        tx, ty = x0, max(0, y0 - 24)
                        label = f"{idx}"
                        draw.rectangle([tx, ty, tx + 24, ty + 24], fill=(255, 0, 0))
                        draw.text((tx + 7, ty + 4), label, fill=(255, 255, 255), font=font)

                # Save annotated
                buf = io.BytesIO()
                base_img.save(buf, format="PNG")
                zf.writestr(f"images/{sid}_annotated.png", buf.getvalue())
                annotated_images_md.append(f"![{sid} annotated](images/{sid}_annotated.png)")

            # Also include originals
            for i, (sid, fname, im) in enumerate(local_images, start=1):
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                zf.writestr(f"images/original_{i}_{fname}", buf.getvalue())

        # ===========================
        # Build final Markdown report
        # ===========================
        scores = data.get("meta", {}).get("scores", {})
        H = scores.get("heuristics", {}) if isinstance(scores, dict) else {}

        top_issues = "\n".join([f"  - {t}" for t in data.get("meta", {}).get("summary", {}).get("top_issues", [])]) or "  - (none)"
        quick_wins = "\n".join([f"  - {t}" for t in data.get("meta", {}).get("summary", {}).get("quick_wins", [])]) or "  - (none)"
        assumptions = "\n".join([f"  - {a}" for a in data.get("assumptions", [])]) or "  - (none)"

        def fmt_issue(iss):
            rec = iss.get("recommendation", {}) or {}
            ac = "\n".join([f"  - {c}" for c in rec.get("acceptance_criteria", [])]) or "  - (none)"
            return f"""### {iss.get('id','ISS')} — {iss.get('title','(title)')} (Severity: {iss.get('severity',0)})
- Heuristic: {iss.get('heuristic','')}
- Evidence: {iss.get('evidence','')}
- Recommendation: {rec.get('action','')}
- Acceptance criteria:
{ac}
"""

        findings_md = "\n".join([fmt_issue(i) for i in data.get("issues", [])]) or "(No issues parsed)"
        impact_effort_md = ", ".join([f"{ie.get('issue_id')} [{ie.get('impact')}/{ie.get('effort')}]" for ie in data.get("impact_effort", [])]) or "(n/a)"

        final_report = REPORT_TEMPLATE_MD.format(
            owner=owner,
            date=today,
            top_issues=top_issues,
            quick_wins=quick_wins,
            assumptions=assumptions,
            h1=H.get("H1",""), h2=H.get("H2",""), h3=H.get("H3",""), h4=H.get("H4",""),
            h5=H.get("H5",""), h6=H.get("H6",""), h7=H.get("H7",""), h8=H.get("H8",""),
            h9=H.get("H9",""), h10=H.get("H10",""),
            annotated_images="\n\n".join(annotated_images_md) or "(annotated images included in ZIP)",
            findings=findings_md,
            impact_effort=impact_effort_md,
            num_screens=len(local_images)
        )

        st.subheader("Report Preview")
        st.markdown(final_report)

        # Append compiled report to the ZIP and offer download
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report/report_compiled.md", final_report)

        st.download_button(
            "⬇️ Download report + images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="heuristic_review.zip"
        )

else:
    st.info("Upload at least one screenshot, then click ‘Run Heuristic Review’.")
