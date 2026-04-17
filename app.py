"""
MOLDIQ — MIM Injection Molding Design Decision Platform
=======================================================
Stage Flow:
  Mold Concept → 0.Feasibility Gate → 1.Flow Analysis
              → 2.Dimension Prediction → 3.Inverse Correction
"""

import streamlit as st
import os, sys, json, re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

# ── 경로 설정 ──────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ── 페이지 설정 (가장 먼저) ───────────────────────────────
st.set_page_config(
    page_title="MOLDIQ — Smart MIM Design System",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 모듈 임포트 ───────────────────────────────────────────
try:
    from core.i18n import TRANSLATIONS
    from core.rule_check import run_feasibility_check, MATERIAL_LIMITS
    from core.cae_analyzer import analyze_cae, load_cae_data, generate_sample_cae_csv, PROCESS_LIMITS
    from core.shrink_model import (
        predict_shrinkage_field, predict_part_dimensions,
        build_shrink_map_grid, get_sample_features, MATERIAL_PVT
    )
    from core.inverse_design import run_inverse_design, build_error_map
    from core.model_processor import process_uploaded_model
    from core.parting_line_analyzer import analyze_parting_line
    from core.slide_core_optimizer import optimize_mold_design
    from core.flow_csv_generator import generate_flow_csv_from_github
    try:
        from core.ml_feedback import train_or_update_model, apply_ml_correction
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
    try:
        from core.drawing_sync import load_drawing_features_from_csv, generate_cad_macro_script
    except ImportError:
        pass
    MODULES_OK = True
except ImportError as e:
    st.error(f"⚠️ Module load failed: {e}")
    st.stop()

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stage-tag {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; color: #00d4aa; text-transform: uppercase; margin-bottom: 4px;
}
.stage-title { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
.stage-desc  { font-size: 0.82rem; color: #8899aa; }
.verdict-pass { background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.verdict-warn { background: rgba(255,107,53,0.08); border: 1px solid rgba(255,107,53,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.verdict-fail { background: rgba(255,59,92,0.08); border: 1px solid rgba(255,59,92,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.mono { font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.info-box { background: #111318; border: 1px solid #252b36; border-radius: 6px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.8rem; color: #8899aa; }
.link-card {
    background: #111827; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; transition: border-color 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Session State 초기화
# ══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "stage0_done": False, "stage1_done": False,
        "stage2_done": False, "stage3_done": False,
        "feasibility_result": None,
        "cae_df": None, "cae_analysis": None,
        "shrink_df": None, "dim_df": None,
        "inverse_result": None,
        "material": "PC+ABS", "avg_thickness": 2.4,
        # Mold Concept → Feasibility 연동 데이터
        "stl_analysis": {
            "file_loaded": False, "geometry": None,
            "undercuts": None, "parting": None,
            "design": None, "pull_direction": "Z"
        },
        # Stage 1 GitHub 연동
        "github_sim_signal_id": None,
        "flow_csv_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════
with st.sidebar:
    # 언어 설정
    st.markdown("### 🌐 Language Settings")
    lang = st.selectbox("Select Language",
        ["English", "한국어", "Español", "日本語", "中文", "Français", "हिन्दी"],
        index=0, key="lang_select")
    T = TRANSLATIONS.get(lang, TRANSLATIONS["English"])

    st.markdown("## 🔩 MOLDIQ")
    st.markdown(f'<div class="mono" style="color:#55667a;font-size:0.65rem;">{T["platform_desc"]}</div>',
                unsafe_allow_html=True)
    st.divider()

    # 워크플로우 단계 선택
    stage_labels = {
        "🎯 Mold Concept (Main)": "mold_concept",
        T["stage0_label"]: "stage0",
        T["stage1_label"]: "stage1",
        T["stage2_label"]: "stage2",
        T["stage3_label"]: "stage3",
    }
    selected_stage = st.radio("workflow_stage", list(stage_labels.keys()), key="stage_select")
    current_stage = stage_labels[selected_stage]

    st.divider()

    # 전역 설정
    st.markdown(f"#### {T['global_settings']}")
    material = st.selectbox(T["material"], list(MATERIAL_LIMITS.keys()), index=0, key="material_select")
    st.session_state["material"] = material
    avg_thickness = st.number_input(T["avg_thickness"], 0.5, 10.0, 2.4, 0.1)
    st.session_state["avg_thickness"] = avg_thickness

    st.divider()

    # 진행 상태
    st.markdown(f"#### {T['status']}")
    for label, done_key in [
        (T["stage0_label"], "stage0_done"),
        (T["stage1_label"], "stage1_done"),
        (T["stage2_label"], "stage2_done"),
        (T["stage3_label"], "stage3_done"),
    ]:
        icon = "✅" if st.session_state[done_key] else "⬜"
        st.markdown(f"{icon} {label}")

    st.divider()

    # ML 피드백
    st.markdown(f"### {T['ml_feedback']}")
    real_csv = st.file_uploader(T["upload_actual"], type=["csv"])
    if real_csv and st.button(T.get("btn_retrain", "Retrain XGBoost Model")):
        if ML_AVAILABLE and "dim_df" in st.session_state and st.session_state["dim_df"] is not None:
            success, msg = train_or_update_model(real_csv, st.session_state["dim_df"])
            st.success(msg) if success else st.error(msg)
        else:
            st.warning(T.get("warn_run_st2_first", "Please run Stage 2 first."))

# ── 언어 객체 (사이드바 밖에서도 사용) ───────────────────
T = TRANSLATIONS.get(st.session_state.get("lang_select", "English"), TRANSLATIONS["English"])

# ══════════════════════════════════════════════════════════
#  공통 유틸
# ══════════════════════════════════════════════════════════
def verdict_color(v):
    if v in ("PASS", "OK"): return "🟢"
    if v in ("WARN", "Attention"): return "🟡"
    return "🔴"

def style_verdict_df(df, verdict_col="판정"):
    def color_row(row):
        v = row.get(verdict_col, "")
        if v in ("PASS", "OK"): return ["background-color: rgba(0,212,170,0.08)"] * len(row)
        if v in ("WARN", "주의", "OVER", "UNDER"): return ["background-color: rgba(255,107,53,0.08)"] * len(row)
        if v == "FAIL": return ["background-color: rgba(255,59,92,0.08)"] * len(row)
        return [""] * len(row)
    return df.style.apply(color_row, axis=1)


# ══════════════════════════════════════════════════════════
#  PHASE: Mold Concept (기본 페이지 — STL 분석)
# ══════════════════════════════════════════════════════════
if current_stage == "mold_concept":
    st.markdown('<div class="stage-tag">STAGE 0</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">🏭 Mold Design Feasibility</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-desc">STL Model Upload → Automated Mold Design Proposals</div>',
                unsafe_allow_html=True)
    st.info("📦 Upload an STL file to automatically generate proposals for parting lines, slides, and cores.")

    tab_upload, tab_geometry, tab_parting, tab_design, tab_summary = st.tabs([
        "📤 STL Upload", "📊 Geometry Analysis", "📍 Parting Line", "🔧 Slide & Core", "📋 Summary"
    ])

    analysis = st.session_state.stl_analysis

    # ── Tab 1: Upload ─────────────────────────────────────
    with tab_upload:
        st.markdown("## 📤 Upload STL File")
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Select STL File", type=["stl", "STL"])
        with col2:
            st.write(""); st.write("")
            st.markdown("**Pull Direction:**")
            pull_direction = st.selectbox("Pull Direction", ["Z", "X", "Y"],
                                          key="pull_dir_select", help="Z = Top-Bottom (standard)")
            st.session_state.stl_analysis["pull_direction"] = pull_direction

        if uploaded_file is not None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Upload successful: {uploaded_file.name}")

            if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("⏳ Analyzing model..."):
                    try:
                        model_result = process_uploaded_model(file_path, pull_direction)
                        if "error" in model_result:
                            st.error(f"❌ {model_result['error']}")
                        else:
                            parting_result = analyze_parting_line(model_result["geometry"], pull_direction)
                            design_result  = optimize_mold_design(
                                model_result["geometry"], model_result["undercut_regions"], parting_result
                            )
                            st.session_state.stl_analysis.update({
                                "file_loaded": True,
                                "geometry": model_result["geometry"],
                                "undercuts": model_result["undercut_regions"],
                                "parting": parting_result,
                                "design": design_result,
                            })
                            # ── Feasibility Gate 자동 연동 ──────────────────
                            geo = model_result["geometry"]
                            _bounds = geo.get("bounds", {})
                            _sx = _bounds.get("size_x", 0)
                            _sy = _bounds.get("size_y", 0)
                            _sz = _bounds.get("size_z", 0)
                            # flow_length ≈ 대각선 최대 길이
                            _flow_len = float(np.sqrt(_sx**2 + _sy**2 + _sz**2)) * 0.5
                            st.session_state["stl_derived"] = {
                                "min_thickness": round(_sz * 0.08, 2) if _sz > 0 else 1.8,
                                "max_thickness": round(_sz * 0.25, 2) if _sz > 0 else 3.2,
                                "flow_length": round(_flow_len, 1),
                                "part_volume": round(geo.get("volume_cm3", 12.5), 2),
                                "undercut": len(model_result["undercut_regions"]) > 0,
                            }
                            st.success("✅ Analysis Complete! Geometry parameters synced to Feasibility Gate.")
                            st.balloons()
                    except ImportError as e:
                        st.error(f"⚠️ Module not found: {e}")
                    except Exception as e:
                        st.error(f"❌ Analysis Error: {e}")

    # ── Tab 2: Geometry ───────────────────────────────────
    with tab_geometry:
        st.markdown("## 📊 Geometry Analysis")
        if analysis["file_loaded"] and analysis["geometry"]:
            geo = analysis["geometry"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔹 Volume",       f"{geo['volume_cm3']:.1f} cm³")
            c2.metric("🔹 Surface Area", f"{geo['surface_area_mm2']:.0f} mm²")
            c3.metric("🔹 Vertices",     f"{geo['vertices_count']:,}")
            c4.metric("🔹 Triangles",    f"{geo['triangles_count']:,}")
            st.divider()
            st.markdown("### 📦 Bounding Box")
            b = geo["bounds"]
            st.dataframe({
                "Axis": ["X","Y","Z"],
                "Min": [f"{b['min_x']:.2f} mm", f"{b['min_y']:.2f} mm", f"{b['min_z']:.2f} mm"],
                "Max": [f"{b['max_x']:.2f} mm", f"{b['max_y']:.2f} mm", f"{b['max_z']:.2f} mm"],
                "Size":[f"{b['size_x']:.2f} mm", f"{b['size_y']:.2f} mm", f"{b['size_z']:.2f} mm"],
            }, use_container_width=True, hide_index=True)
            st.markdown("### 🎯 Centroid")
            c = geo["centroid"]
            co1, co2, co3 = st.columns(3)
            co1.metric("X", f"{c['x']:.2f} mm")
            co2.metric("Y", f"{c['y']:.2f} mm")
            co3.metric("Z", f"{c['z']:.2f} mm")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 3: Parting Line ───────────────────────────────
    with tab_parting:
        st.markdown("## 📍 Parting Line Recommendation")
        if analysis["file_loaded"] and analysis["parting"]:
            parting = analysis["parting"]
            rec = parting["recommendation"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", rec["recommended_parting_line"].split("(")[0].strip())
            c2.metric("Complexity", rec["complexity"])
            c3.metric("Est. Depth", f"{rec['estimated_depth']:.1f} mm")
            c4.metric("Confidence", rec["confidence"].split("(")[0].strip())
            st.divider()
            st.markdown("### 📊 All Options Comparison")
            opts = parting["parting_analysis"]["all_options"]
            st.dataframe([{
                "Parting Line": o["name"],
                "Complexity": o["complexity_level"],
                "Score": f"{o['complexity_score']:.1f}",
                "Est. Area": f"{o['estimated_area_mm2']:.0f} mm²",
            } for o in opts], use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### ⚠️ Flash Risk")
            flash = parting["flash_risk_assessment"]
            emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.markdown(f"{emoji[flash['flash_risk_level']]} **{flash['flash_risk_level']}** (Score: {flash['flash_risk_score']})")
            st.write(f"**Recommendation:** {flash['recommendation']}")
            with st.expander("🛡️ Mitigation Measures"):
                for m in flash["mitigation_measures"]: st.write(f"• {m}")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 4: Slide & Core ───────────────────────────────
    with tab_design:
        st.markdown("## 🔧 Slide & Core Design")
        if analysis["file_loaded"] and analysis["design"]:
            design = analysis["design"]
            us = design["undercut_summary"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Undercut", "Detected" if us.get("has_undercut") else "None",
                      f"{us.get('total_undercut_regions', 0)} Regions")
            c2.metric("Slide", "Required" if us["slide_required"] else "Not Required")
            c3.metric("Core",  "Required" if us["core_required"]  else "Not Required")
            st.divider()
            if design["slide_design"]["count"] > 0:
                st.markdown(f"### 🎯 Slides ({design['slide_design']['count']} units)")
                st.dataframe([{
                    "ID": s["slide_id"], "Severity": s["severity"],
                    "Thickness": f"{s['estimated_dimensions']['thickness_mm']} mm",
                    "Width":     f"{s['estimated_dimensions']['width_mm']} mm",
                    "Length":    f"{s['estimated_dimensions']['length_mm']} mm",
                    "Material":  s["material_grade"],
                    "Cost":      f"${int(s['estimated_cost_usd'])}",
                } for s in design["slide_design"]["slides"]], use_container_width=True, hide_index=True)
            else:
                st.info("✓ No slides required.")
            st.divider()
            if design["core_design"]["count"] > 0:
                st.markdown(f"### 🔨 Cores ({design['core_design']['count']} units)")
                st.dataframe([{
                    "ID": c["core_id"], "Type": c["type"],
                    "Diameter": f"{c['estimated_dimensions']['diameter_mm']} mm",
                    "Length":   f"{c['estimated_dimensions']['length_mm']} mm",
                    "Material": c["material_grade"],
                    "Cost":     f"${int(c['estimated_cost_usd'])}",
                } for c in design["core_design"]["cores"]], use_container_width=True, hide_index=True)
            else:
                st.info("✓ No additional cores required.")
        else:
            st.info("💡 Please upload and analyze an STL file first.")

    # ── Tab 5: Summary ────────────────────────────────────
    with tab_summary:
        st.markdown("## 📋 Complete Design Summary")
        if analysis["file_loaded"] and analysis["design"]:
            design = analysis["design"]
            for rec in design["recommendations"]: st.write(f"• {rec}")
            st.divider()
            view_mode = st.radio("Display", ["Analysis Summary", "Export Data", "Next Steps"],
                                 horizontal=True, key="summary_view_mode")
            if view_mode == "Analysis Summary":
                cx = design["complexity_assessment"]
                cost = design["cost_estimate"]
                lead = design["lead_time"]
                co1, co2 = st.columns(2)
                with co1:
                    st.markdown("#### 🏗️ Design Complexity")
                    st.metric("Level", cx["level"])
                    st.metric("Mechanisms", f"{cx['total_mechanisms']} units")
                    st.metric("Score", f"{cx['score']}/4")
                with co2:
                    st.markdown("#### 💰 Cost & Lead Time")
                    st.metric("Total Cost", f"${int(cost['total_mechanism_cost_usd'])}")
                    st.metric("Impact", f"{cost['percentage_of_mold']}%")
                    st.metric("Critical Path", f"{lead['critical_path_days']} Days")
            elif view_mode == "Export Data":
                import json as _json
                json_str = _json.dumps(design, indent=2, default=str)
                st.download_button("📄 Download JSON", json_str, "mold_design_report.json", "application/json",
                                   use_container_width=True)
            else:
                st.info("1. Review results with the design team")
                st.info("2. Proceed to **0. Feasibility Gate** → parameters are auto-filled from STL analysis")
                st.info("3. Proceed to **1. Flow Analysis** → link to MIM-Ops simulation")
        else:
            st.info("💡 Please upload and analyze an STL file first (Tab 1).")


# ══════════════════════════════════════════════════════════
#  STAGE 0: Feasibility Gate
# ══════════════════════════════════════════════════════════
elif current_stage == "stage0":
    st.markdown('<div class="stage-tag">STAGE 0 · FEASIBILITY GATE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T["st0_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">{T["st0_desc"]}</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── STL 분석 결과 자동 연동 안내 ─────────────────────
    derived = st.session_state.get("stl_derived")
    if derived:
        st.success("✅ Geometry parameters auto-filled from Mold Concept STL analysis. You can adjust if needed.")
    else:
        st.info("💡 Tip: Run STL analysis in **Mold Concept** tab first to auto-fill geometry parameters.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### {T['input_geometry']}")
        t_min  = st.number_input(T["min_thick"],   0.5, 10.0,
                                 value=float(derived["min_thickness"]) if derived else 1.8, step=0.1)
        t_max  = st.number_input(T["max_thick"],   0.5, 20.0,
                                 value=float(derived["max_thickness"]) if derived else 3.2, step=0.1)
        flow_l = st.number_input(T["flow_length"], 10.0, 500.0,
                                 value=float(derived["flow_length"])   if derived else 148.0, step=1.0)
        draft  = st.number_input(T["draft_angle"], 0.0, 10.0, 1.5, 0.25)

    with col_b:
        st.markdown(f"#### {T['input_mold']}")
        # gate_count: 기본값 1, 수동 수정 가능
        gate_count = st.number_input(T["gate_count"], 1, 8, 1, 1)
        # undercut: STL에서 자동 반영
        undercut   = st.checkbox(T["has_undercut"],
                                 value=bool(derived["undercut"]) if derived else False)
        part_vol   = st.number_input(T["part_volume"], 0.1, 500.0,
                                     value=float(derived["part_volume"]) if derived else 12.5, step=0.5)

    st.markdown("")
    if st.button(T["btn_st0"], type="primary", use_container_width=True):
        params = {
            "material": material,
            "min_thickness": t_min, "max_thickness": t_max,
            "avg_thickness": avg_thickness,
            "flow_length": flow_l, "draft_angle": draft,
            "gate_count": gate_count, "undercut": undercut,
            "part_volume": part_vol,
        }
        with st.spinner(T.get("st0_checking", "Analyzing...")):
            result = run_feasibility_check(params)
        st.session_state["feasibility_result"] = result
        st.session_state["stage0_done"] = True

    if st.session_state["feasibility_result"]:
        res     = st.session_state["feasibility_result"]
        overall = res["overall"]
        summary = res["summary"]

        st.markdown("---")
        vc  = {"PASS": "verdict-pass", "WARN": "verdict-warn", "FAIL": "verdict-fail"}[overall]
        vic = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[overall]
        st.markdown(f"""
        <div class="{vc}">
            <span style="font-size:1.4rem;">{vic}</span>
            <strong style="font-size:1.0rem; margin-left:8px;">{overall} — {summary['verdict_text']}</strong><br>
            <span style="font-size:0.8rem; color:#8899aa; margin-left:32px;">
            ✅ {summary['PASS']} Passed &nbsp; ⚠ {summary['WARN']} Warnings &nbsp; ❌ {summary['FAIL']} Failed
            </span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("L/t Ratio",          f"{res['lt_ratio']:.1f}")
        m2.metric("Thickness Ratio",    f"{res['thickness_ratio']:.2f}×")
        m3.metric("Est. Shrinkage",
                  f"{res['shrink_range_pct'][0]:.1f}~{res['shrink_range_pct'][1]:.1f}%")

        st.markdown("#### 📋 Detailed Results by Item")
        df_items = pd.DataFrame(res["items"])
        rename_map = {"Item":"항목","Value":"측정값","Reference":"기준값","Verdict":"판정","Action":"권장 조치"}
        df_items = df_items.rename(columns=rename_map)
        if "판정" in df_items.columns:
            df_items.insert(0, "", df_items["판정"].apply(verdict_color))
        display_cols = [c for c in ["","항목","측정값","기준값","판정","권장 조치"] if c in df_items.columns]
        st.dataframe(df_items[display_cols], use_container_width=True, hide_index=True)

        if overall != "FAIL":
            st.success("✅ Manufacturability Confirmed")
            st.divider()
            # ── Flow Analysis 링크 카드 ─────────────────
            st.markdown("#### 🌊 Flow Analysis — Next Step")
            st.markdown("""
            <div class="link-card">
            <strong>🚀 MIM-Ops Pro: OpenFOAM Cloud Simulation</strong><br>
            <span style="color:#8899aa; font-size:0.85rem;">
            Run injection flow simulation on the cloud (GitHub Actions).
            After simulation, return here and load the results in <b>1. Flow Analysis</b>.
            </span>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(
                "🔗 Open MIM-Ops Simulation →",
                "https://openfoam-injection-automation.streamlit.app/",
                use_container_width=True,
            )
            st.info("📌 After simulation: go to **1. Flow Analysis** sidebar → enter your Signal ID to load results.")
        else:
            st.error("❌ Design Revision Required before proceeding.")


# ══════════════════════════════════════════════════════════
#  STAGE 1: Flow Analysis
# ══════════════════════════════════════════════════════════
elif current_stage == "stage1":
    st.markdown('<div class="stage-tag">STAGE 1 · FLOW ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T["st1_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">{T["st1_desc"]}</div>', unsafe_allow_html=True)
    st.markdown("")

    tab_import, tab_field, tab_defect, tab_window = st.tabs([
        T["tab_data"], T["tab_field"], T["tab_defect"], T["tab_window"]
    ])

    with tab_import:
        st.markdown("#### 📂 Load Flow Analysis Results")

        # ── Option A: GitHub Signal ID로 자동 CSV 생성 ─────────────
        with st.expander("🔗 Option A — Load from MIM-Ops Simulation (GitHub)", expanded=True):
            st.markdown("""
            <div class="link-card">
            <strong>Step 1:</strong> Run simulation at 
            <a href="https://openfoam-injection-automation.streamlit.app/" target="_blank">
            🚀 MIM-Ops Pro</a><br>
            <strong>Step 2:</strong> Enter your Signal ID below to load results automatically.
            </div>
            """, unsafe_allow_html=True)

            sig_col1, sig_col2 = st.columns([3, 1])
            with sig_col1:
                signal_id = st.text_input(
                    "Signal ID (from MIM-Ops simulation)",
                    value=st.session_state.get("github_sim_signal_id", ""),
                    placeholder="e.g. e2d394fe",
                    help="Found in simulation results folder name: simulation-{signal_id}"
                )
            with sig_col2:
                st.write("")
                st.write("")
                if st.button("📥 Generate CSV", use_container_width=True, type="primary"):
                    if signal_id.strip():
                        with st.spinner("Fetching from GitHub..."):
                            try:
                                cae_df = generate_flow_csv_from_github(signal_id.strip())
                                st.session_state["cae_df"] = cae_df
                                st.session_state["github_sim_signal_id"] = signal_id.strip()
                                st.session_state["flow_csv_ready"] = True
                                st.success(f"✅ Flow data loaded from simulation-{signal_id.strip()}")
                            except Exception as e:
                                st.error(f"❌ {e}")
                    else:
                        st.warning("Please enter a Signal ID.")

            if st.session_state.get("flow_csv_ready") and st.session_state.get("cae_df") is not None:
                df_preview = st.session_state["cae_df"]
                st.dataframe(df_preview.head(5), use_container_width=True)
                csv_bytes = df_preview.to_csv(index=False).encode("utf-8-sig")
                st.download_button("💾 Download CSV", csv_bytes, "flow_analysis.csv", "text/csv",
                                   use_container_width=True)

        # ── Option B: 수동 CSV 업로드 ──────────────────────────────
        with st.expander("📄 Option B — Manual CSV Upload"):
            st.markdown(f"""
            <div class="info-box">
            <strong>{T['required_cols']}</strong><br>
            <span class="mono">x, y, pressure(MPa), temperature(°C), fill_time(s)</span><br>
            OpenFOAM → postProcess → CSV export / Moldflow export
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader(T.get("select_cae_file", "Select CAE CSV File"), type=["csv"])
            use_sample = st.checkbox(T["use_sample"], value=False)

        use_ml = st.toggle(T["apply_ml"], value=False)

        if st.button(T["btn_st1"], type="primary", use_container_width=True):
            with st.spinner(T.get("st1_analyzing", "Analyzing...")):
                # ── 데이터 소스 우선순위: GitHub > 수동 업로드 > 샘플 ──
                cae_df = st.session_state.get("cae_df")
                if cae_df is None:
                    if uploaded and not use_sample:
                        try:
                            cae_df = load_cae_data(uploaded)
                        except Exception as e:
                            st.error(f"❌ CSV 로드 실패: {e}")
                            st.stop()
                    else:
                        cae_df = generate_sample_cae_csv(n_points=300, material=material)
                        st.info(f"📌 {T.get('msg_using_sample', 'Using sample data.')}")

                # ── 필수 컬럼 체크 ──────────────────────────────────
                required_cols = {"x", "y", "pressure", "temperature", "fill_time"}
                missing = required_cols - set(cae_df.columns)
                if missing:
                    st.error(f"❌ CSV에 필수 컬럼이 없습니다: {missing}")
                    st.stop()

                if use_ml:
                    st.toast(T["toast_ml_analyzing"], icon="🤖")

                try:
                    analysis = analyze_cae(cae_df, material=material)
                except Exception as e:
                    import traceback as _tb
                    st.error(f"❌ analyze_cae 실패: {e}")
                    st.code(_tb.format_exc(), language="python")
                    st.stop()

                st.session_state["cae_df"]      = cae_df
                st.session_state["cae_analysis"] = analysis
                st.session_state["stage1_done"]  = True
                st.success(T["msg_analysis_done"])

    # ── 결과 탭들 ─────────────────────────────────────────
    if st.session_state["cae_analysis"]:
        analysis = st.session_state["cae_analysis"]
        cae_df   = st.session_state["cae_df"]
        stats    = analysis["stats"]

        # ── Tab: Field Map (개선된 3D 유동 시각화) ──────────
        with tab_field:
            st.markdown(f"#### 📊 {T.get('title_field_map', 'Field Distribution Map')}")

            has_z = "z" in cae_df.columns

            # ── 게이트 위치 탐지 (fill_time 최솟값 = 절대 앵커) ──
            gate_idx = cae_df["fill_time"].idxmin()
            gate_x   = float(cae_df.loc[gate_idx, "x"])
            gate_y   = float(cae_df.loc[gate_idx, "y"])
            gate_z   = float(cae_df.loc[gate_idx, "z"]) if has_z else 0.0

            # ── 게이트 상대 거리 계산 ────────────────────────────
            df_v = cae_df.copy()
            if has_z:
                df_v["gate_dist"] = np.sqrt(
                    (df_v["x"] - gate_x)**2 +
                    (df_v["y"] - gate_y)**2 +
                    (df_v["z"] - gate_z)**2
                )
            else:
                df_v["gate_dist"] = np.sqrt(
                    (df_v["x"] - gate_x)**2 +
                    (df_v["y"] - gate_y)**2
                )

            fill_max = float(df_v["fill_time"].max())
            fill_min = float(df_v["fill_time"].min())

            # ── 공통 레이아웃 설정 ───────────────────────────────
            _scene = dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)",
                zaxis_title="Z (mm)" if has_z else "",
                bgcolor="#111318",
                xaxis=dict(color="#e2e8f0", showbackground=True,
                            backgroundcolor="#0d1117"),
                yaxis=dict(color="#e2e8f0", showbackground=True,
                            backgroundcolor="#0d1117"),
                zaxis=dict(color="#e2e8f0", showbackground=True,
                            backgroundcolor="#0d1117"),
                aspectmode="data",
            )
            _layout_base = dict(
                paper_bgcolor="#0a0c0f", font_color="#e2e8f0",
                height=520, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
            )

            # ── 게이트 마커 (공통) ───────────────────────────────
            def _gate_marker(z_val):
                return go.Scatter3d(
                    x=[gate_x], y=[gate_y], z=[z_val],
                    mode="markers+text",
                    name="🔴 Gate",
                    text=["GATE"],
                    textposition="top center",
                    textfont=dict(color="#ff4444", size=11),
                    marker=dict(size=14, color="#ff2222",
                                symbol="diamond", opacity=1.0,
                                line=dict(color="#ffffff", width=2)),
                    showlegend=True,
                )

            # ── 외곽 경계 박스 (파트 형상 힌트) ─────────────────
            def _bbox_traces(df):
                xs = [df["x"].min(), df["x"].max()]
                ys = [df["y"].min(), df["y"].max()]
                zs = ([df["z"].min(), df["z"].max()]
                      if has_z else [0, 0])
                corners_x, corners_y, corners_z = [], [], []
                for xi in xs:
                    for yi in ys:
                        for zi in zs:
                            corners_x.append(xi); corners_y.append(yi)
                            corners_z.append(zi)
                return go.Scatter3d(
                    x=corners_x, y=corners_y, z=corners_z,
                    mode="markers",
                    marker=dict(size=2, color="rgba(255,255,255,0.05)"),
                    showlegend=False, hoverinfo="skip",
                )

            field_tabs = st.tabs([
                "🟢 Fill Time (Flow Front)",
                "🔵 Pressure (MPa)",
                "🔴 Temperature (°C)",
            ])

            # ════════════════════════════════════════════════
            # TAB 1: Fill Time → 유동 선단 애니메이션
            # ════════════════════════════════════════════════
            with field_tabs[0]:
                st.markdown("""
                <div style="background:#0d1117;border:1px solid #1e3a2f;border-radius:6px;
                    padding:8px 14px;margin-bottom:10px;font-size:0.8rem;color:#55aa88;">
                ▶ 유동 선단(Flow Front) 시각화 — 게이트(🔴)에서 충전이 시작되어 시간 순서대로 채워집니다.
                슬라이더로 충전 시간을 조절하거나 Play 버튼으로 애니메이션을 실행하세요.
                </div>
                """, unsafe_allow_html=True)

                # 프레임 수 및 시간 단계 생성
                N_FRAMES = 20
                t_steps  = np.linspace(fill_min, fill_max, N_FRAMES)

                frames = []
                for t_val in t_steps:
                    mask  = df_v["fill_time"] <= t_val
                    ratio = (t_val - fill_min) / (fill_max - fill_min + 1e-9)
                    filled_pts = df_v[mask]
                    front_mask = (df_v["fill_time"] > t_val - fill_max / N_FRAMES) & mask
                    front_pts  = df_v[front_mask]

                    frame_data = [
                        # 이미 채워진 영역 (반투명)
                        go.Scatter3d(
                            x=filled_pts["x"], y=filled_pts["y"],
                            z=filled_pts["z"] if has_z else [0]*len(filled_pts),
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=filled_pts["fill_time"],
                                colorscale="Viridis",
                                cmin=fill_min, cmax=fill_max,
                                opacity=0.6,
                                showscale=False,
                            ),
                            name="Filled", showlegend=False, hoverinfo="skip",
                        ),
                        # 현재 유동 선단 (강조)
                        go.Scatter3d(
                            x=front_pts["x"], y=front_pts["y"],
                            z=front_pts["z"] if has_z else [0]*len(front_pts),
                            mode="markers",
                            marker=dict(
                                size=8,
                                color="#00ffcc",
                                opacity=1.0,
                                showscale=False,
                            ),
                            name="Flow Front", showlegend=False, hoverinfo="skip",
                        ),
                        # 게이트
                        _gate_marker(gate_z),
                    ]
                    frames.append(go.Frame(data=frame_data,
                                           name=f"{t_val:.2f}"))

                # 초기 상태: 게이트 근방만 표시
                init_mask = df_v["fill_time"] <= t_steps[0]
                init_pts  = df_v[init_mask] if init_mask.any() else df_v.head(1)

                fig_flow = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=init_pts["x"], y=init_pts["y"],
                            z=init_pts["z"] if has_z else [0]*len(init_pts),
                            mode="markers",
                            marker=dict(size=5, color="#00d4aa", opacity=0.7),
                            name="Filled", showlegend=False,
                        ),
                        go.Scatter3d(x=[], y=[], z=[],
                                     mode="markers", name="Flow Front",
                                     marker=dict(size=8, color="#00ffcc"),
                                     showlegend=False),
                        _gate_marker(gate_z),
                        _bbox_traces(df_v),
                    ],
                    frames=frames,
                )

                sliders = [dict(
                    active=0,
                    steps=[dict(
                        args=[[f.name],
                              dict(frame=dict(duration=80, redraw=True),
                                   mode="immediate",
                                   transition=dict(duration=0))],
                        label=f"{float(f.name):.1f}s",
                        method="animate",
                    ) for f in frames],
                    x=0.05, xanchor="left",
                    y=-0.02, yanchor="top",
                    len=0.9,
                    bgcolor="#1e3a2f",
                    activebgcolor="#00d4aa",
                    font=dict(color="#e2e8f0", size=9),
                    currentvalue=dict(
                        prefix="Fill Time: ", suffix=" s",
                        font=dict(color="#00d4aa"),
                        visible=True, xanchor="center",
                    ),
                    pad=dict(t=30),
                )]

                updatemenus = [dict(
                    type="buttons", showactive=False,
                    x=0.01, xanchor="left",
                    y=0.02, yanchor="bottom",
                    bgcolor="#111318", bordercolor="#00d4aa",
                    font=dict(color="#e2e8f0"),
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=80, redraw=True),
                                              fromcurrent=True,
                                              mode="immediate",
                                              transition=dict(duration=0))]),
                        dict(label="⏸ Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate",
                                                transition=dict(duration=0))]),
                    ],
                )]

                fig_flow.update_layout(
                    **_layout_base,
                    scene=_scene,
                    title=dict(text="Flow Front Propagation (Gate → End)",
                               font=dict(color="#00d4aa")),
                    sliders=sliders,
                    updatemenus=updatemenus,
                )
                st.plotly_chart(fig_flow, use_container_width=True)

                # 게이트 정보 표시
                gm1, gm2, gm3, gm4 = st.columns(4)
                gm1.metric("🔴 Gate X", f"{gate_x:.3f} mm")
                gm2.metric("🔴 Gate Y", f"{gate_y:.3f} mm")
                if has_z:
                    gm3.metric("🔴 Gate Z", f"{gate_z:.3f} mm")
                gm4.metric("Total Fill Time", f"{fill_max:.2f} s")

            # ════════════════════════════════════════════════
            # TAB 2: Pressure (MPa)
            # ════════════════════════════════════════════════
            with field_tabs[1]:
                st.markdown("""
                <div style="background:#0d1117;border:1px solid #1e2a3f;border-radius:6px;
                    padding:8px 14px;margin-bottom:10px;font-size:0.8rem;color:#5588aa;">
                🔵 압력 분포 — 게이트(🔴) 근처가 최고압, 끝단으로 갈수록 낮아집니다.
                점 크기는 게이트까지의 거리에 반비례 (가까울수록 크게 표시).
                </div>
                """, unsafe_allow_html=True)

                # 게이트 거리 역수로 점 크기 가중
                dist_norm = (df_v["gate_dist"] - df_v["gate_dist"].min()) / \
                            (df_v["gate_dist"].max() - df_v["gate_dist"].min() + 1e-9)
                pt_size_p = (3 + 8 * (1 - dist_norm)).clip(3, 11)

                fig_p = go.Figure(data=[
                    go.Scatter3d(
                        x=df_v["x"], y=df_v["y"],
                        z=df_v["z"] if has_z else [0]*len(df_v),
                        mode="markers",
                        marker=dict(
                            size=pt_size_p,
                            color=df_v["pressure"],
                            colorscale="Blues",
                            colorbar=dict(title="Pressure<br>(MPa)",
                                          tickfont=dict(color="#e2e8f0"),
                                          titlefont=dict(color="#e2e8f0")),
                            cmin=df_v["pressure"].min(),
                            cmax=df_v["pressure"].max(),
                            opacity=0.85,
                            line=dict(width=0),
                        ),
                        customdata=np.stack([df_v["pressure"],
                                              df_v["gate_dist"]], axis=-1),
                        hovertemplate=(
                            "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>"
                            "Pressure: %{customdata[0]:.1f} MPa<br>"
                            "Gate Distance: %{customdata[1]:.2f} mm<extra></extra>"
                        ),
                        name="Pressure Field",
                    ),
                    _gate_marker(gate_z),
                    _bbox_traces(df_v),
                ])
                fig_p.update_layout(
                    **_layout_base, scene=_scene,
                    title=dict(text="Pressure Distribution (MPa)",
                               font=dict(color="#4499dd")),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # ════════════════════════════════════════════════
            # TAB 3: Temperature (°C)
            # ════════════════════════════════════════════════
            with field_tabs[2]:
                st.markdown("""
                <div style="background:#0d1117;border:1px solid #3f1e1e;border-radius:6px;
                    padding:8px 14px;margin-bottom:10px;font-size:0.8rem;color:#aa5555;">
                🔴 온도 분포 — 게이트 근처 용융 온도가 가장 높고 유동 선단에서 냉각됩니다.
                웰드라인 위험 구간(저온 영역)을 주의깊게 확인하세요.
                </div>
                """, unsafe_allow_html=True)

                # 온도별 투명도 (고온 = 불투명, 저온 = 반투명)
                temp_norm = (df_v["temperature"] - df_v["temperature"].min()) / \
                            (df_v["temperature"].max() - df_v["temperature"].min() + 1e-9)
                opacity_t = (0.4 + 0.6 * temp_norm).clip(0.3, 1.0)

                fig_t = go.Figure(data=[
                    go.Scatter3d(
                        x=df_v["x"], y=df_v["y"],
                        z=df_v["z"] if has_z else [0]*len(df_v),
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=df_v["temperature"],
                            colorscale="Hot",
                            colorbar=dict(title="Temp (°C)",
                                          tickfont=dict(color="#e2e8f0"),
                                          titlefont=dict(color="#e2e8f0")),
                            opacity=0.85,
                            line=dict(width=0),
                        ),
                        customdata=np.stack([df_v["temperature"],
                                              df_v["fill_time"]], axis=-1),
                        hovertemplate=(
                            "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>"
                            "Temp: %{customdata[0]:.1f} °C<br>"
                            "Fill Time: %{customdata[1]:.2f} s<extra></extra>"
                        ),
                        name="Temperature Field",
                    ),
                    _gate_marker(gate_z),
                    _bbox_traces(df_v),
                ])
                fig_t.update_layout(
                    **_layout_base, scene=_scene,
                    title=dict(text="Temperature Distribution (°C)",
                               font=dict(color="#dd4444")),
                )
                st.plotly_chart(fig_t, use_container_width=True)

            # ── 통계 지표 ──────────────────────────────────
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T.get("metric_max_pressure",  "Max Pressure"), f"{stats['max_pressure_MPa']:.1f} MPa")
            c2.metric(T.get("metric_max_temp",       "Max Temp"),     f"{stats['max_temperature_C']:.1f} °C")
            c3.metric(T.get("metric_fill_time",      "Fill Time"),    f"{stats['fill_time_s']:.2f} s")
            c4.metric(T.get("metric_pressure_grad",  "Pres. Grad."), f"{stats['pressure_gradient']:.2f} MPa/mm")

        # ── Tab: Defect Analysis ──────────────────────────
        with tab_defect:
            st.markdown(f"#### ⚠️ {T['title_defect_risk']}")
            risks = analysis["defect_risks"]
            defect_names = {
                "short_shot": T["def_short_shot"], "weld_line": T["def_weld_line"],
                "sink_mark": T["def_sink_mark"],   "warpage":   T["def_warpage"],
                "air_trap":  T["def_air_trap"],
            }
            cols = st.columns(len(risks))
            for i, (key, risk) in enumerate(risks.items()):
                score = risk["score"]; level = risk["level"]
                icon = {"LOW": "🟢", "MED": "🟡", "HIGH": "🔴"}[level]
                cols[i].metric(defect_names.get(key, key), f"{icon} {level}", f"{risk['score']*100:.0f}%")

            for key, risk in risks.items():
                if risk["level"] != "LOW":
                    cls = "verdict-warn" if risk["level"] == "MED" else "verdict-fail"
                    st.markdown(f"""
                    <div class="{cls}">
                    <strong>{defect_names.get(key, key)}</strong><br>
                    <span style="font-size:0.82rem;color:#8899aa;">{risk['detail']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab: Process Window ───────────────────────────
        with tab_window:
            st.markdown(f"#### 🎯 {T['tab_window_title']}")
            optimal = analysis["optimal_conditions"]
            param_labels = {
                "melt_temperature": T["param_melt_temp"],
                "mold_temperature": T["param_mold_temp"],
                "injection_speed":  T["param_inj_speed"],
                "packing_pressure": T["param_pack_pres"],
                "packing_time":     T["param_pack_time"],
            }
            for param, vals in optimal.items():
                label = param_labels.get(param, param)
                lo, hi = vals["range"]; opt = vals["optimal"]; unit = vals["unit"]
                ratio = (opt - lo) / (hi - lo) if hi != lo else 0.5
                st.markdown(f"**{label}**")
                ca, cb = st.columns([3, 1])
                with ca:
                    st.progress(int(ratio * 100))
                    st.caption(f"Range: {lo}~{hi} {unit}  |  Optimal: {opt} {unit}")
                with cb:
                    st.markdown(f"""
                    <div class="mono" style="text-align:center;color:#00d4aa;font-size:1.1rem;font-weight:700;">
                    {opt} {unit}
                    </div>""", unsafe_allow_html=True)
                st.markdown("")
            st.success(f"✅ {T.get('msg_ready_for_st2', 'Ready for Stage 2 Dimension Prediction')}")
    else:
        for tab in [tab_field, tab_defect, tab_window]:
            with tab:
                st.info("💡 Run analysis in **Data Input** tab first.")


# ══════════════════════════════════════════════════════════
#  STAGE 2: Dimension Prediction
# ══════════════════════════════════════════════════════════
elif current_stage == "stage2":
    st.markdown('<div class="stage-tag">STAGE 2 · DIMENSION PREDICTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-title">Dimension Prediction & Quality Analysis</div>', unsafe_allow_html=True)

    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf1:
        st.markdown("##### Example Tolerance Standards")
        default_tol_data = {
            "Range (mm)": ["1~4","4~10","10~20","20~30","> 30"],
            "Example Tolerance (±)": [0.030, 0.050, 0.060, 0.080, 0.100]
        }
        edited_tol_df = st.data_editor(pd.DataFrame(default_tol_data), num_rows="fixed",
                                        use_container_width=True, key="tol_editor")
    with col_conf2:
        st.markdown("##### Mold Expansion Factor")
        expansion_factor = st.number_input("Factor (k)", 1.000, 1.300, 1.000, 0.001, format="%.3f")
        if "last_k" not in st.session_state:
            st.session_state.last_k = expansion_factor
        if st.session_state.last_k != expansion_factor:
            df = st.session_state.get("input_df", pd.DataFrame())
            for idx in df.index:
                if df.at[idx, "L_nominal"] > 0:
                    df.at[idx, "L_mold"] = round(df.at[idx, "L_nominal"] * expansion_factor, 3)
            st.session_state.input_df = df
            st.session_state.last_k = expansion_factor
            st.rerun()

    if "input_df" not in st.session_state or len(st.session_state.input_df) > 5:
        st.session_state.input_df = pd.DataFrame([
            {"Name": f"Part_{i+1}", "L_nominal": 0.0, "L_mold": 0.0, "tolerance": 0.0}
            for i in range(3)
        ])

    def sync_on_edit():
        state = st.session_state["dim_editor"]
        df = st.session_state.input_df
        for row_idx, changes in state["edited_rows"].items():
            idx = int(row_idx)
            for key, val in changes.items():
                try:
                    curr_val = float(val)
                    df.at[idx, key] = curr_val
                    if key == "L_nominal" and curr_val > 0:
                        df.at[idx, "L_mold"] = round(curr_val * expansion_factor, 3)
                        tols = edited_tol_df["Example Tolerance (±)"].values
                        if curr_val < 4: df.at[idx, "tolerance"] = tols[0]
                        elif curr_val < 10: df.at[idx, "tolerance"] = tols[1]
                        elif curr_val < 20: df.at[idx, "tolerance"] = tols[2]
                        elif curr_val <= 30: df.at[idx, "tolerance"] = tols[3]
                        else: df.at[idx, "tolerance"] = tols[4]
                except: pass
        st.session_state.input_df = df

    st.markdown("---")
    st.markdown("### 📝 Input Dimensions")
    edited_df = st.data_editor(
        st.session_state.input_df, num_rows="dynamic", use_container_width=True,
        key="dim_editor", on_change=sync_on_edit,
        column_config={
            "L_nominal": st.column_config.NumberColumn("Target (mm)",   format="%.3f", step=0.001),
            "L_mold":    st.column_config.NumberColumn("Mold Dim (mm)", format="%.3f", step=0.001),
            "tolerance": st.column_config.NumberColumn("Tolerance (±)", format="%.3f", step=0.001),
        }
    )
    st.session_state.input_df = edited_df

    if st.button("Run Prediction", type="primary", use_container_width=True):
        active_df = edited_df[edited_df["L_nominal"] > 0].copy()
        if not active_df.empty:
            with st.spinner("Analyzing with CAE Results..."):
                cae_df = st.session_state.get("cae_df")
                if cae_df is None:
                    cae_df = generate_sample_cae_csv(n_points=300, material=material)
                shrink_df = predict_shrinkage_field(cae_df, material=material, avg_thickness=avg_thickness)
                results = []
                for _, row in active_df.iterrows():
                    local_shrink = shrink_df["shrinkage_pct"].sample(1).values[0] / 100.0
                    l_final = round(row["L_mold"] * (1 - local_shrink), 3)
                    dev = round(l_final - row["L_nominal"], 3)
                    status = "OK" if abs(dev) <= row["tolerance"] else ("OVER" if dev > 0 else "UNDER")
                    results.append({
                        "Dimension Name": row["Name"],
                        "Target (L_nom)": row["L_nominal"],
                        "Mold (L_mold)": row["L_mold"],
                        "Predicted (L_final)": l_final,
                        "Deviation": dev,
                        "Tolerance": row["tolerance"],
                        "Status": status,
                    })
                st.session_state["dim_results"] = pd.DataFrame(results)
                st.session_state["stage2_done"] = True

    if st.session_state.get("dim_results") is not None:
        res = st.session_state["dim_results"]
        st.dataframe(res.style.map(
            lambda x: "background-color: rgba(0,212,170,0.2)" if x == "OK"
                      else "background-color: rgba(255,75,75,0.2)",
            subset=["Status"]
        ), use_container_width=True)
        csv = res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Download Analysis Report", csv, "MIM_Prediction.csv", "text/csv",
                           use_container_width=True)


# ══════════════════════════════════════════════════════════
#  STAGE 3: Inverse Correction
# ══════════════════════════════════════════════════════════
elif current_stage == "stage3":
    st.markdown('<div class="stage-tag">STAGE 3 · INVERSE CORRECTION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stage-title">{T.get("st3_title","Tolerance-based Inverse Correction")}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="stage-desc">E(x) = L_target - L_predicted → Linear correction within tolerance</div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""
    <div class="info-box">
    ❌ Exclude non-linear/asymmetric &nbsp;|&nbsp; ❌ No global scale changes<br>
    ✅ Uniform offset &nbsp;|&nbsp; ✅ Local linear correction &nbsp;|&nbsp; ✅ Feature-level adjustment
    </div>
    """, unsafe_allow_html=True)

    if st.button(T.get("btn_st3", "▶ Run Phase 3 Inverse Calculation"),
                 type="primary", use_container_width=True):
        with st.spinner(T.get("st3_calculating", "Calculating...")):
            dim_df   = st.session_state.get("dim_df")
            shrink_df = st.session_state.get("shrink_df")
            if dim_df is None:
                features = get_sample_features(material=material)
                dim_df = predict_part_dimensions(features, material=material)
            pvt = MATERIAL_PVT.get(material, MATERIAL_PVT["PC+ABS"])
            global_shrink = float(shrink_df["shrinkage"].mean()) if shrink_df is not None else pvt["base_shrink"]
            inverse_result = run_inverse_design(dim_df, global_shrink_avg=global_shrink)
            st.session_state["inverse_result"] = inverse_result
            st.session_state["stage3_done"] = True

    if st.session_state["inverse_result"]:
        inv = st.session_state["inverse_result"]
        corrections = inv["corrections"]
        summary = inv["summary"]
        cost = inv["cost_estimate"]

        is_pass = summary.get("HIGH", 0) == 0
        vc = "verdict-pass" if is_pass else "verdict-warn"
        vt = T.get("verdict_ok","PASS") if is_pass else T.get("verdict_check","REVISION REQUIRED")
        st.markdown(f"""
        <div class="{vc}">
            <strong style="font-size:1.0rem;">{vt}</strong><br>
            <span style="font-size:0.8rem;color:#8899aa;">
            HIGH: {summary['HIGH']}ea &nbsp; MED: {summary['MED']}ea &nbsp; LOW: {summary['LOW']}ea
            &nbsp;|&nbsp; Correctable: {summary['total_correctable']}ea
           </span>
        </div>
        """, unsafe_allow_html=True)

        tab_list, tab_post, tab_cost = st.tabs([
            T.get("tab_st3_list","🔧 Correction List"),
            T.get("tab_st3_compare","📊 Before & After"),
            T.get("tab_st3_cost","💰 Cost Savings"),
        ])

        with tab_list:
            st.divider()
            priority_icon = {"HIGH": "🔴", "MED": "🟡", "LOW": "🟢"}
            type_label = {
                "uniform_offset": T.get("type_uniform","Uniform Offset"),
                "local_linear":   T.get("type_local","Local Linear"),
                "feature_level":  T.get("type_feature","Feature Correction"),
                "process":        T.get("type_process","Process Adj."),
            }
            for c in corrections:
                icon  = priority_icon.get(c.priority, "⚪")
                ttype = type_label.get(c.correction_type, c.correction_type)
                if c.correction_type in ("uniform_offset", "process"):
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #252b36;border-radius:6px;
                        padding:10px 14px;margin:6px 0;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <strong> {c.id} [{ttype}] {c.feature_name}</strong><br>
                    <span style="font-size:0.78rem;color:#8899aa;">{c.note}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    clr = "#00d4aa" if c.correction_mm >= 0 else "#ff6b35"
                    dir_s = "+" if c.correction_mm >= 0 else ""
                    st.markdown(f"""
                    <div style="background:#111318;border:1px solid #252b36;border-radius:6px;
                        padding:10px 14px;margin:6px 0;display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div style="flex:1;">
                    <strong style="font-size:0.9rem;">{c.id} [{ttype}] {c.feature_name}</strong><br>
                    <span style="font-size:0.78rem;color:#8899aa;">{c.note}</span><br>
                    <span class="mono" style="font-size:0.72rem;color:#55667a;">
                        Mold: {c.mold_current:.4f} → {c.mold_corrected:.4f} mm
                    </span>
                    </div>
                    <div style="text-align:right;">
                    <span style="font-family:'Space Mono';font-size:1.1rem;font-weight:700;color:{clr};">
                        {dir_s}{c.correction_mm:.4f} mm
                    </span>
                    </div>
                    </div>""", unsafe_allow_html=True)

        with tab_post:
            if not inv["post_correction"].empty:
                post_df = inv["post_correction"]
                st.dataframe(style_verdict_df(post_df, verdict_col="결과"),
                             use_container_width=True, hide_index=True)
                if "Feature" in post_df.columns:
                    fig = go.Figure([
                        go.Bar(name=T.get("label_pre_dev","Pre-Dev."), x=post_df["Feature"],
                               y=post_df["보정 전 편차"].abs(), marker_color="#ff6b35"),
                        go.Bar(name=T.get("label_post_dev","Post-Dev."), x=post_df["Feature"],
                               y=post_df["보정 후 편차"].abs(), marker_color="#00d4aa"),
                    ])
                    fig.update_layout(barmode="group", paper_bgcolor="#0a0c0f",
                                      plot_bgcolor="#111318", font_color="#e2e8f0",
                                      height=300, margin=dict(l=20,r=20,t=30,b=20))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(T.get("msg_no_correction_needed","No correction needed."))

        with tab_cost:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T.get("metric_trial_before","Current Trials"), f"{cost['trial_before']} times")
            c2.metric(T.get("metric_trial_after", "Expected Trials"), f"{cost['trial_after']} times",
                      delta=f"-{cost['trial_before']-cost['trial_after']}")
            c3.metric(T.get("metric_cost_saving", "Cost Reduction"), f"~{cost['cost_reduction_pct']}%")
            c4.metric(T.get("metric_time_saving", "Time Saving"),    f"~{cost['dev_time_saving_weeks']} wks")

            if not inv["post_correction"].empty:
                st.divider()
                st.markdown(f"#### {T.get('title_export','💾 Export Results')}")
                b1, b2 = st.columns(2)
                with b1:
                    csv_data = inv["post_correction"].to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(T.get("btn_download_csv","📄 Download CSV"),
                                       csv_data, "mold_correction_report.csv", "text/csv",
                                       use_container_width=True)
                with b2:
                    try:
                        cad_script = generate_cad_macro_script(inv["post_correction"])
                        st.download_button(T.get("btn_download_cad","📐 CAD Script"),
                                           cad_script, "AutoCAD_Update.scr", "text/plain",
                                           use_container_width=True)
                    except:
                        pass
