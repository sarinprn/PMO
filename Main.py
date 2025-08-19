# app.py — Streamlit PMO WebApp (Version 1 – UX/UI Focus)
# ---------------------------------------------------------
# A lovable, executive-friendly PMO dashboard prototype for Thailand Post.
# Features:
# - Upload & register projects with documents (Excel/PDF/Word/JPG)
# - AI-like similarity check (TF-IDF + cosine) to detect overlaps
# - Visual relationship map (Graphviz)
# - Executive dashboard with progress/status KPIs
# - Notification center & simple report export (CSV)
#
# Notes:
# - This is a self-contained prototype. Some document text extraction is simplified.
# - For real OCR/parse, wire to PyMuPDF/python-docx/openpyxl or cloud services.

import io
import base64
from datetime import date, datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import graphviz
import plotly.express as px
pip install scikit-learn

# ----------------------
# App Config & Theming
# ----------------------
st.set_page_config(
    page_title="Thailand Post PMO – Lovable Dashboard",
    page_icon="📮",
    layout="wide",
    initial_sidebar_state="expanded",
)

POSTAL_RED = "#E60012"
ACCENT = "#FFB3B8"  # soft supportive color
MUTED = "#F7F7F7"

# Inject a bit of lovable styling
st.markdown(
    f"""
    <style>
      .lovable-header h1, .lovable-header h2 {{
          color: {POSTAL_RED};
      }}
      .status-badge {{
          display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600;
          background:{MUTED}; color:#333; border:1px solid #eaeaea; margin-right:6px;
      }}
      .badge-green {{ background:#E8F5E9; color:#2E7D32; border:1px solid #C8E6C9; }}
      .badge-yellow {{ background:#FFFDE7; color:#F9A825; border:1px solid #FFF59D; }}
      .badge-red {{ background:#FFEBEE; color:#C62828; border:1px solid #FFCDD2; }}
      .upload-hint {{ color:#666; font-size:13px; }}
      .pill {{ background:{ACCENT}; color:#5a2a2d; padding:4px 10px; border-radius:999px; font-weight:600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Session State & Helpers
# ----------------------
if "projects" not in st.session_state:
    # Seed with a few demo projects for first-time experience
    st.session_state.projects: List[Dict] = [
        {
            "project_id": 1,
            "name": "EV Delivery Pilot 2024",
            "department": "Logistics",
            "start": date(2024, 6, 1),
            "end": date(2025, 3, 31),
            "budget": 18_000_000,
            "status": "ongoing",
            "progress": 62,
            "description": "Pilot deployment of electric vehicles for last-mile delivery in Bangkok metro.",
            "documents": [],
        },
        {
            "project_id": 2,
            "name": "Solar Roof Phase II",
            "department": "Facilities",
            "start": date(2024, 1, 10),
            "end": date(2025, 1, 10),
            "budget": 12_500_000,
            "status": "ongoing",
            "progress": 48,
            "description": "Expansion of PV capacity on sorting centers; integrate with EMS cooling systems.",
            "documents": [],
        },
        {
            "project_id": 3,
            "name": "Customer Engagement App Refresh",
            "department": "Digital",
            "start": date(2025, 2, 1),
            "end": date(2025, 9, 30),
            "budget": 7_800_000,
            "status": "planned",
            "progress": 10,
            "description": "Redesign mobile app flows; add chatbot & proactive delivery notifications.",
            "documents": [],
        },
    ]

if "next_id" not in st.session_state:
    st.session_state.next_id = 4

if "notifications" not in st.session_state:
    st.session_state.notifications = []

STATUS_COLORS = {
    "planned": "badge-yellow",
    "ongoing": "badge-green",
    "delayed": "badge-red",
    "completed": "badge-green",
}


def human_budget(x: float) -> str:
    return f"฿{x:,.0f}"


def make_similarity_matrix(projects: List[Dict]) -> pd.DataFrame:
    texts = [p.get("description", "") + " " + p.get("name", "") for p in projects]
    if len(texts) < 2:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english")
    try:
        X = vec.fit_transform(texts)
        sim = cosine_similarity(X)
        df = pd.DataFrame(sim, index=[p["name"] for p in projects], columns=[p["name"] for p in projects])
        np.fill_diagonal(df.values, np.nan)  # ignore self-similarity
        return df
    except ValueError:
        # Fallback if texts are empty
        return pd.DataFrame()


def find_overlaps(sim_df: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    overlaps = []
    if sim_df.empty:
        return overlaps
    for a in sim_df.index:
        for b in sim_df.columns:
            if a < b:  # dedupe pair
                score = sim_df.loc[a, b]
                if pd.notna(score) and score >= threshold:
                    overlaps.append({"a": a, "b": b, "score": float(score)})
    return overlaps


def add_notification(message: str, ntype: str = "info", project_name: str | None = None):
    st.session_state.notifications.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type": ntype,
        "project": project_name,
        "message": message,
        "read": False,
    })


# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("📮 PMO — Thailand Post")
menu = st.sidebar.radio(
    "ไปที่หน้า:",
    ["🏠 Overview", "📤 Upload Project", "🔎 Analysis", "📊 Dashboard", "🔔 Notifications", "📄 Reports"],
)

# Quick filters
with st.sidebar.expander("🔎 Filter Projects"):
    dept_filter = st.selectbox("Department", ["All"] + sorted({p["department"] for p in st.session_state.projects}))
    status_filter = st.selectbox("Status", ["All", "planned", "ongoing", "delayed", "completed"])

# Apply filters
filtered_projects = [
    p for p in st.session_state.projects
    if (dept_filter == "All" or p["department"] == dept_filter)
    and (status_filter == "All" or p["status"] == status_filter)
]

# ----------------------
# Overview
# ----------------------
if menu == "🏠 Overview":
    st.markdown("<div class='lovable-header'><h1>PMO Overview</h1></div>", unsafe_allow_html=True)
    st.write("จัดการโครงการแบบไม่ซ้ำซ้อน • เชื่อมโยงทุกหน่วยงาน • เข้าใจภาพรวมใน 1 นาที ✨")

    c1, c2, c3, c4 = st.columns(4)
    total = len(filtered_projects)
    ongoing = sum(1 for p in filtered_projects if p["status"] == "ongoing")
    delayed = sum(1 for p in filtered_projects if p["status"] == "delayed")
    budget_sum = sum(p["budget"] for p in filtered_projects)

    c1.metric("Total Projects", total)
    c2.metric("Ongoing", ongoing)
    c3.metric("Delayed", delayed)
    c4.metric("Total Budget", human_budget(budget_sum))

    # List cards
    st.subheader("โครงการล่าสุด")
    for p in filtered_projects:
        badge = STATUS_COLORS.get(p["status"], "")
        with st.container(border=True):
            st.markdown(f"<span class='status-badge {badge}'>{p['status'].title()}</span>", unsafe_allow_html=True)
            st.markdown(f"### {p['name']}  <span class='pill'>{p['department']}</span>", unsafe_allow_html=True)
            st.progress(int(p["progress"]))
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Timeline:** {p['start']} → {p['end']}")
            c2.write(f"**Budget:** {human_budget(p['budget'])}")
            c3.write(p.get("description", ""))

# ----------------------
# Upload Project
# ----------------------
elif menu == "📤 Upload Project":
    st.markdown("<div class='lovable-header'><h1>Upload / Register Project</h1></div>", unsafe_allow_html=True)
    st.caption("อัปโหลดไฟล์โครงการ (Excel, PDF, Word, JPG) พร้อมกรอกข้อมูลสำคัญ • ระบบจะช่วยวิเคราะห์ความซ้ำซ้อนให้")

    with st.form("upload_form", clear_on_submit=True):
        name = st.text_input("Project Name ชื่อโครงการ")
        dept = st.selectbox("Department หน่วยงาน", ["Logistics", "Facilities", "Digital", "HR", "Finance", "Operations"])
        c1, c2, c3 = st.columns(3)
        with c1:
            start = st.date_input("Start Date", value=date.today())
        with c2:
            end = st.date_input("End Date", value=date.today())
        with c3:
            budget = st.number_input("Budget (THB)", min_value=0, step=100000)
        status = st.selectbox("Status", ["planned", "ongoing", "delayed", "completed"], index=0)
        progress = st.slider("Progress %", 0, 100, 0)
        desc = st.text_area("Project Description (สำหรับวิเคราะห์ความเชื่อมโยง)")

        files = st.file_uploader(
            "แนบไฟล์เอกสาร (รองรับหลายไฟล์)",
            type=["pdf", "docx", "xlsx", "xls", "csv", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="อัปโหลดไฟล์ที่เกี่ยวข้องกับโครงการเพื่อเก็บรวมศูนย์",
        )

        submitted = st.form_submit_button("Save Project ➕")
        if submitted:
            if not name:
                st.error("กรุณากรอกชื่อโครงการ")
            else:
                pid = st.session_state.next_id
                st.session_state.next_id += 1
                docs = []
                for f in files or []:
                    docs.append({
                        "filename": f.name,
                        "type": f.type,
                        "size": getattr(f, 'size', None),
                    })
                st.session_state.projects.append({
                    "project_id": pid,
                    "name": name,
                    "department": dept,
                    "start": start,
                    "end": end,
                    "budget": float(budget),
                    "status": status,
                    "progress": int(progress),
                    "description": desc,
                    "documents": docs,
                })
                add_notification(f"สร้างโครงการใหม่: {name}", "success", name)
                st.success("บันทึกสำเร็จ! ระบบจะวิเคราะห์ความซ้ำซ้อนให้อัตโนมัติในหน้า Analysis ✨")

# ----------------------
# Analysis
# ----------------------
elif menu == "🔎 Analysis":
    st.markdown("<div class='lovable-header'><h1>Project Relationship Analysis</h1></div>", unsafe_allow_html=True)
    st.caption("AI จะช่วยเชื่อมโยงและเตือนโครงการที่ซ้ำซ้อน พร้อมแนะนำการรวม/ปรับโครงการ")

    projects = st.session_state.projects
    sim_df = make_similarity_matrix(projects)

    if sim_df.empty:
        st.info("ยังมีข้อมูลไม่เพียงพอสำหรับการคำนวณความใกล้เคียง โปรดเพิ่มคำอธิบายโครงการ")
    else:
        threshold = st.slider("Similarity Threshold (ซ้ำซ้อนเมื่อมากกว่า)", 0.5, 0.95, 0.7, 0.05)
        overlaps = find_overlaps(sim_df, threshold)

        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Similarity Matrix")
            st.dataframe(sim_df.style.format("{:.2f}").background_gradient(axis=None))
        with c2:
            st.subheader("Relationship Map (Graph)")
            g = graphviz.Digraph()
            g.attr(rankdir="LR", splines="spline", concentrate="true")
            # Add nodes
            for p in projects:
                color = POSTAL_RED if p["status"] in ("ongoing", "delayed") else "#999999"
                g.node(p["name"], shape="box", style="rounded,filled", fillcolor="#FFFFFF", color=color)
            # Add edges above threshold
            for ov in overlaps:
                penwidth = str(1 + (ov["score"] - threshold) * 6)
                g.edge(ov["a"], ov["b"], label=f"{ov['score']:.2f}", penwidth=penwidth)
            st.graphviz_chart(g, use_container_width=True)

        st.subheader("🔔 Detected Overlaps")
        if not overlaps:
            st.success("ไม่พบโครงการที่ซ้ำซ้อนตามเกณฑ์ที่ตั้งไว้ 🎉")
        else:
            for ov in sorted(overlaps, key=lambda x: -x["score"]):
                st.warning(f"{ov['a']} ↔ {ov['b']} • Similarity: {ov['score']:.2f}")
                add_notification(
                    f"พบโครงการซ้ำซ้อน: {ov['a']} ↔ {ov['b']} (Similarity {ov['score']:.2f})",
                    "warning",
                    project_name=f"{ov['a']} / {ov['b']}"
                )

# ----------------------
# Dashboard
# ----------------------
elif menu == "📊 Dashboard":
    st.markdown("<div class='lovable-header'><h1>Executive Dashboard</h1></div>", unsafe_allow_html=True)
    projects = filtered_projects
    if not projects:
        st.info("ไม่มีโครงการตามเงื่อนไขตัวกรองด้านซ้าย")
    else:
        df = pd.DataFrame(projects)
        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projects", len(df))
        c2.metric("Avg Progress", f"{df['progress'].mean():.0f}%")
        c3.metric("Budget", human_budget(df['budget'].sum()))
        delayed = (df["status"] == "delayed").sum()
        c4.metric("Delayed", int(delayed))

        # Charts
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Projects by Status")
            fig1 = px.histogram(df, x="status")
            st.plotly_chart(fig1, use_container_width=True)
        with c6:
            st.subheader("Budget by Department")
            fig2 = px.bar(df.groupby("department", as_index=False)["budget"].sum(), x="department", y="budget")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Project Progress")
        fig3 = px.bar(df.sort_values("progress", ascending=False), x="name", y="progress")
        st.plotly_chart(fig3, use_container_width=True)

        # Table
        st.subheader("Project Table")
        st.dataframe(df[["name", "department", "status", "progress", "start", "end", "budget"]])

# ----------------------
# Notifications
# ----------------------
elif menu == "🔔 Notifications":
    st.markdown("<div class='lovable-header'><h1>Notification Center</h1></div>", unsafe_allow_html=True)

    if not st.session_state.notifications:
        st.info("ยังไม่มีการแจ้งเตือน")
    else:
        for i, n in enumerate(reversed(st.session_state.notifications)):
            tag = n["type"].upper()
            st.write(f"**[{n['timestamp']}] {tag}** — {n['message']}")

# ----------------------
# Reports
# ----------------------
elif menu == "📄 Reports":
    st.markdown("<div class='lovable-header'><h1>Reports & Export</h1></div>", unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.projects)
    df_export = df[["project_id", "name", "department", "status", "progress", "start", "end", "budget", "description"]]

    st.write("สรุปโครงการทั้งหมด พร้อมส่งออกเป็น CSV ให้ผู้บริหารหรือคณะกรรมการ")
    st.dataframe(df_export)

    # Export CSV
    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="pmo_projects_export.csv",
        mime="text/csv",
    )

    st.caption("Tip: ต่อ API หรือฐานข้อมูลจริง แล้วใช้ตารางนี้สร้างรายงาน PDF แบบทางการได้ใน production.")
