import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="Deep Research — UI", layout="centered")
st.title("🧠 Deep Research — Graphical Interface")
st.markdown(
    """
Enter your **research topic**, **breadth**, and **depth**, then click **Generate Report**.  
This will internally run your TypeScript script and display the generated **output.md**.
"""
)

# Input form
with st.form("research_form"):
    topic = st.text_input("Research Topic", value="Education in India")
    breadth = st.slider("Research Breadth (recommended 2–10)", 1, 10, 4)
    depth = st.slider("Research Depth (recommended 1–5)", 1, 5, 2)
    submit = st.form_submit_button("Generate Report")

# Paths
project_root = Path(__file__).resolve().parent
output_md_path = project_root / "output.md"

if submit:
    st.info("⏳ Starting research process... Please wait.")
    cli_input = f"{topic}\n{breadth}\n{depth}\n"

    try:
        process = subprocess.run(
            ["cmd", "/c", "npx", "tsx", "--env-file=.env.local", "src/run.ts"],
            input=cli_input,
            text=True,
            capture_output=True,
            cwd=str(project_root),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        st.error("❌ Process timed out. Try smaller breadth/depth.")
    else:
        st.subheader("🧩 CLI Output")
        if process.stdout:
            st.code(process.stdout)
        if process.stderr:
            st.code(process.stderr)

        if output_md_path.exists():
            content = output_md_path.read_text(encoding="utf-8")
            st.success("✅ Report generated successfully!")
            st.markdown("---")
            st.subheader("📄 Generated Report")
            st.markdown(content, unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Report",
                data=content,
                file_name="output.md",
                mime="text/markdown",
            )
        else:
            st.error("⚠️ output.md file not found. Check CLI logs above for issues.")
