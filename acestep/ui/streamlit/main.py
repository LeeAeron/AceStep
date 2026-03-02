"""
ACE Studio Portable - Modern Streamlit UI for Music Generation
Main application entry point
"""

import torch

_original_cuda_mem_get_info = torch.cuda.mem_get_info

def patched_cuda_mem_get_info(device=None):
    """Возвращаем фейковые значения VRAM - много свободной памяти"""
    total = 8589934592  # 8GB
    free = 4294967296   # 4GB
    return (free, total)

torch.cuda.mem_get_info = patched_cuda_mem_get_info

import streamlit as st
import sys
import os
from pathlib import Path

os.environ["ACESTEP_VAE_ON_CPU"] = "0"  # deny VAE on CPU
os.environ["ACESTEP_FORCE_GPU_DECODE"] = "1"  # Force GPU decode

# Configure Streamlit page
st.set_page_config(
    page_title="ACE Studio Portable",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "ACE Studio Portable v0.1.0 - Streamlit UI for "
            "ACE-Step Music Generation"
        ),
    },
)

# Custom CSS
st.markdown(
    """
<style>
    .main { padding: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .stButton > button { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "tab" not in st.session_state:
    st.session_state.tab = "dashboard"
if "editor_mode" not in st.session_state:
    st.session_state.editor_mode = "repaint"
if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

# Import components
from components import (
    show_dashboard,
    show_generation_wizard,
    show_editor,
    show_batch_generator,
    show_settings_panel,
)
from utils import is_dit_ready, initialize_dit, initialize_llm


if "_models_auto_init_done" not in st.session_state:
    st.session_state._models_auto_init_done = True
    if not is_dit_ready():
        with st.spinner(
            "Loading DiT model (first launch, may take a minute)..."
        ):
            _status, _ok = initialize_dit(
                config_path="acestep-v15-sft",
                device="cuda",
                offload_to_cpu=True,
                compile_model=False,
            )
            if _ok:
                st.toast("DiT model loaded successfully", icon="✅")
            else:
                st.toast(
                    f"DiT auto-init failed: {_status}",
                    icon="⚠️",
                )
        # Also try LLM (non-blocking; optional)
        _backend = "pt"
        with st.spinner("Loading LLM (optional, for CoT)..."):
            _lm_status, _lm_ok = initialize_llm(
                backend=_backend,
                device="cuda",
                offload_to_cpu=True,
            )
            if _lm_ok:
                st.toast("LLM loaded successfully", icon="✅")
            else:
                st.toast(
                    "LLM not loaded (optional)", icon="ℹ️",
                )


with st.sidebar:
    st.markdown("### 🎹 ACE Studio Portable")

    nav_selection = st.radio(
        "Select Tab",
        options=[
            "📊 Dashboard",
            "🎵 Generate",
            "🎛️ Edit",
            "📦 Batch",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
        index=[
            "dashboard",
            "generate",
            "editor",
            "batch",
            "settings",
        ].index(st.session_state.tab),
    )

    tab_map = {
        "📊 Dashboard": "dashboard",
        "🎵 Generate": "generate",
        "🎛️ Edit": "editor",
        "📦 Batch": "batch",
        "⚙️ Settings": "settings",
    }
    st.session_state.tab = tab_map[nav_selection]

    st.divider()

    # Quick project count
    try:
        from utils import ProjectManager
        from config import PROJECTS_DIR

        pm = ProjectManager(PROJECTS_DIR)
        projects = pm.list_projects()
        st.metric("💾 Projects", len(projects))
    except Exception:
        pass

    st.divider()

    # Model status (lightweight - never loads weights here)
    st.markdown("### 🤖 Model Status")

    from utils import is_llm_ready

    col1, col2 = st.columns(2)
    with col1:
        if is_dit_ready():
            st.success("✅ DiT")
        else:
            st.warning("⏳ DiT")
    with col2:
        if is_llm_ready():
            st.success("✅ LLM")
        else:
            st.info("⏸️ LLM")

    if not is_dit_ready():
        st.caption(
            "Go to **⚙️ Settings → Models** to initialise."
        )

    st.divider()

    # Quick help
    with st.expander("❓ Quick Help"):
        st.markdown(
            """
**Getting Started:**
1. Go to **Settings → Models** to load the AI model
2. Use **Generate** to create new songs
3. Use **Edit** to modify generated audio
4. Use **Batch** to generate multiple songs

**Tips:**
- Be descriptive in song captions
- Use editing to refine generated songs
"""
        )

# Main content area – route to selected tab
if st.session_state.tab == "dashboard":
    show_dashboard()
elif st.session_state.tab == "generate":
    show_generation_wizard()
elif st.session_state.tab == "editor":
    show_editor()
elif st.session_state.tab == "batch":
    show_batch_generator()
elif st.session_state.tab == "settings":
    show_settings_panel()
else:
    st.error(f"Unknown tab: {st.session_state.tab}")
    show_dashboard()

# Footer
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>
        🎵 <strong>ACE Studio Portable</strong> v0.1.0 |
        Powered by
        <a href="https://github.com/LeeAeron/AceStep ">
        ACE-Step</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)