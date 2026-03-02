"""
Settings panel component - hardware and model configuration.
"""
import sys
from pathlib import Path

import streamlit as st
from loguru import logger

from config import PROJECTS_DIR, CACHE_DIR, CHECKPOINTS_DIR
from utils import (
    get_dit_handler,
    get_llm_handler,
    is_dit_ready,
    is_llm_ready,
    initialize_dit,
    initialize_llm,
)


def show_settings_panel() -> None:
    """Display settings and configuration panel."""
    st.markdown("## ⚙️ Settings & Configuration")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🤖 Models", "🖥️ Hardware", "📦 Storage", "ℹ️ About"]
    )

    with tab1:
        _show_model_settings()
    with tab2:
        _show_hardware_settings()
    with tab3:
        _show_storage_settings()
    with tab4:
        _show_about_section()


# Models tab

def _show_model_settings() -> None:
    """Model initialisation controls."""
    st.markdown("### 🤖 Model Initialisation")

    # --- DiT ---
    st.markdown("#### DiT (Diffusion Transformer)")

    dit_status = "✅ Loaded" if is_dit_ready() else "⏳ Not loaded"
    st.write(f"**Status:** {dit_status}")

    # Detect available DiT checkpoint names from disk
    dit_models = _list_dit_models()
    is_mac = sys.platform == "darwin"

    col1, col2 = st.columns(2)
    with col1:
        dit_model = st.selectbox(
            "DiT Model",
            options=dit_models,
            index=(
                dit_models.index("acestep-v15-sft")
                if "acestep-v15-sft" in dit_models
                else 0
            ),
            key="dit_model_select",
        )
    with col2:
        dit_device = st.selectbox(
            "Device",
            options=["cuda", "cpu"],
            index=0,
            key="dit_device_select",
        )

    offload_cpu = st.checkbox(
        "Offload to CPU when idle",
        value=not is_mac,
        key="dit_offload",
    )

    if st.button(
        "🚀 Load DiT Model",
        key="init_dit_btn",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Loading DiT model (may take a minute)..."):
            status, ok = initialize_dit(
                config_path=dit_model,
                device=dit_device,
                offload_to_cpu=offload_cpu,
            )
        if ok:
            st.success(f"✅ DiT loaded: {dit_model}")
        else:
            st.error(f"❌ DiT init failed: {status}")

    st.divider()

    # --- LLM ---
    st.markdown("#### 5Hz LM (Language Model)")

    llm_status = "✅ Loaded" if is_llm_ready() else "⏳ Not loaded"
    st.write(f"**Status:** {llm_status}")

    lm_models = _list_lm_models()
    default_backend = "pt" or "vllm"

    col1, col2 = st.columns(2)
    with col1:
        lm_model = st.selectbox(
            "LM Model",
            options=lm_models if lm_models else ["acestep-5Hz-lm-1.7B"],
            index=0,
            key="lm_model_select",
        )
    with col2:
        backend = st.selectbox(
            "Backend",
            options=["pt", "vllm"],
            index=["pt", "vllm"].index(default_backend),
            key="lm_backend_select",
        )

    if st.button(
        "🚀 Load LLM",
        key="init_llm_btn",
        use_container_width=True,
    ):
        with st.spinner("Loading LLM (may take a minute)..."):
            status, ok = initialize_llm(
                lm_model_path=lm_model,
                backend=backend,
                device=dit_device if "dit_device_select" in st.session_state else "cuda",
            )
        if ok:
            st.success(f"✅ LLM loaded: {lm_model}")
        else:
            st.error(f"❌ LLM init failed: {status}")

    st.caption(
        "LLM is **optional** — it enriches generation with CoT "
        "reasoning but is not required for basic text-to-music."
    )


# Hardware tab

def _show_hardware_settings() -> None:
    """Hardware and GPU configuration display."""
    st.markdown("### 🖥️ Hardware Info")

    try:
        import torch

        col1, col2 = st.columns(2)
        with col1:
            st.metric("PyTorch", torch.__version__)
        with col2:
            cuda = torch.cuda.is_available()
            st.metric("CUDA", "✅" if cuda else "❌")

        if cuda:
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = (
                    torch.cuda.get_device_properties(i).total_memory
                    / 1e9
                )
                st.write(f"GPU {i}: **{name}** — {mem:.1f} GB")
    except ImportError:
        st.warning("PyTorch not installed")

    st.markdown("#### System")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Platform", sys.platform)
    with col2:
        st.metric("Python", sys.version.split()[0])


# Storage tab

def _show_storage_settings() -> None:
    """Storage and cache management."""
    st.markdown("### 📦 Storage & Cache")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Projects**")
        st.code(str(PROJECTS_DIR), language="bash")
        n_projects = len(list(PROJECTS_DIR.glob("*/")))
        st.metric("Projects", n_projects)

    with col2:
        st.markdown("**Cache**")
        st.code(str(CACHE_DIR), language="bash")
        import os

        cache_bytes = sum(
            os.path.getsize(f)
            for f in CACHE_DIR.rglob("*")
            if f.is_file()
        )
        st.metric("Cache Size", f"{cache_bytes / 1e6:.1f} MB")

    if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
        import shutil

        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)
        st.success("Cache cleared")


# About tab

def _show_about_section() -> None:
    """About ACE Studio Portable."""
    st.markdown("### ℹ️ About ACE Studio Portable")
    st.markdown(
        """
**ACE Studio Portable** is a modern Streamlit UI for
[ACE-Step 1.5 Portable] — an open-source music generation foundation model.

**Features:** text-to-music, covers, repainting, batch
generation (up to 8 songs), project management.
"""
    )

    st.link_button("GitHub", "https://github.com/LeeAeron/AceStep")

    st.caption("ACE Studio Portable v0.1.0")


# Helpers

def _list_dit_models() -> list:
    """Scan checkpoints dir for DiT model folders."""
    handler = get_dit_handler()
    if handler is not None:
        try:
            models = handler.get_available_acestep_v15_models()
            if models:
                return models
        except Exception:
            pass
    # Fallback: scan disk
    pattern = "acestep-v15-*"
    found = sorted(
        p.name
        for p in CHECKPOINTS_DIR.glob(pattern)
        if p.is_dir()
    )
    return found if found else ["acestep-v15-sft"]


def _list_lm_models() -> list:
    """Scan checkpoints dir for LM model folders."""
    handler = get_llm_handler()
    if handler is not None:
        try:
            models = handler.get_available_5hz_lm_models()
            if models:
                return models
        except Exception:
            pass
    # Fallback: scan disk
    pattern = "acestep-5Hz-lm-*"
    found = sorted(
        p.name
        for p in CHECKPOINTS_DIR.glob(pattern)
        if p.is_dir()
    )
    return found if found else ["acestep-5Hz-lm-1.7B"]
