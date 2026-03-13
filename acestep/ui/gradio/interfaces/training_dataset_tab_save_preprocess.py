"""Dataset save and preprocess controls for the training dataset tab."""

from __future__ import annotations

import gradio as gr
import os
from pathlib import Path

from acestep.ui.gradio.i18n import t

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_project_root() -> Path:
    """Finds the root of the project."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent.parent

def get_default_tensor_dir() -> str:
    """Returns the default path to tensors."""
    project_root = get_project_root()
    return str(project_root / "datasets" / "preprocessed_tensors")

def sync_save_path(tensor_dir: str) -> str:
    """Generates a path to save dataset.json within the selected folder."""
    if not tensor_dir:
        return ""
    if not os.path.isabs(tensor_dir):
        tensor_dir = os.path.join(PROJECT_ROOT, tensor_dir)
    return os.path.join(tensor_dir, "dataset.json")

def sync_load_existing_path(tensor_dir: str) -> str:
    """Generates a path for loading JSON within the selected folder."""
    if not tensor_dir:
        return ""
    if not os.path.isabs(tensor_dir):
        tensor_dir = os.path.join(PROJECT_ROOT, tensor_dir)
    return os.path.join(tensor_dir, "dataset.json")


def build_dataset_save_and_preprocess_controls(tensor_output_dir_val: str = None) -> dict[str, object]:
    """Render dataset save/load-preprocess controls and return component handles.
    
    Args:
        tensor_output_dir_val: Value from tensor_output_dir из scan_settings
    """

    gr.HTML(f"<hr><h3>💾 {t('training.step4_title')}</h3>")

    if tensor_output_dir_val:
        initial_path = tensor_output_dir_val
    else:
        initial_path = get_default_tensor_dir()

    with gr.Row():
        with gr.Column(scale=3):
            preprocess_output_dir = gr.Textbox(
                label=t("training.tensor_output_dir"),
                value=initial_path,
                placeholder="Select tensor output directory",
                info=t("training.tensor_output_info"),
                elem_classes=["has-info-container"],
            )

        with gr.Column(scale=1):
            preprocess_btn = gr.Button(
                t("training.preprocess_btn"),
                variant="primary",
                size="lg",
            )

    save_path = gr.Textbox(
        label=t("training.save_path"),
        value=sync_save_path(initial_path),
        interactive=False,
        elem_classes=["has-info-container"],
    )

    save_dataset_btn = gr.Button(
        t("training.save_dataset_btn"),
        variant="primary",
        size="lg",
    )

    save_status = gr.Textbox(
        label=t("training.save_status"),
        interactive=False,
        lines=5,
    )

    gr.HTML(f"<hr><h3>⚡ {t('training.step5_title')}</h3>")
    gr.Markdown(t("training.step5_intro"))

    load_existing_dataset_path = gr.Textbox(
        label=t("training.load_existing_label"),
        value=sync_load_existing_path(initial_path),
        placeholder="./datasets/dataset.json",
        info=t("training.load_existing_info"),
        elem_classes=["has-info-container"],
        interactive=False,
    )

    load_existing_dataset_btn = gr.Button(
        t("training.load_dataset_btn"),
        variant="secondary",
        size="lg",
    )

    load_existing_status = gr.Textbox(
        label=t("training.load_status"),
        interactive=False,
    )

    gr.Markdown(t("training.step5_details"))

    preprocess_mode = gr.Dropdown(
        label="Preprocess For",
        choices=["LoRA", "LoKr"],
        value="LoRA",
        info="LoRA keeps compatibility mode; LoKr uses per-sample source-style context.",
        elem_classes=["has-info-container"],
    )

    preprocess_progress = gr.Textbox(
        label=t("training.preprocess_progress"),
        interactive=False,
        lines=5,
    )

    preprocess_output_dir.change(
        sync_save_path,
        inputs=preprocess_output_dir,
        outputs=save_path
    )

    preprocess_output_dir.change(
        sync_load_existing_path,
        inputs=preprocess_output_dir,
        outputs=load_existing_dataset_path
    )

    return {
        "save_path": save_path,
        "save_dataset_btn": save_dataset_btn,
        "save_status": save_status,
        "load_existing_dataset_path": load_existing_dataset_path,
        "load_existing_dataset_btn": load_existing_dataset_btn,
        "load_existing_status": load_existing_status,
        "preprocess_mode": preprocess_mode,
        "preprocess_output_dir": preprocess_output_dir,
        "preprocess_btn": preprocess_btn,
        "preprocess_progress": preprocess_progress,
    }