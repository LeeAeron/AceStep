"""LoKr tab dataset and adapter-setting controls."""

from __future__ import annotations

import gradio as gr
from pathlib import Path
import os

from acestep.ui.gradio.i18n import t


def get_project_root() -> Path:
    """Finds the project root (the parent directory relative to the current file)."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent.parent


def get_default_dataset_path() -> str:
    """Returns the path to datasets relative to the project root."""
    project_root = get_project_root()
    dataset_path = project_root / "datasets" / "preprocessed_tensors"
    return str(dataset_path)


def resolve_tensor_path(tensor_path: str | None) -> str:
    """Resolves the path to tensors. If an external path is passed, it is used; if None or empty, the default is used."""
    if tensor_path and str(tensor_path).strip():
        return str(tensor_path)
    return get_default_dataset_path()


def build_lokr_dataset_and_adapter_controls(tensor_output_dir: str = None) -> dict[str, object]:
    """Render LoKr dataset selector and adapter-parameter controls.
    
    Args:
        tensor_output_dir: Path to the directory with tensors from the dataset tab (can be external)
    """

    resolved_path = resolve_tensor_path(tensor_output_dir)

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(f"<h3>📊 {t('training.lokr_section_tensors')}</h3>")
            gr.Markdown(t("training.lokr_tensor_selection_desc"))

            lokr_training_tensor_dir = gr.Textbox(
                label=t("training.preprocessed_tensors_dir"),
                placeholder="Select tensor directory",
                value=resolved_path,
                info=t("training.preprocessed_tensors_info"),
                elem_classes=["has-info-container"],
            )

            lokr_load_dataset_btn = gr.Button(t("training.load_dataset_btn"), variant="secondary")

            lokr_training_dataset_info = gr.Textbox(
                label=t("training.dataset_info"),
                interactive=False,
                lines=7,
            )

        with gr.Column(scale=1):
            gr.HTML(f"<h3>⚙️ {t('training.lokr_section_settings')}</h3>")

            lokr_linear_dim = gr.Slider(
                minimum=4,
                maximum=256,
                step=4,
                value=64,
                label=t("training.lokr_linear_dim"),
                info=t("training.lokr_linear_dim_info"),
                elem_classes=["has-info-container"],
            )
            lokr_linear_alpha = gr.Slider(
                minimum=4,
                maximum=512,
                step=4,
                value=128,
                label=t("training.lokr_linear_alpha"),
                info=t("training.lokr_linear_alpha_info"),
                elem_classes=["has-info-container"],
            )
            lokr_factor = gr.Number(
                label=t("training.lokr_factor"),
                value=-1,
                precision=0,
                info=t("training.lokr_factor_info"),
                elem_classes=["has-info-container"],
            )
            lokr_decompose_both = gr.Checkbox(
                label=t("training.lokr_decompose_both"),
                value=False,
                info=t("training.lokr_decompose_both_info"),
                elem_classes=["has-info-container"],
            )
            lokr_use_tucker = gr.Checkbox(
                label=t("training.lokr_use_tucker"),
                value=False,
                info=t("training.lokr_use_tucker_info"),
                elem_classes=["has-info-container"],
            )
            lokr_use_scalar = gr.Checkbox(
                label=t("training.lokr_use_scalar"),
                value=False,
                info=t("training.lokr_use_scalar_info"),
                elem_classes=["has-info-container"],
            )
            lokr_weight_decompose = gr.Checkbox(
                label=t("training.lokr_weight_decompose"),
                value=True,
                info=t("training.lokr_weight_decompose_info"),
                elem_classes=["has-info-container"],
            )

    return {
        "lokr_training_tensor_dir": lokr_training_tensor_dir,
        "lokr_load_dataset_btn": lokr_load_dataset_btn,
        "lokr_training_dataset_info": lokr_training_dataset_info,
        "lokr_linear_dim": lokr_linear_dim,
        "lokr_linear_alpha": lokr_linear_alpha,
        "lokr_factor": lokr_factor,
        "lokr_decompose_both": lokr_decompose_both,
        "lokr_use_tucker": lokr_use_tucker,
        "lokr_use_scalar": lokr_use_scalar,
        "lokr_weight_decompose": lokr_weight_decompose,
    }