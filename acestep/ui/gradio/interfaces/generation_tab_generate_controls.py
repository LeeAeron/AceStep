from typing import Any
import gradio as gr
from acestep.ui.gradio.i18n import t

import sys, os, threading, time, logging

logger = logging.getLogger(__name__)


def _build_left_generate_toggles(
    lm_initialized: bool,
    service_mode: bool,
) -> tuple[gr.Checkbox, gr.Checkbox]:
    with gr.Column(scale=1, variant="compact"):
        think_checkbox = gr.Checkbox(
            label=t("generation.think_label"),
            value=lm_initialized,
            scale=1,
            interactive=lm_initialized,
        )
        auto_score = gr.Checkbox(
            label=t("generation.auto_score_label"),
            value=False,
            scale=1,
            interactive=not service_mode,
        )
    return think_checkbox, auto_score


def _build_right_generate_toggles(service_mode: bool) -> tuple[gr.Checkbox, gr.Checkbox]:
    with gr.Column(scale=1, variant="compact"):
        autogen_checkbox = gr.Checkbox(
            label=t("generation.autogen_label"),
            value=False,
            scale=1,
            interactive=not service_mode,
        )
        auto_lrc = gr.Checkbox(
            label=t("generation.auto_lrc_label"),
            value=False,
            scale=1,
            interactive=not service_mode,
        )
    return autogen_checkbox, auto_lrc


def restart_engine():
    import sys, os, threading, time
    logger.warning("⚠ Restart request received via UI button.")

    def _delayed_restart():
        time.sleep(2)
        python = sys.executable
        os.environ["ACE_STEP_RESTARTED"] = "1"
        os.execl(python, python, *sys.argv)

    threading.Thread(target=_delayed_restart).start()
    return "⚠ Restarting engine..."


def build_generate_row_controls(
    service_pre_initialized: bool,
    init_params: dict[str, Any] | None,
    lm_initialized: bool,
    service_mode: bool,
) -> dict[str, Any]:
    params = init_params or {}
    generate_btn_interactive = params.get("enable_generate", False) if service_pre_initialized else False
    with gr.Row(equal_height=True, visible=True) as generate_btn_row:
        think_checkbox, auto_score = _build_left_generate_toggles(
            lm_initialized=lm_initialized,
            service_mode=service_mode,
        )
        with gr.Column(scale=18):
            generate_btn = gr.Button(
                t("generation.generate_btn"),
                variant="primary",
                size="lg",
                interactive=generate_btn_interactive,
            )
            restart_btn = gr.Button(
                t("generation.restart_btn"),
                variant="secondary",
                size="lg",
                interactive=True,
            )
        autogen_checkbox, auto_lrc = _build_right_generate_toggles(service_mode=service_mode)

    controls = {
        "think_checkbox": think_checkbox,
        "auto_score": auto_score,
        "generate_btn": generate_btn,
        "restart_btn": restart_btn,
        "generate_btn_row": generate_btn_row,
        "autogen_checkbox": autogen_checkbox,
        "auto_lrc": auto_lrc,
    }

    js_reload_on_ready = """
    () => {
        const checkServer = async () => {
            try {
                const resp = await fetch(window.location.origin, {cache: "no-store"});
                if (resp.ok) {
                    window.location.reload();
                } else {
                    setTimeout(checkServer, 2000);
                }
            } catch (e) {
                setTimeout(checkServer, 2000);
            }
        };
        setTimeout(checkServer, 2000);
    }
    """

    restart_btn.click(
        restart_engine,
        inputs=None,
        outputs=None,
        js=js_reload_on_ready
    )

    return controls
