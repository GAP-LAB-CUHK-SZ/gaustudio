import gradio as gr

# import spaces
import torch
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import uuid

from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AsymmetricCroCo3DStereo.from_pretrained(
    "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to(DEVICE)


def create_blueprint(image_name_list: list[str], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    if len(image_name_list) > 4:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"{log_path}"),
            ),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Spatial3DView(origin=f"{log_path}"),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{log_path}/camera_{i}/pinhole/",
                                contents=[
                                    "+ $origin/**",
                                ],
                            )
                            for i in range(len(image_name_list))
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint


# @spaces.GPU
def predict(image_name_list: list[str] | str):
    # check if is list or string and if not raise error
    if not isinstance(image_name_list, list) and not isinstance(image_name_list, str):
        raise gr.Error(
            f"Input must be a list of strings or a string, got: {type(image_name_list)}"
        )
    uuid_str = str(uuid.uuid4())
    filename = Path("tmp/gradio") / f"{uuid_str}.rrd"
    filename.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"{uuid_str}")
    log_path = Path("world")

    if isinstance(image_name_list, str):
        image_name_list = [image_name_list]

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_name_list,
        model=model,
        device=DEVICE,
        batch_size=1,
    )

    blueprint: rrb.Blueprint = create_blueprint(image_name_list, log_path)
    rr.send_blueprint(blueprint)

    rr.set_time_sequence("sequence", 0)
    log_optimized_result(optimized_results, log_path)
    rr.save(filename.as_posix())
    return filename.as_posix()


with gr.Blocks(
    css=""".gradio-container {margin: 0 !important; min-width: 100%};""",
    title="Mini-DUSt3R Demo",
) as demo:
    # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
    gr.HTML('<h2 style="text-align: center;">Mini-DUSt3R Demo</h2>')
    gr.HTML(
        '<p style="text-align: center;">Unofficial DUSt3R demo using the mini-dust3r pip package</p>'
    )
    gr.HTML(
        '<p style="text-align: center;">More info <a href="https://github.com/pablovela5620/mini-dust3r">here</a></p>'
    )
    with gr.Tab(label="Single Image"):
        with gr.Column():
            single_image = gr.Image(type="filepath", height=300)
            run_btn_single = gr.Button("Run")
            rerun_viewer_single = Rerun(height=900)
            run_btn_single.click(
                fn=predict, inputs=[single_image], outputs=[rerun_viewer_single]
            )

            example_single_dir = Path("examples/single_image")
            example_single_files = sorted(example_single_dir.glob("*.png"))

            examples_single = gr.Examples(
                examples=example_single_files,
                inputs=[single_image],
                outputs=[rerun_viewer_single],
                fn=predict,
                cache_examples="lazy",
            )
    with gr.Tab(label="Multi Image"):
        with gr.Column():
            multi_files = gr.File(file_count="multiple")
            run_btn_multi = gr.Button("Run")
            rerun_viewer_multi = Rerun(height=900)
            run_btn_multi.click(
                fn=predict, inputs=[multi_files], outputs=[rerun_viewer_multi]
            )


demo.launch()
