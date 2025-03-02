import os
import gradio as gr
from gradio_client import Client, handle_file
from pathlib import Path
from gradio.utils import get_cache_folder

import torch
import torchvision.transforms as transforms
from PIL import Image

import cv2
import numpy as np
import ast

# from zerogpu import init_zerogpu

# init_zerogpu()


class Examples(gr.helpers.Examples):
    def __init__(self, *args, cached_folder=None, **kwargs):
        super().__init__(*args, **kwargs, _initiated_directly=False)
        if cached_folder is not None:
            self.cached_folder = cached_folder
            # self.cached_file = Path(self.cached_folder) / "log.csv"
        self.create()


def postprocess(output, prompt):
    result = []
    image = Image.open(output)
    w, h = image.size
    n = len(prompt)
    slice_width = w // n

    for i in range(n):
        left = i * slice_width
        right = (i + 1) * slice_width if i < n - 1 else w
        cropped_img = image.crop((left, 0, right, h))

        caption = prompt[i]

        result.append((cropped_img, caption))
    return result

# user click the image to get points, and show the points on the image
def get_point(img, sel_pix, evt: gr.SelectData):
    # print(img, sel_pix)
    if len(sel_pix) < 5:
        sel_pix.append((evt.index, 1))    # default foreground_point
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # draw points
    
    for point, label in sel_pix:
        cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    # if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(sel_pix)
    return img, sel_pix

def set_point(img, checkbox_group, sel_pix, semantic_input):
    ori_img = img
    # print(img, checkbox_group, sel_pix, semantic_input)
    sel_pix = ast.literal_eval(sel_pix)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(sel_pix) <= 5 and len(sel_pix) > 0:
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    
    return ori_img, img, sel_pix


# undo the selected point
def undo_points(orig_img, sel_pix):
    if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
        temp = cv2.imread(image_examples[orig_img][0])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    else:
        temp = cv2.imread(orig_img)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    # draw points
    if len(sel_pix) != 0:
        sel_pix.pop()
        for point, label in sel_pix:
            cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    return temp, sel_pix


HF_TOKEN = os.environ.get('HF_KEY')

client = Client("Canyu/Diception",
                max_workers=3,
                hf_token=HF_TOKEN)

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

def process_image_check(path_input, prompt, sel_points, semantic):
    if path_input is None:
        raise gr.Error(
            "Missing image in the left pane: please upload an image first."
        )
    if len(prompt) == 0:
        raise gr.Error(
            "At least 1 prediction type is needed."
        )


def inf(image_path, prompt, sel_points, semantic):
    if isinstance(sel_points, str):
        sel_points = ast.literal_eval(selected_points)
    print('=========== PROCESS IMAGE CHECK ===========')
    print(f"Image Path: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Selected Points (before processing): {sel_points}")
    print(f"Semantic Input: {semantic}")
    print('===========================================')

    if 'point segmentation' in prompt and len(sel_points) == 0:
        raise gr.Error(
            "At least 1 point is needed."
        )
        return
    if 'point segmentation' not in prompt and len(sel_points) != 0:
        raise gr.Error(
            "You must select 'point segmentation' when performing point segmentation."
        )
        return

    if 'semantic segmentation' in prompt and semantic == '':
        raise gr.Error(
            "Target category is needed."
        )
        return
    if 'semantic segmentation' not in prompt and semantic != '':
        raise gr.Error(
            "You must select 'semantic segmentation' when performing semantic segmentation."
        )
        return

    # return None
    # inputs = process_image_4(image_path, prompt, sel_points, semantic)

    prompt_str = str(sel_points)
    
    result = client.predict(
      input_image=handle_file(image_path), 
      checkbox_group=prompt, 
      selected_points=prompt_str, 
      semantic_input=semantic,
      api_name="/inf"
    )

    result = postprocess(result, prompt)
    return result

def clear_cache():
    return None, None

def dummy():
    pass

def run_demo_server():
    options = ['depth', 'normal', 'entity segmentation', 'human pose', 'point segmentation', 'semantic segmentation']
    gradio_theme = gr.themes.Default()
    with gr.Blocks(
        theme=gradio_theme,
        title="Diception",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
            .hideme {
                display: none;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        selected_points = gr.State([])      # store points
        original_image = gr.State(value=None)   # store original image without points, default None
        gr.HTML(
            """
            <h1>DICEPTION: A Generalist Diffusion Model for Vision Perception</h1>
            <h3>One single model solves multiple perception tasks, producing impressive results!</h3>
            <h3>Due to the GPU quota limit, if an error occurs, please wait for 5 minutes before retrying.</h3>
            <p align="center">
            <a title="arXiv" href="https://arxiv.org/abs/2502.17157" target="_blank" rel="noopener noreferrer" 
                    style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
            </a>
            <a title="Github" href="https://github.com/aim-uofa/Diception" target="_blank" rel="noopener noreferrer" 
                    style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/aim-uofa/Diception?label=GitHub%20%E2%98%85&logo=github&color=C8C" 
                        alt="badge-github-stars">
            </a>
            </p>
        """
        )
        selected_points_tmp = gr.Textbox(label="Points", elem_classes="hideme")

        with gr.Row():
            checkbox_group = gr.CheckboxGroup(choices=options, label="Task")
        with gr.Row():
            semantic_input = gr.Textbox(label="Category Name", placeholder="e.g. person/cat/dog/elephant......  (for semantic segmentation only, in COCO)")
        with gr.Row():
            gr.Markdown('For non-human image inputs, the pose results may have issues. Same when perform semantic segmentation with categories that are not in COCO.')
        with gr.Row():
            gr.Markdown('The results of semantic segmentation may be unstable because:')
        with gr.Row():
            gr.Markdown('- We only trained on COCO, whose quality and quantity are insufficient to meet the requirements.')
        with gr.Row():
            gr.Markdown('- Semantic segmentation is more complex than other tasks, as it requires accurately learning the relationship between semantics and objects.')
        with gr.Row():
            gr.Markdown('However, we are still able to produce some high-quality semantic segmentation results, strongly demonstrating the potential of our approach.')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                )

                with gr.Column():
                    with gr.Row():
                        gr.Markdown('You can click on the image to select points prompt. At most 5 point.')

                    matting_image_submit_btn = gr.Button(
                        value="Run", variant="primary"
                    )

                with gr.Row():
                    undo_button = gr.Button('Undo point')
                    matting_image_reset_btn = gr.Button(value="Reset")
                
                
            with gr.Column():
                matting_image_output = gr.Gallery(label="Results")
                    
                
            

        # img_clear_button.click(clear_cache, outputs=[input_image, matting_image_output])

        matting_image_submit_btn.click(
            fn=process_image_check,
            inputs=[input_image, checkbox_group, selected_points, semantic_input],
            outputs=None,
            preprocess=False,
            queue=False,
        ).success(
            fn=inf,
            inputs=[original_image, checkbox_group, selected_points, semantic_input],
            outputs=[matting_image_output],
            concurrency_limit=1,
        )

        matting_image_reset_btn.click(
            fn=lambda: (
                None,
                None,
                []
            ),
            inputs=[],
            outputs=[
                input_image,
                matting_image_output,
                selected_points
            ],
            queue=False,
        )


        # once user upload an image, the original image is stored in `original_image`
        def store_img(img):
            return img, []  # when new image is uploaded, `selected_points` should be empty
        input_image.upload(
            store_img,
            [input_image],
            [original_image, selected_points]
        )

        input_image.select(
            get_point,
            [original_image, selected_points],
            [input_image, selected_points],
        )

        undo_button.click(
            undo_points,
            [original_image, selected_points],
            [input_image, selected_points]
        )

        examples = gr.Examples(
            fn=set_point,
	        run_on_click=True,
            examples=[
                ["assets/woman.jpg", ['point segmentation', 'depth', 'normal', 'entity segmentation', 'human pose', 'semantic segmentation'], '[([2744, 975], 1), ([3440, 1954], 1), ([2123, 2405], 1), ([838, 1678], 1), ([4688, 1922], 1)]', 'person'],         
                ["assets/woman2.jpg", ['point segmentation', 'depth', 'entity segmentation', 'semantic segmentation', 'human pose'], '[([687, 1416], 1), ([1021, 707], 1), ([1138, 1138], 1), ([1182, 1583], 1), ([1188, 2172], 1)]', 'person'],         
                ["assets/board.jpg", ['point segmentation', 'depth', 'entity segmentation', 'normal'], '[([1003, 2163], 1)]', ''],         
                ["assets/lion.jpg", ['point segmentation', 'depth', 'semantic segmentation'], '[([1287, 671], 1)]', 'lion'],       
                ["assets/apple.jpg", ['point segmentation', 'depth', 'semantic segmentation', 'normal', 'entity segmentation'], '[([3367, 1950], 1)]','apple'],
                ["assets/room.jpg", ['point segmentation', 'depth', 'semantic segmentation', 'normal', 'entity segmentation'], '[([1308, 2215], 1)]', 'chair'],       
                ["assets/car.jpg", ['point segmentation', 'depth', 'semantic segmentation', 'normal', 'entity segmentation'], '[([1276, 1369], 1)]', 'car'],       
                ["assets/person.jpg", ['point segmentation', 'depth', 'semantic segmentation', 'normal', 'entity segmentation', 'human pose'], '[([3253, 1459], 1)]', 'tie'],       
                ["assets/woman3.jpg", ['point segmentation', 'depth', 'entity segmentation'], '[([420, 692], 1)]', ''],
                ["assets/cat.jpg", ['point segmentation', 'depth', 'entity segmentation', 'semantic segmentation'], '[([756, 661], 1)]', 'cat'],
                ["assets/room2.jpg", ['point segmentation', 'depth', 'entity segmentation', 'semantic segmentation', 'normal'], '[([3946, 224], 1)]', 'laptop'],
                ["assets/cartoon_cat.png", ['point segmentation', 'depth', 'entity segmentation', 'semantic segmentation', 'normal'], '[([1478, 3048], 1)]', 'cat'],
                ["assets/sheep.jpg", ['point segmentation', 'depth', 'entity segmentation', 'semantic segmentation'], '[([1789, 1791], 1), ([1869, 1333], 1)]', 'sheep'],
                ["assets/cartoon_girl.jpeg", ['point segmentation', 'depth', 'entity segmentation', 'normal', 'human pose', 'semantic segmentation'], '[([1208, 2089], 1), ([635, 2731], 1), ([1070, 2888], 1), ([1493, 2350], 1)]', 'person'],
            ],
            inputs=[input_image, checkbox_group, selected_points_tmp, semantic_input],
            outputs=[original_image, input_image, selected_points],
            cache_examples=False,
        )

        
    demo.queue(
        api_open=False,
    ).launch()


if __name__ == '__main__':

    run_demo_server()
