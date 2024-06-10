import logging

import gradio as gr


def face_detect(image):
    logging.info("Image: {}".format(image))
    return image


demo = gr.Interface(
    fn=face_detect,
    inputs=[
        gr.Image(height=500),
        gr.Dropdown(
            ["add", "face-detect", "represent", "verify", "recognize"],
            label="Animal",
            info="Will add more animals later!",
        ),
    ],
    outputs="image",
)
demo.launch()
