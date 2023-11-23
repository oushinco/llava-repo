import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests
import base64

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

police_logo = "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"

title_markdown = (f"""
# <img src="data:image/png;base64,{police_logo}" alt="Police" width="50"/>   NSWPF - Large Language and Vision Assistant
Contact DataScience@police.nsw.gov.au for questions or supports
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
""")


learn_more_markdown = ("""
### License
The service is a research preview. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    #[f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                    #[f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                    [f"{cur_dir}/examples/nswpf1.jpg", "What is in this image?"],
                    [f"{cur_dir}/examples/nswpf2.jpg", "What are the possible things I should do to find a staff?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaVA Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count,
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
