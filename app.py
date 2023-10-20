# coding=utf-8
import os
import re
import argparse
import utils
import commons
import json
import torch
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from torch import no_grad, LongTensor
import gradio.processing_utils as gr_processing_utils
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces

hps_ms = utils.get_hparams_from_file(r'config/config.json')

audio_postprocess_ori = gr.Audio.postprocess

def audio_postprocess(self, y):
    data = audio_postprocess_ori(self, y)
    if data is None:
        return None
    return gr_processing_utils.encode_url_or_file_to_base64(data["name"])


gr.Audio.postprocess = audio_postprocess

def get_text(text, hps, is_symbol):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

def create_tts_fn(net_g_ms, speaker_id):
    def tts_fn(text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
        if limitation:
            text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
            max_len = 100
            if is_symbol:
                max_len *= 3
            if text_len > max_len:
                return "Error: Text is too long", None
        if not is_symbol:
            if language == 0:
                text = f"[ZH]{text}[ZH]"
            elif language == 1:
                text = f"[JA]{text}[JA]"
            else:
                text = f"{text}"
        stn_tst, clean_text = get_text(text, hps_ms, is_symbol)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        return "Success", (22050, audio)
    return tts_fn

def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_lang):
        if temp_lang == 0:
            clean_text = f'[ZH]{input_text}[ZH]'
        elif temp_lang == 1:
            clean_text = f'[JA]{input_text}[JA]'
        else:
            clean_text = input_text
        return _clean_text(clean_text, hps.data.text_cleaners) if is_symbol_input else ''

    return to_symbol_fn
def change_lang(language):
    if language == 0:
        return 0.6, 0.668, 1.2
    elif language == 1:
        return 0.6, 0.668, 1
    else:
        return 0.6, 0.668, 1

download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#tts-audio-{audio_id}").querySelector("audio");
    let text = root.querySelector("#input-text-{audio_id}").querySelector("textarea");
    if (audio == undefined)
        return;
    text = text.value;
    if (text == undefined)
        text = Math.floor(Math.random()*100000000);
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = text.substr(0, 20)+'.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    device = torch.device(args.device)
    categories = ["Genshin Impact", "Honkai Impact 3rd"]
    others = {
        "Blue Archive": "https://huggingface.co/spaces/sayashi/vits-models",
        "Princess Connect! Re:Dive": "https://huggingface.co/spaces/sayashi/vits-models-pcr",
        "Overwatch 2": "https://huggingface.co/spaces/sayashi/vits-models-ow2",
    }
    models = []
    with open("pretrained_models/info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)
    for i, info in models_info.items():
        if info['title'].split("-")[0] not in categories or not info['enable']:
            continue
        sid = info['sid']
        name_en = info['name_en']
        name_zh = info['name_zh']
        title = info['title']
        cover = f"pretrained_models/{i}/{info['cover']}"
        example = info['example']
        language = info['language']
        net_g_ms = SynthesizerTrn(
            len(hps_ms.symbols),
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=hps_ms.data.n_speakers if info['type'] == "multi" else 0,
            **hps_ms.model)
        utils.load_checkpoint(f'pretrained_models/{i}/{i}.pth', net_g_ms, None)
        _ = net_g_ms.eval().to(device)
        models.append((sid, name_en, name_zh, title, cover, example, language, net_g_ms, create_tts_fn(net_g_ms, sid), create_to_symbol_fn(hps_ms)))
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> vits-models\n"
            "## <center> Please do not generate content that could infringe upon the rights or cause harm to individuals or organizations.\n"
            "## <center> 请不要生成会对个人以及组织造成侵害的内容\n"
            "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=sayashi.vits-models)\n\n"
            "[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10QOk9NPgoKZUXkIhhuVaZ7SYra1MPMKH?usp=share_link)\n\n"
            "[![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm-dark.svg)](https://huggingface.co/spaces/sayashi/vits-models?duplicate=true)\n\n"
            "[![Finetune your own model](https://badgen.net/badge/icon/github?icon=github&label=Finetune%20your%20own%20model)](https://github.com/SayaSS/vits-finetuning)"
        )

        with gr.Tabs():
            for category in categories:
                with gr.TabItem(category):
                    with gr.TabItem("EN"):
                        for (sid, name_en, name_zh, title, cover, example, language, net_g_ms, tts_fn, to_symbol_fn) in models:
                            if title.split("-")[0] != category:
                                continue
                            with gr.TabItem(name_en):
                                with gr.Row():
                                    gr.Markdown(
                                        '<div align="center">'
                                        f'<a><strong>{title}</strong></a>'
                                        f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else ""
                                        '</div>'
                                    )
                                with gr.Row():
                                    with gr.Column():
                                        input_text = gr.Textbox(label="Text (100 words limitation)" if limitation else "Text", lines=5, value=example, elem_id=f"input-text-en-{name_en.replace(' ','')}")
                                        lang = gr.Dropdown(label="Language", choices=["Chinese", "Japanese", "Mix（wrap the Chinese text with [ZH][ZH], wrap the Japanese text with [JA][JA]）"],
                                                    type="index", value=language)
                                        with gr.Accordion(label="Advanced Options", open=False):
                                            symbol_input = gr.Checkbox(value=False, label="Symbol input")
                                            symbol_list = gr.Dataset(label="Symbol list", components=[input_text],
                                                                     samples=[[x] for x in hps_ms.symbols])
                                            symbol_list_json = gr.Json(value=hps_ms.symbols, visible=False)
                                        btn = gr.Button(value="Generate", variant="primary")
                                        with gr.Row():
                                            ns = gr.Slider(label="noise_scale", minimum=0.1, maximum=1.0, step=0.1, value=0.6, interactive=True)
                                            nsw = gr.Slider(label="noise_scale_w", minimum=0.1, maximum=1.0, step=0.1, value=0.668, interactive=True)
                                            ls = gr.Slider(label="length_scale", minimum=0.1, maximum=2.0, step=0.1, value=1.2 if language=="Chinese" else 1, interactive=True)
                                    with gr.Column():
                                        o1 = gr.Textbox(label="Output Message")
                                        o2 = gr.Audio(label="Output Audio", elem_id=f"tts-audio-en-{name_en.replace(' ','')}")
                                        download = gr.Button("Download Audio")
                                    btn.click(tts_fn, inputs=[input_text, lang,  ns, nsw, ls, symbol_input], outputs=[o1, o2], api_name=f"tts-{name_en}")
                                    download.click(None, [], [], _js=download_audio_js.format(audio_id=f"en-{name_en.replace(' ', '')}"))
                                    lang.change(change_lang, inputs=[lang], outputs=[ns, nsw, ls])
                                    symbol_input.change(
                                        to_symbol_fn,
                                        [symbol_input, input_text, lang],
                                        [input_text]
                                    )
                                    symbol_list.click(None, [symbol_list, symbol_list_json], [input_text],
                                                      _js=f"""
                                    (i,symbols) => {{
                                        let root = document.querySelector("body > gradio-app");
                                        if (root.shadowRoot != null)
                                            root = root.shadowRoot;
                                        let text_input = root.querySelector("#input-text-en-{name_en.replace(' ', '')}").querySelector("textarea");
                                        let startPos = text_input.selectionStart;
                                        let endPos = text_input.selectionEnd;
                                        let oldTxt = text_input.value;
                                        let result = oldTxt.substring(0, startPos) + symbols[i] + oldTxt.substring(endPos);
                                        text_input.value = result;
                                        let x = window.scrollX, y = window.scrollY;
                                        text_input.focus();
                                        text_input.selectionStart = startPos + symbols[i].length;
                                        text_input.selectionEnd = startPos + symbols[i].length;
                                        text_input.blur();
                                        window.scrollTo(x, y);
                                        return text_input.value;
                                    }}""")
                    with gr.TabItem("中文"):
                        for (sid, name_en, name_zh, title, cover, example, language,  net_g_ms, tts_fn, to_symbol_fn) in models:
                            if title.split("-")[0] != category:
                                continue
                            with gr.TabItem(name_zh):
                                with gr.Row():
                                    gr.Markdown(
                                        '<div align="center">'
                                        f'<a><strong>{title}</strong></a>'
                                        f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else ""
                                        '</div>'
                                    )
                                with gr.Row():
                                    with gr.Column():
                                        input_text = gr.Textbox(label="文本 (100字上限)" if limitation else "文本", lines=5, value=example, elem_id=f"input-text-zh-{name_zh}")
                                        lang = gr.Dropdown(label="语言", choices=["中文", "日语", "中日混合（中文用[ZH][ZH]包裹起来，日文用[JA][JA]包裹起来）"],
                                                    type="index", value="中文"if language == "Chinese" else "日语")
                                        with gr.Accordion(label="高级选项", open=False):
                                            symbol_input = gr.Checkbox(value=False, label="符号输入")
                                            symbol_list = gr.Dataset(label="符号列表", components=[input_text],
                                                                     samples=[[x] for x in hps_ms.symbols])
                                            symbol_list_json = gr.Json(value=hps_ms.symbols, visible=False)
                                        btn = gr.Button(value="生成", variant="primary")
                                        with gr.Row():
                                            ns = gr.Slider(label="控制感情变化程度", minimum=0.1, maximum=1.0, step=0.1, value=0.6, interactive=True)
                                            nsw = gr.Slider(label="控制音素发音长度", minimum=0.1, maximum=1.0, step=0.1, value=0.668, interactive=True)
                                            ls = gr.Slider(label="控制整体语速", minimum=0.1, maximum=2.0, step=0.1, value=1.2 if language=="Chinese" else 1, interactive=True)
                                    with gr.Column():
                                        o1 = gr.Textbox(label="输出信息")
                                        o2 = gr.Audio(label="输出音频", elem_id=f"tts-audio-zh-{name_zh}")
                                        download = gr.Button("下载音频")
                                    btn.click(tts_fn, inputs=[input_text, lang,  ns, nsw, ls, symbol_input], outputs=[o1, o2])
                                    download.click(None, [], [], _js=download_audio_js.format(audio_id=f"zh-{name_zh}"))
                                    lang.change(change_lang, inputs=[lang], outputs=[ns, nsw, ls])
                                    symbol_input.change(
                                        to_symbol_fn,
                                        [symbol_input, input_text, lang],
                                        [input_text]
                                    )
                                    symbol_list.click(None, [symbol_list, symbol_list_json], [input_text],
                                                      _js=f"""
                                    (i,symbols) => {{
                                        let root = document.querySelector("body > gradio-app");
                                        if (root.shadowRoot != null)
                                            root = root.shadowRoot;
                                        let text_input = root.querySelector("#input-text-zh-{name_zh}").querySelector("textarea");
                                        let startPos = text_input.selectionStart;
                                        let endPos = text_input.selectionEnd;
                                        let oldTxt = text_input.value;
                                        let result = oldTxt.substring(0, startPos) + symbols[i] + oldTxt.substring(endPos);
                                        text_input.value = result;
                                        let x = window.scrollX, y = window.scrollY;
                                        text_input.focus();
                                        text_input.selectionStart = startPos + symbols[i].length;
                                        text_input.selectionEnd = startPos + symbols[i].length;
                                        text_input.blur();
                                        window.scrollTo(x, y);
                                        return text_input.value;
                                    }}""")
            for category, link in others.items():
                with gr.TabItem(category):
                    gr.Markdown(
                        f'''
                        <center>
                          <h2>Click to Go</h2>
                          <a href="{link}">
                            <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl-dark.svg"
                          </a>
                        </center>
                        '''
                    )
    app.queue(concurrency_count=1, api_open=args.api).launch(share=args.share)
