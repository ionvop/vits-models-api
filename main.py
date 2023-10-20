import torch
import utils
import commons
import re
import pyopenjtalk
import json
import winsound
import os
from models import SynthesizerTrn
from torch import no_grad, LongTensor
from text import text_to_sequence
from scipy.io.wavfile import write
from unidecode import unidecode

os.chdir(os.path.dirname(os.path.realpath(__file__)))
hps_ms = utils.get_hparams_from_file("config/config.json")

def main():
    model = "keqing"
    tts = initialize(model)
    
    while True:
        std_in = input("Enter text: ")
        say(tts, std_in)
        winsound.PlaySound("output.wav", winsound.SND_FILENAME)

def say(tts, text):
    accent = japanese_to_romaji_with_accent(text)
    print(accent)
    print("Generating speech...")
    status, audio = tts(accent, "Japanese", 0.6, 0.668, 1, False)
    sample_rate, data = audio
    write("output.wav", sample_rate, data)

def initialize(model):
    device = torch.device("cpu")
    model_info = json.loads(open("pretrained_models\info.json", encoding = "utf8").read())
    sid = model_info[model]["sid"]

    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers if model_info[model]["type"] == "multi" else 0,
        **hps_ms.model)

    utils.load_checkpoint(f"pretrained_models/{model}/{model}.pth", net_g_ms, None)
    _ = net_g_ms.eval().to(device)
    tts = create_tts_fn(net_g_ms, sid)
    return tts

def create_tts_fn(net_g_ms, speaker_id):
    def tts_fn(text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")

        if not is_symbol:
            if language == 0:
                text = f"[ZH]{text}[ZH]"
            elif language == 1:
                text = f"[JA]{text}[JA]"
            else:
                text = f"{text}"
        stn_tst, clean_text = get_text(text, hps_ms, is_symbol)

        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to("cpu")
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to("cpu")
            sid = LongTensor([speaker_id]).to("cpu")
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        return "Success", (22050, audio)
    return tts_fn

def get_text(text, hps, is_symbol):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text

def japanese_to_romaji_with_accent(text):
    '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
    _japanese_characters = re.compile(r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')
    _japanese_marks = re.compile(r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = ''
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if text!='':
                text+=' '
            labels = pyopenjtalk.extract_fullcontext(sentence)
            for n, label in enumerate(labels):
                phoneme = re.search(r'\-([^\+]*)\+', label).group(1)
                if phoneme not in ['sil','pau']:
                    text += phoneme.replace('ch','ʧ').replace('sh','ʃ').replace('cl','Q')
                else:
                    continue
                n_moras = int(re.search(r'/F:(\d+)_', label).group(1))
                a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
                a2 = int(re.search(r"\+(\d+)\+", label).group(1))
                a3 = int(re.search(r"\+(\d+)/", label).group(1))
                if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil','pau']:
                    a2_next=-1
                else:
                    a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
                # Accent phrase boundary
                if a3 == 1 and a2_next == 1:
                    text += ' '
                # Falling
                elif a1 == 0 and a2_next == a2 + 1 and a2 != n_moras:
                    text += '↓'
                # Rising
                elif a2 == 1 and a2_next == 2:
                    text += '↑'
        if i<len(marks):
            text += unidecode(marks[i]).replace(' ','')
    return text

main()