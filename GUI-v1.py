import subprocess, torch, os, traceback, sys, warnings, shutil, numpy as np
from mega import Mega
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import threading
from time import time
from subprocess import Popen
import datetime, requests
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import soundfile as sf
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config

from utils import load_audio, CSVutil
import demucs.separate
import audiosegment

DoFormant = False
Quefrency = 1.0
Timbre = 1.0

f0_method = 'rmvpe' 
f0_up_key = 0
crepe_hop_length = 120
filter_radius = 3
resample_sr = 1
rms_mix_rate = 0.21
protect = 0.33
index_rate = 0.66

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

# essa parte excluir dps
if not os.path.isdir('csvdb/'):
    os.makedirs('csvdb')
    frmnt, stp = open("csvdb/formanting.csv", 'w'), open("csvdb/stop.csv", 'w')
    frmnt.close()
    stp.close()

try:
    DoFormant, Quefrency, Timbre = CSVutil('csvdb/formanting.csv', 'r', 'formanting')
    DoFormant = (
        lambda DoFormant: True if DoFormant.lower() == 'true' else (False if DoFormant.lower() == 'false' else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, Quefrency, Timbre)

def download_models():
    # Download hubert base model if not present
    if not os.path.isfile('./hubert_base.pt'):
        response = requests.get('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt')

        if response.status_code == 200:
            with open('./hubert_base.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded hubert base model file successfully. File saved to ./hubert_base.pt.")
        else:
            raise Exception("Failed to download hubert base model file. Status code: " + str(response.status_code) + ".")
        
    # Download rmvpe model if not present
    if not os.path.isfile('./rmvpe.pt'):
        response = requests.get('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt')

        if response.status_code == 200:
            with open('./rmvpe.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded rmvpe model file successfully. File saved to ./rmvpe.pt.")
        else:
            raise Exception("Failed to download rmvpe model file. Status code: " + str(response.status_code) + ".")

download_models()

# Check if we're in a Google Colab environment
if os.path.exists('/content/'):
    print("\n-------------------------------\nRVC v2 Easy GUI (Colab Edition)\n-------------------------------\n")

    print("-------------------------------")
        # Check if the file exists at the specified path
    if os.path.exists('/content/Mangio-RVC-Fork/hubert_base.pt'):
        # If the file exists, print a statement saying so
        print("File /content/Mangio-RVC-Fork/hubert_base.pt already exists. No need to download.")
    else:
        # If the file doesn't exist, print a statement saying it's downloading
        print("File /content/Mangio-RVC-Fork/hubert_base.pt does not exist. Starting download.")

        # Make a request to the URL
        response = requests.get('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt')

        # Ensure the request was successful
        if response.status_code == 200:
            # If the response was a success, save the content to the specified file path
            with open('/content/Mangio-RVC-Fork/hubert_base.pt', 'wb') as f:
                f.write(response.content)
            print("Download complete. File saved to /content/Mangio-RVC-Fork/hubert_base.pt.")
        else:
            # If the response was a failure, print an error message
            print("Failed to download file. Status code: " + str(response.status_code) + ".")
else:
    print("\n-------------------------------\nRVC v2 Easy GUI (Local Edition)\n-------------------------------\n")
    print("-------------------------------\nNot running on Google Colab, skipping download.")

i18n = I18nAuto()
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "A60" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

weight_root = "weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

def vc_single(
    input_audio,
    separate_vocals_bool,
    progress = gr.Progress()
):
    progress(0, desc="Preparando áudio...")
    overlay_audios_bool = False
    input_audio_path = input_audio
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    try:
        t1 = 0
        t2 = 0
        if (separate_vocals_bool):
            t1 = time()
            progress(0.1, desc="Separando vocais...")
            path_to_separated_vocals = separate_vocals(input_audio_path)
            if (path_to_separated_vocals):
                input_audio_path = path_to_separated_vocals
                overlay_audios_bool = True
            t2 = time()
        progress(0.2, desc="Carregando áudio...")
        audio = load_audio(input_audio_path, 16000, DoFormant, Quefrency, Timbre)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0, t2 - t1, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = get_index()
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        )
        progress(0.3, desc="Gerando áudio...")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            progress,
            f0_file=None,
        )
        progress(0.8, desc="Áudio convertido...")
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        if (overlay_audios_bool):
            t1 = time()
            progress(0.9, desc="Juntando vocal e instrumental...")
            (tgt_sr, audio_opt) = overlay_audios(tgt_sr, audio_opt, input_audio_path.replace("vocals", "no_vocals"))
            remove_separated_files(input_audio_path)
            t2 = time()
        times[4] = t2 - t1
        return {"visible": True, "__type__": "update", "value": "Áudio convertido com sucesso!\nTempo: %1fs" % (
            sum(times),
        )}, (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}

def update_dropdowns():
    return [change_choices(), change_choices2()]

#region RVC WebUI App
def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg','.flac','.m4a','.aac','.mp4')):
            audio_files.append(os.path.join('./audios',filename).replace('\\', '/'))
    return {"choices": sorted(audio_files), "__type__": "update"}
    
audio_files=[]
for filename in os.listdir("./audios"):
    if filename.endswith(('.wav','.mp3','.ogg','.flac','.m4a','.aac','.mp4')):
        audio_files.append(os.path.join('./audios',filename).replace('\\', '/'))
        
def get_index():
    if check_for_name() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
    return ''

def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_path
    
def save_to_wav2(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'./audios')
    return os.path.join('./audios',os.path.basename(file_path))
                
def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
            
def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model =='':
        return "You need to name your model. For example: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."

def download_from_youtube(url):
    if url == '':
        pass
    filename = subprocess.getoutput(f'yt-dlp --print filename {url} --format m4a -o "./audios/%(title)s.%(ext)s"')
    subprocess.getoutput(f'yt-dlp {url} --format m4a -o "./audios/%(title)s.%(ext)s"')
    if os.path.exists(filename):
        return filename

def find_vocals(root_directory, target_folder_name, file_name='vocals.wav'):
    for root, dirs, files in os.walk(root_directory):
        if target_folder_name in dirs:
            folder_path = os.path.join(root, target_folder_name)
            vocals_path = os.path.join(folder_path, file_name)
            if os.path.exists(vocals_path):
                return vocals_path
    return None

def separate_vocals(audio_path):
    audio_name = audio_path[9:-4]
    if (os.path.exists(audio_path) and audio_name):
        demucs.separate.main(["--two-stems", "vocals", audio_path, "-o", './audios'])
        vocals_path = find_vocals('./audios', audio_name)
        if vocals_path:
            return vocals_path
    return None

# aqui ainda não tá 100%
def overlay_audios(sample_rate, np_array, accompaniment_path):
    if (not os.path.exists(accompaniment_path)):
        return (sample_rate, np_array)
    sound1 = audiosegment.from_numpy_array(np_array, sample_rate)
    sound2 = audiosegment.from_file(accompaniment_path)
    overlay = sound1.overlay(sound2, position=0)
    return (overlay.frame_rate, overlay.to_numpy_array())

def remove_separated_files(vocals_path):
    parent_dir = os.path.dirname(vocals_path)
    try:
        shutil.rmtree(parent_dir)
        print(f"Deleted {parent_dir} folder and its contents")
    except FileNotFoundError:
        print(f"{parent_dir} folder not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def hide_output_text():
    return {"visible": False, "__type__": "update", "value": ""}

def show_selected_audio(input_audio_path):
    return input_audio_path

css = """
.padding {padding-left: 15px; padding-top: 5px;}
"""

with gr.Blocks(theme = gr.themes.Base(), title="Vocais da Loirinha 👱🏻‍♀️", css=css) as app:
    gr.HTML("<h1>Vocais da Loirinha 👱🏻‍♀️</h1>")

    gr.HTML("<h2>Como usar?</h2>")
    gr.Markdown("""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et volutpat eros. Nunc id magna vel ligula blandit ullamcorper. Proin commodo tincidunt gravida. Morbi posuere, lorem eu ornare auctor, dolor est volutpat eros, sed aliquet justo mi eu ligula. Maecenas convallis risus metus, at convallis ex gravida in. Suspendisse varius libero nec tellus placerat vulputate. Quisque ornare enim sed tristique ultrices.""")

    gr.HTML("<h2>Comece aqui!</h2>")
    with gr.Tabs():        
        with gr.TabItem("Inferência"):
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    with gr.Row():
                        model_dropdown = gr.Dropdown(label="1. Selecione a voz:", choices=sorted(names), value=check_for_name())
                        if check_for_name() != '':
                            get_vc(sorted(names)[0])
                        model_dropdown.change(
                            fn=get_vc,
                            inputs=[model_dropdown],
                            outputs=[],
                        )
                    gr.HTML("<p>2. Adicione um arquivo de áudio</p>", elem_classes="padding")
                    yt_link_textbox = gr.Textbox(label="Insira um link para uma música no Youtube:")
                    download_yt_button = gr.Button("Baixar áudio do vídeo")
                    dropbox = gr.File(label="OU selecione um arquivo:")
                    record_button = gr.Audio(source="microphone", label="OU grave o áudio:", type="filepath")
                        
                with gr.Column():
                    with gr.Row():
                        audio_dropdown = gr.Dropdown(
                            label="3. Selecione o áudio",
                            value="",
                            choices=audio_files,
                            scale=1
                        )
                        refresh_button = gr.Button("Atualizar listas de vozes e áudios", variant="primary", scale=0)
                        # Events
                        download_yt_button.click(fn=download_from_youtube, inputs=[yt_link_textbox], outputs=[audio_dropdown])
                        dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[audio_dropdown])
                        dropbox.upload(fn=change_choices2, inputs=[], outputs=[audio_dropdown])
                        record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[audio_dropdown])
                        record_button.change(fn=change_choices2, inputs=[], outputs=[audio_dropdown])
                        refresh_button.click(fn=update_dropdowns, inputs=[], outputs=[model_dropdown, audio_dropdown])
                    selected_audio = gr.Audio(label="Áudio selecionado", interactive=False)
                    audio_dropdown.select(show_selected_audio, inputs=[audio_dropdown], outputs=[selected_audio])
                    separate_checkbox = gr.Checkbox(label="Separar vocais e instrumental", 
                                                    info="Marque esta opção quando o áudio selecionado NÃO tiver a voz isolada. Os vocais serão extraídos para a conversão e depois reintegrados ao áudio final com os instrumentais. ⚠️ O tempo de conversão pode aumentar significamente com essa opção ativada.")
                    convert_button = gr.Button("Gerar áudio", variant="primary")
                    output_audio = gr.Audio(
                        label="Áudio convertido (Clique nos três pontos para fazer o download)",
                        type='filepath',
                        interactive=False,
                    )
                    output_audio_textbox = gr.Textbox(label="Resultado", interactive=False, visible=True, placeholder="Nenhum áudio gerado.")           
                    convert_button.click(hide_output_text, outputs=[output_audio_textbox]).then(vc_single, [audio_dropdown, separate_checkbox], [output_audio_textbox, output_audio])
                        
        with gr.TabItem("Adicione uma voz"):
            with gr.Column():
                model_link_textbox = gr.Textbox(label="1. Insira o link para o modelo:", info="A URL inserida deve ser o link para o download de um arquivo zip que contém o arquivo .pth. Pode ser um link do Google Drive, Mega ou Hugging Face.")
                model_name_textbox = gr.Textbox(label="2. Escolha um nome para identificar o modelo:", info="Esse nome deve ser diferente do nome dos modelos (vozes) já existentes!")
                download_button = gr.Button("Baixar modelo")
                output_download_textbox = gr.Textbox(label="Resultado", interactive=False, placeholder="Nenhum modelo baixado.")
                download_button.click(fn=download_from_url, inputs=[model_link_textbox, model_name_textbox], outputs=[output_download_textbox])
            with gr.Row():
                gr.Markdown(
                """
                Original RVC:https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
                Mangio's RVC Fork:https://github.com/Mangio621/Mangio-RVC-Fork
                ❤️ If you like the EasyGUI, help me keep it.❤️ 
                https://paypal.me/lesantillan
                """
                )
    
    if config.iscolab or config.paperspace: # Share gradio link for colab and paperspace (FORK FEATURE)
        app.queue(concurrency_count=511, max_size=1022).launch(share=True, quiet=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(share=False, quiet=True)
#endregion