import os,shutil,sys,pdb,re
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import json,yaml,warnings,torch
import platform
import psutil
import signal

warnings.filterwarnings("ignore")
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if(os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if(name=="jieba.cache"):continue
        path="%s/%s"%(tmp,name)
        delete=os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass
from tools import my_utils
import traceback
import shutil
import pdb
import gradio as gr
from subprocess import Popen
import signal
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_subfix,is_share
from scipy.io import wavfile
from tools.my_utils import load_audio
from multiprocessing import cpu_count

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # When encountering steps that MPS does not support, use CPU.

n_cpu=cpu_count()
           
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# Determine if there is an NVIDIA GPU available that can be used for training and accelerating inference.
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060"]):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # At least one usable NVIDIA GPU is available.
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
# # Determine whether MPS acceleration is supported.
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # In practice, using system memory as video memory will not cause out-of-memory errors.

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    default_batch_size = psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2
gpus = "-".join([i[0] for i in gpu_infos])

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append(name)
    return SoVITS_names,GPT_names
SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
SoVITS_names,GPT_names = get_weights_names()

def custom_sort_key(s):
    # Use regular expressions to extract the numeric and non-numeric parts of a string.
    parts = re.split('(\d+)', s)
    # Convert the numeric part to an integer, while keeping the non-numeric part unchanged.
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

p_label=None
p_asr=None
p_tts_inference=None

def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)
    

def change_label(if_label,path_list):
    global p_label
    if(if_label==True and p_label==None):
        path_list=my_utils.clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s'%(python_exec,path_list,webui_port_subfix,is_share)
        yield "Labeling tool WebUI has been opened."
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif(if_label==False and p_label!=None):
        kill_process(p_label.pid)
        p_label=None
        yield "Labeling tool WebUI has been shut down."



def change_tts_inference(if_tts,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path):
    global p_tts_inference
    if(if_tts==True and p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        os.environ["is_share"]=str(is_share)
        cmd = '"%s" GPT_SoVITS/inference_webui.py'%(python_exec)
        yield "TTS inference process has been started."
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    elif(if_tts==False and p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        yield "TTS inference process has been stopped."

from tools.asr.config import asr_dict
def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang):
    global p_asr
    if(p_asr==None):
        asr_inp_dir=my_utils.clean_path(asr_inp_dir)
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s"%("float16"if is_half==True else "float32")

        yield "ASR task started:%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr=None
        yield f"ASR task completed, check the terminal for the next step.",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "There is an ongoing ASR task. It needs to be terminated before starting the next task.",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        # return None

def close_asr():
    global p_asr
    if(p_asr!=None):
        kill_process(p_asr.pid)
        p_asr=None
    return "ASR process has been terminated",{"__type__":"update","visible":True},{"__type__":"update","visible":False}



p_train_SoVITS=None
def open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D):
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root
        data["name"]=exp_name
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        yield "SoVITS training started：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        yield "SoVITS training completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "There is an ongoing SoVITS training task. It needs to be terminated before starting the next task",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Ba():
    global p_train_SoVITS
    if(p_train_SoVITS!=None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS=None
    return "SoVITS training has been terminated",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_GPT=None
def open1Bb(batch_size,total_epoch,exp_name,if_dpo,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        yield "GPT training started:%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        yield "GPT training completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "There is an ongoing GPT training task. It needs to be terminated before starting the next task.",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Bb():
    global p_train_GPT
    if(p_train_GPT!=None):
        kill_process(p_train_GPT.pid)
        p_train_GPT=None
    return "GPT training has been terminated",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

ps_slice=[]
def open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    if(os.path.exists(inp)==False):
        yield "Input path does not exist",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if os.path.isfile(inp):n_parts=1
    elif os.path.isdir(inp):pass
    else:
        yield "The input path exists but is neither a file nor a folder",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (python_exec,inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "Slicing in progress", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps_slice:
            p.wait()
        ps_slice=[]
        yield "Slicing completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "Slicing completed", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice=[]
    return "All slicing processes have been terminated", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1a=[]
def open1a(inp_text,inp_wav_dir,exp_name,gpu_numbers,bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1a == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "bert_pretrained_dir":bert_pretrained_dir,
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield "Text processing in progress", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a=[]
        yield "Text processing completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "There is an ongoing text task. It needs to be terminated before starting the next task.", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1a():
    global ps1a
    if (ps1a != []):
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a=[]
    return "All 1a processes have been terminated", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1b=[]
def open1b(inp_text,inp_wav_dir,exp_name,gpu_numbers,ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1b == []):
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":"%s/%s"%(exp_root,exp_name),
            "cnhubert_base_dir":ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "SSL extraction process in progress", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1b:
            p.wait()
        ps1b=[]
        yield "SSL extraction process completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "All SSL extraction processes have been terminated", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b=[]
    return "All 1b processes have been terminated", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1c=[]
def open1c(inp_text,exp_name,gpu_numbers,pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if (ps1c == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "pretrained_s2G":pretrained_s2G_path,
            "s2config_path":"GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "Semantic token extraction in progress", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c=[]
        yield "Semantic token extraction completed",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "There is an ongoing semantic token extraction task. It needs to be terminated before starting the next task.", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c=[]
    return "All semantic token processes have been terminated", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc=[]
def open1abc(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "Progress：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            yield "progress：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "progress：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            for p in ps1abc:p.wait()
            yield "progress：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "progress：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "progress：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "One-click three-in-one process finished", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "One-click three-in-one process encountered an error midway", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "There is an ongoing one-click three-in-one task. It needs to be terminated before starting the next task.", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "All one-click three-in-one processes have been terminated", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=
            "This software is open-sourced under the MIT license. The author does not have any control over the software. Users of the software and those who distribute the audio generated by the software bear full responsibility. If you do not agree to these terms, you may not use or reference any code or files within the software package. See the root directory for details. <b>LICENSE</b>."
    )
    gr.Markdown(
        value=
            "Chinese Tutorial Documentation：https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e"
    )

    with gr.Tabs():
        with gr.TabItem("0-Prerequisite Dataset Acquisition Tool"):# randomly slice to prevent uvr5 from running out of memory -> uvr5 -> slicer -> asr -> labeling

            gr.Markdown(value="0b-Audio slicer")
            with gr.Row():
                with gr.Row():
                    slice_inp_path=gr.Textbox(label="Audio slicer input (file or folder)",value="")
                    slice_opt_root=gr.Textbox(label="Audio slicer output folder",value="output/slicer_opt")
                    threshold=gr.Textbox(label="Noise gate threshold (loudness below this value will be treated as noise",value="-34")
                    min_length=gr.Textbox(label="min_length:Minimum length",value="4000")
                    min_interval=gr.Textbox(label="min_interval:Minumum interval for audio cutting",value="300")
                    hop_size=gr.Textbox(label="hop_size:hop_size: FO hop size, the smaller the value, the higher the accuracy",value="10")
                    max_sil_kept=gr.Textbox(label="max_sil_kept:Maximum length for silence to be kept",value="500")
                with gr.Row():
                    open_slicer_button=gr.Button("Start audio slicer", variant="primary",visible=True)
                    close_slicer_button=gr.Button("Stop audio cutting", variant="primary",visible=False)
                    _max=gr.Slider(minimum=0,maximum=1,step=0.05,label="max:Loudness multiplier after normalized",value=0.9,interactive=True)
                    alpha=gr.Slider(minimum=0,maximum=1,step=0.05,label="aalpha_mix: proportion of normalized audio merged into dataset",value=0.25,interactive=True)
                    n_process=gr.Slider(minimum=1,maximum=n_cpu,step=1,label="CPU threads used for audio slicing",value=4,interactive=True)
                    slicer_info = gr.Textbox(label="Audio slicer output log")

            gr.Markdown(value="0c-Chinese ASR tool")
            with gr.Row():
                open_asr_button = gr.Button("Start batch ASR", variant="primary",visible=True)
                close_asr_button = gr.Button("Stop ASR task", variant="primary",visible=False)
                with gr.Column():
                    with gr.Row():
                        asr_inp_dir = gr.Textbox(
                            label="Input folder path",
                            value="D:\\GPT-SoVITS\\raw\\xxx",
                            interactive=True,
                        )
                        asr_opt_dir = gr.Textbox(
                            label       = "Output folder path",
                            value       = "output/asr_opt",
                            interactive = True,
                        )
                    with gr.Row():
                        asr_model = gr.Dropdown(
                            label       = "ASR model",
                            choices     = list(asr_dict.keys()),
                            interactive = True,
                            value="Damo ASR (Chinese)"
                        )
                        asr_size = gr.Dropdown(
                            label       = "ASR model size",
                            choices     = ["large"],
                            interactive = True,
                            value="large"
                        )
                        asr_lang = gr.Dropdown(
                            label       = "ASR language",
                            choices     = ["zh"],
                            interactive = True,
                            value="zh"
                        )
                    with gr.Row():
                        asr_info = gr.Textbox(label="ASR output log")

                def change_lang_choices(key): #Modify optional languages based on the selected model
                    # return gr.Dropdown(choices=asr_dict[key]['lang'])
                    return {"__type__": "update", "choices": asr_dict[key]['lang'],"value":asr_dict[key]['lang'][0]}
                def change_size_choices(key): # Modify optional model sizes based on the selected model
                    # return gr.Dropdown(choices=asr_dict[key]['size'])
                    return {"__type__": "update", "choices": asr_dict[key]['size']}
                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                
            gr.Markdown(value="0d-Speech to text proofreading tool")
            with gr.Row():
                if_label = gr.Checkbox(label="Open labelling WebUI",show_label=True)
                path_list = gr.Textbox(
                    label=".list annotation file path",
                    value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                    interactive=True,
                )
                label_info = gr.Textbox(label="Proofreading tool output log")
            if_label.change(change_label, [if_label,path_list], [label_info])

            open_asr_button.click(open_asr, [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang], [asr_info,open_asr_button,close_asr_button])
            close_asr_button.click(close_asr, [], [asr_info,open_asr_button,close_asr_button])
            open_slicer_button.click(open_slice, [slice_inp_path,slice_opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_process], [slicer_info,open_slicer_button,close_slicer_button])
            close_slicer_button.click(close_slice, [], [slicer_info,open_slicer_button,close_slicer_button])

        with gr.TabItem("1-GPT-SoVITS-TTS"):
            with gr.Row():
                exp_name = gr.Textbox(label="*Experiment/model name", value="xxx", interactive=True)
                gpu_info = gr.Textbox(label="GPU Information", value=gpu_info, visible=True, interactive=False)
                pretrained_s2G = gr.Textbox(label="Pretrained SoVITS-G model path", value="GPT_SoVITS/pretrained_models/s2G488k.pth", interactive=True)
                pretrained_s2D = gr.Textbox(label="Pretrained SoVITS-D model path", value="GPT_SoVITS/pretrained_models/s2D488k.pth", interactive=True)
                pretrained_s1 = gr.Textbox(label="Pretrained GPT model path", value="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", interactive=True)
            with gr.TabItem("1A-Dataset formatting"):
                gr.Markdown(value="output folder (logs/{experiment name}) should have files and folders starts with 23456.")
                with gr.Row():
                    inp_text = gr.Textbox(label="*Text labelling file", value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",interactive=True)
                    inp_wav_dir = gr.Textbox(
                        label="*Audio dataset folder",
                        # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                        interactive=True,
                        placeholder="Fill in the directory of segmented audio. The complete path of the read audio file is equal to the directory concatenated with the waveform's corresponding filename from the list file (not the full path)."
                    )
                gr.Markdown(value="1Aa-Text")
                with gr.Row():
                    gpu_numbers1a = gr.Textbox(label="GPU number is separated by -, each GPU will run one process ",value="%s-%s"%(gpus,gpus),interactive=True)
                    bert_pretrained_dir = gr.Textbox(label="Pretrained BERT model path",value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",interactive=False)
                    button1a_open = gr.Button("Start speech-to-text", variant="primary",visible=True)
                    button1a_close = gr.Button("Stop speech-to-text", variant="primary",visible=False)
                    info1a=gr.Textbox(label="Text processing output")
                gr.Markdown(value="1Ab-SSL self-supervised feature extraction")
                with gr.Row():
                    gpu_numbers1Ba = gr.Textbox(label="GPU number is separated by -, each GPU will run one process ",value="%s-%s"%(gpus,gpus),interactive=True)
                    cnhubert_base_dir = gr.Textbox(label="Pretrained SSL model path",value="GPT_SoVITS/pretrained_models/chinese-hubert-base",interactive=False)
                    button1b_open = gr.Button("Start SSL extracting", variant="primary",visible=True)
                    button1b_close = gr.Button("Stop SSL extraction", variant="primary",visible=False)
                    info1b=gr.Textbox(label="SSL output log")
                gr.Markdown(value="1Ac-semantics token extraction")
                with gr.Row():
                    gpu_numbers1c = gr.Textbox(label="GPU number is separated by -, each GPU will run one process",value="%s-%s"%(gpus,gpus),interactive=True)
                    button1c_open = gr.Button("Start semantics token extraction", variant="primary",visible=True)
                    button1c_close = gr.Button("Stop semantics token extraction", variant="primary",visible=False)
                    info1c=gr.Textbox(label="Sematics token extraction output log")
                gr.Markdown(value="1Aabc-One-click formatting")
                with gr.Row():
                    button1abc_open = gr.Button("Start one-click formatting", variant="primary",visible=True)
                    button1abc_close = gr.Button("Stop one-click formatting", variant="primary",visible=False)
                    info1abc=gr.Textbox(label="One-click formatting output")
            button1a_open.click(open1a, [inp_text,inp_wav_dir,exp_name,gpu_numbers1a,bert_pretrained_dir], [info1a,button1a_open,button1a_close])
            button1a_close.click(close1a, [], [info1a,button1a_open,button1a_close])
            button1b_open.click(open1b, [inp_text,inp_wav_dir,exp_name,gpu_numbers1Ba,cnhubert_base_dir], [info1b,button1b_open,button1b_close])
            button1b_close.click(close1b, [], [info1b,button1b_open,button1b_close])
            button1c_open.click(open1c, [inp_text,exp_name,gpu_numbers1c,pretrained_s2G], [info1c,button1c_open,button1c_close])
            button1c_close.click(close1c, [], [info1c,button1c_open,button1c_close])
            button1abc_open.click(open1abc, [inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G], [info1abc,button1abc_open,button1abc_close])
            button1abc_close.click(close1abc, [], [info1abc,button1abc_open,button1abc_close])
            with gr.TabItem("1B-Fine-tuned training"):
                gr.Markdown(value="1Ba-SoVITS training. The model is located in SoVITS_weights.")
                with gr.Row():
                    batch_size = gr.Slider(minimum=1,maximum=40,step=1,label="Batch size per GPU:",value=default_batch_size,interactive=True)
                    total_epoch = gr.Slider(minimum=1,maximum=25,step=1,label="Total epochs, do not increase to a value that is too high",value=8,interactive=True)
                    text_low_lr_rate = gr.Slider(minimum=0.2,maximum=0.6,step=0.05,label="Text model learning rate weighting",value=0.4,interactive=True)
                    save_every_epoch = gr.Slider(minimum=1,maximum=25,step=1,label="Save frequency (save_every_epoch):",value=4,interactive=True)
                    if_save_latest = gr.Checkbox(label="Save only the latest '.ckpt' file to save disk space:", value=True, interactive=True, show_label=True)
                    if_save_every_weights = gr.Checkbox(label="Save a small final model to the 'weights' folder at each save point:", value=True, interactive=True, show_label=True)
                    gpu_numbers1Ba = gr.Textbox(label="GPU number is separated by -, each GPU will run one process", value="%s" % (gpus), interactive=True)
                with gr.Row():
                    button1Ba_open = gr.Button("Start SoVITS training", variant="primary",visible=True)
                    button1Ba_close = gr.Button("Stop SoVITS training", variant="primary",visible=False)
                    info1Ba=gr.Textbox(label="SoVITS training output log")
                gr.Markdown(value="1Bb-GPT training. The model is located in GPT_weights.")
                with gr.Row():
                    batch_size1Bb = gr.Slider(minimum=1,maximum=40,step=1,label="Batch size per GPU:",value=default_batch_size,interactive=True)
                    total_epoch1Bb = gr.Slider(minimum=2,maximum=50,step=1,label="Total epochs, do not increase to a value that is too high",value=15,interactive=True)
                    if_dpo = gr.Checkbox(label="Enable DPO training (experimental feature)", value=False, interactive=True, show_label=True)
                    if_save_latest1Bb = gr.Checkbox(label="Save only the latest '.ckpt' file to save disk space:", value=True, interactive=True, show_label=True)
                    if_save_every_weights1Bb = gr.Checkbox(label="Save a small final model to the 'weights' folder at each save point:", value=True, interactive=True, show_label=True)
                    save_every_epoch1Bb = gr.Slider(minimum=1,maximum=50,step=1,label="Save frequency (save_every_epoch):",value=5,interactive=True)
                    gpu_numbers1Bb = gr.Textbox(label="GPU number is separated by -, each GPU will run one process ", value="%s" % (gpus), interactive=True)
                with gr.Row():
                    button1Bb_open = gr.Button("Start GPT training", variant="primary",visible=True)
                    button1Bb_close = gr.Button("Stop GPT training", variant="primary",visible=False)
                    info1Bb=gr.Textbox(label="GPT training output log")
            button1Ba_open.click(open1Ba, [batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D], [info1Ba,button1Ba_open,button1Ba_close])
            button1Ba_close.click(close1Ba, [], [info1Ba,button1Ba_open,button1Ba_close])
            button1Bb_open.click(open1Bb, [batch_size1Bb,total_epoch1Bb,exp_name,if_dpo,if_save_latest1Bb,if_save_every_weights1Bb,save_every_epoch1Bb,gpu_numbers1Bb,pretrained_s1],   [info1Bb,button1Bb_open,button1Bb_close])
            button1Bb_close.click(close1Bb, [], [info1Bb,button1Bb_open,button1Bb_close])
            with gr.TabItem("1C-inference"):
                gr.Markdown(value="Choose the models from SoVITS_weights and GPT_weights. The default one is a pretrain, so you can experience zero shot TTS.")
                with gr.Row():
                    GPT_dropdown = gr.Dropdown(label="GPT weight list", choices=sorted(GPT_names,key=custom_sort_key),value=pretrained_gpt_name,interactive=True)
                    SoVITS_dropdown = gr.Dropdown(label="SoVITS weight list", choices=sorted(SoVITS_names,key=custom_sort_key),value=pretrained_sovits_name,interactive=True)
                    gpu_number_1C=gr.Textbox(label="GPU number, can only input ONE integer", value=gpus, interactive=True)
                    refresh_button = gr.Button("refreshing model paths", variant="primary")
                    refresh_button.click(fn=change_choices,inputs=[],outputs=[SoVITS_dropdown,GPT_dropdown])
                with gr.Row():
                    if_tts = gr.Checkbox(label="Open TTS inference WEBUI", show_label=True)
                    tts_info = gr.Textbox(label="TTS inference webui output log")
                    if_tts.change(change_tts_inference, [if_tts,bert_pretrained_dir,cnhubert_base_dir,gpu_number_1C,GPT_dropdown,SoVITS_dropdown], [tts_info])
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_main,
        quiet=True,
    )
