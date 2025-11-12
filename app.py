import gradio as gr
import requests
import os
import shutil
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime
from queue import Queue
import subprocess
import platform
import ffmpeg
import sys
import codecs
import psutil
import time
from threading import Event
import psutil
from pynvml import *
import atexit

# 在导入gradio之前添加这些环境变量
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_IS_EVALUATION"] = "False"
#sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
#sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
task_status_dict = {}  # 用于存储任务状态
task_creation_time = {}  # 用于存储任务的固定创建时间
monitor_flag = Event()
def get_resource_usage():
    # CPU利用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # GPU利用率
    gpu_percent = "N/A"
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    util = nvmlDeviceGetUtilizationRates(handle)
    gpu_percent = f"{util.gpu}%"
    
    return f"{cpu_percent}%", gpu_percent

def start_monitoring():
    monitor_flag.set()
    while monitor_flag.is_set():
        cpu, gpu = get_resource_usage()
        yield cpu, gpu
        time.sleep(1)

def stop_monitoring():
    monitor_flag.clear()
    return "已停止", "已停止"
# 全局配置


# 全局配置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
API_URL = "http://127.0.0.1:6006/generate"
TEMP_DIR = "temp"
#获取result所在目录，这个取决于config.ini文件
result_dir=os.path.join(PARENT_DIR,"META","app_backen","code")

os.makedirs(TEMP_DIR, exist_ok=True)

# ------------------------- 语音合成部分 -------------------------
import re
REPLACE_RULES = {}  # 用于存储替换规则
CORRECTION_FILE = "念法纠正.txt"  # 固定的替换规则文件名
# 新增函数：加载替换规则
def load_replace_rules():
    """从固定文件加载替换规则"""
    global REPLACE_RULES
    REPLACE_RULES = {}
    
    try:
        file_path = os.path.join(ROOT_DIR, CORRECTION_FILE)
        if not os.path.exists(file_path):
            return f"未找到替换规则文件: {CORRECTION_FILE}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 忽略空行和注释
                    parts = [p.strip() for p in line.split()]
                    if len(parts) >= 2:
                        REPLACE_RULES[parts[0]] = parts[1]
        
        return f"成功加载 {len(REPLACE_RULES)} 条替换规则"
    except Exception as e:
        return f"加载替换规则失败: {str(e)}"
    
# 新增函数：应用替换规则
def apply_replace_rules(text):
    """应用替换规则到文本"""
    load_replace_rules()
    if not REPLACE_RULES:
        return text
    
    for original, replacement in REPLACE_RULES.items():
        text = re.sub(re.escape(original), replacement, text)
    
    return text

def load_preview_audio(speaker_name):
    if not speaker_name:
        return gr.Audio(visible=False), gr.Group(visible=False)
    # 在voices目录查找同名的MP3文件
    audio_path = os.path.join(ROOT_DIR, "voices", f"{speaker_name}.wav")
    if os.path.exists(audio_path):
        return gr.Audio(value=audio_path, visible=True)
    return gr.Audio(visible=False)

def delete_voice_model(voice_name):
    try:
        voice_dir = os.path.join(ROOT_DIR, "voices")
        pt_file = os.path.join(voice_dir, f"{voice_name}.pt")
        wav_file = os.path.join(voice_dir, f"{voice_name}.wav")
        
        if os.path.exists(pt_file):
            os.remove(pt_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)
            
        return f"音色 {voice_name} 删除成功", refresh_voice_list()
    except Exception as e:
        return f"删除失败: {str(e)}", gr.update()
def refresh_voice_list():
    voice_dir = os.path.join(ROOT_DIR, "voices")
    voice_files = []
    if os.path.exists(voice_dir):
        voice_files = [f.replace(".pt", "") for f in os.listdir(voice_dir) 
                     if f.endswith(".pt")]
    return gr.update(choices=voice_files, value=voice_files[0] if voice_files else None)

def generate_audio(tts_text, speaker,
                   emo_control_method="与音色参考音频相同",
                   vec1=0.0, vec2=0.0, vec3=0.0, vec4=0.0,
                   vec5=0.0, vec6=0.0, vec7=0.0, vec8=0.0,
                   emo_weight=0.65):
    """调用 http://127.0.0.1:6006/generate API 生成音频"""
    # 1. 基础参数校验
    if not tts_text or tts_text.strip() == "":
        return "❌ 请输入合成文本"
    if not speaker:
        return "❌ 请先选择音色"
    
    # 2. 处理文本（应用替换规则）
    processed_text = apply_replace_rules(tts_text.strip())
    
    # 3. 获取选中音色的参考音频路径（API需要的 prompt_audio 参数）
    prompt_audio_path = os.path.join(ROOT_DIR, "voices", f"{speaker}.wav")
    if not os.path.exists(prompt_audio_path):
        return f"❌ 音色「{speaker}」的音频文件丢失，请重新定制"
    
    # 4. 构造API请求参数（匹配目标API的 form-data 格式）
    data = {
        "text": processed_text,
        "max_text_tokens_per_segment": 120,
        "do_sample": "true",
        "top_p": 0.8,
        "temperature": 0.8
        # 移除API不支持的speed参数
    }

    # 情感控制相关参数（与API匹配）
    if emo_control_method == "使用情感向量控制":
        data["emo_control_method"] = 1  # API中1表示向量控制
        # 传递8维情感向量（与API参数名匹配）
        data["vec1"] = vec1
        data["vec2"] = vec2
        data["vec3"] = vec3
        data["vec4"] = vec4
        data["vec5"] = vec5
        data["vec6"] = vec6
        data["vec7"] = vec7
        data["vec8"] = vec8
    else:
        # 默认：与音色参考音频相同
        data["emo_control_method"] = 0
    print(emo_control_method,data)
    try:
        # 5. 使用with语句安全处理文件流
        with open(prompt_audio_path, "rb") as f:
            files = {
                "prompt_audio": (
                    os.path.basename(prompt_audio_path),
                    f,
                    "audio/wav"
                )
            }
            
            # 6. 发送API请求
            response = requests.post(
                "http://127.0.0.1:6006/generate",
                data=data,
                files=files,
                timeout=30000
            )
        
        # 7. 处理API响应
        if response.status_code != 200:
            raise Exception(f"API请求失败：{response.status_code}\n{response.text}")
        
        api_result = response.json()
        if api_result.get("status") != "success":
            raise Exception(f"API返回失败：{api_result.get('message', '未知错误')}")
        print(api_result)
        # 8. 下载生成的音频
        audio_relative_path = api_result.get("audio_path")
        
        if not audio_relative_path:
            raise Exception("API未返回音频路径")

        # 处理路径格式
        audio_relative_path = audio_relative_path.replace("\\", "/")
        print(audio_relative_path)
        if not audio_relative_path.startswith('/'):
            audio_relative_path = f'/{audio_relative_path}'
        audio_url = f"http://127.0.0.1:6006{audio_relative_path}"

        # 下载音频并检查状态
        audio_res = requests.get(audio_url, timeout=3000)
        if audio_res.status_code != 200:
            raise Exception(f"音频下载失败：{audio_res.status_code}")
        
        # 保存到临时目录
        temp_audio_path = os.path.join(TEMP_DIR, f"gen_{uuid.uuid4().hex[:8]}.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_res.content)
        
        return temp_audio_path
    
    except Exception as e:
        return f"❌ 音频生成失败：{str(e)}"
 
def customize_voice(prompt_wav, speaker_name):  # 移除 prompt_text 参数
    """简化版音色定制：仅将上传音频保存到voices目录"""
    # 1. 参数校验
    if not speaker_name or speaker_name.strip() == "":
        return "❌ 音色名称不能为空，请输入名称"
    if not prompt_wav or not os.path.exists(prompt_wav):
        return "❌ 请先上传参考音频（支持WAV/MP3格式）"
    
    # 2. 定义保存路径（voices目录）
    voices_dir = os.path.join(ROOT_DIR, "voices")
    os.makedirs(voices_dir, exist_ok=True)  # 确保目录存在
    target_wav_path = os.path.join(voices_dir, f"{speaker_name.strip()}.wav")
    
    try:
        # 3. 音频格式统一（转为WAV，避免后续生成报错）
        if prompt_wav.lower().endswith(".wav"):
            # 若已是WAV，直接复制
            shutil.copyfile(prompt_wav, target_wav_path)
        else:
            # 非WAV格式（如MP3），用ffmpeg转码为WAV
            (
                ffmpeg.input(prompt_wav)
                .output(target_wav_path, ac=1, ar=16000)  # 单声道16k采样率（TTS通用格式）
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        
        # 4. 生成空PT文件（兼容原有代码逻辑，避免后续加载音色报错）
        target_pt_path = os.path.join(voices_dir, f"{speaker_name.strip()}.pt")
        with open(target_pt_path, "w", encoding="utf-8") as f:
            f.write("voice_placeholder")  # 占位内容
        
        return f"✅ 音色「{speaker_name.strip()}」保存成功！\n位置：{voices_dir}"
    except Exception as e:
        return f"❌ 保存失败：{str(e)}（请确保已安装ffmpeg）"

# ------------------------- 数字人部分 -------------------------
task_queue = Queue()
def delete_video_model(folder_name):
    try:
        if folder_name == "无" or not folder_name:
            return "请选择有效的视频模型", gr.update()
            
        video_dir = os.path.join(ROOT_DIR, "result", folder_name)
        video_file = os.path.join(video_dir, f"{folder_name}.mp4")
        
        if os.path.exists(video_file):
            os.remove(video_file)
            return f"视频模型 {folder_name}.mp4 删除成功", gr.update(choices=get_result_folders())
        return f"未找到视频文件 {folder_name}.mp4", gr.update()
    except Exception as e:
        return f"删除失败: {str(e)}", gr.update()
#获取上传定制过的模特列表
def get_result_folders():
    result_dir = os.path.join(ROOT_DIR, "result")
    if not os.path.exists(result_dir):
        return ["None"]
    
    valid_folders = []
    for folder in os.listdir(result_dir):
        folder_path = os.path.join(result_dir, folder)
        if os.path.isdir(folder_path):
            # 检查文件夹中是否存在与文件夹同名的视频文件
            video_file = os.path.join(folder_path, f"{folder}.mp4")
            if os.path.exists(video_file):
                valid_folders.append(folder)
    
    return ["None"] + valid_folders

#获取上传视频的信息，分辨率，帧速率，音频采样率等
def get_video_metadata(video_path):
    """增强版元数据获取（包含音频声道信息）"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        if not video_stream:
            raise ValueError("未检测到视频流")
            
        # 获取音频声道信息
        audio_channels = None
        if audio_stream:
            audio_channels = int(audio_stream.get('channels', 2))
            channel_layout = audio_stream.get('channel_layout', 'stereo' if audio_channels > 1 else 'mono')

        return {
            "width": int(video_stream.get('width', 0)),
            "height": int(video_stream.get('height', 0)),
            "bitrate": int(video_stream.get('bit_rate', 0)) // 1000 if video_stream.get('bit_rate') else None,
            "framerate": eval(video_stream['avg_frame_rate']) if 'avg_frame_rate' in video_stream else None,
            "audio_sample_rate": int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
            "audio_channels": audio_channels,  # 新增声道数
            "channel_layout": channel_layout,  # 新增声道布局
            "codec": video_stream.get('codec_name'),
        }
        
    except Exception as e:
        print(f"获取元数据失败: {str(e)}")
        return None
#处理
def reprocess_video(input_path, reference_path):

    if input_path is None:
        return None
    metadata = get_video_metadata(reference_path)
    if not metadata:
        return input_path

    output_path = os.path.splitext(input_path)[0] + "_adjusted.mp4"
    
    try:
        # 音频参数设置
        audio_args = {
            'c:a': 'aac',
            'ar': metadata.get("audio_sample_rate", 44100),
            'ac': metadata.get("audio_channels", 2),  # 关键修改：设置声道数
            'channel_layout': metadata.get("channel_layout", 'stereo')  # 设置声道布局
        }

        # 视频参数设置（保持不变）
        video_args = {
            'c:v': 'libx264',
            'vf': f'scale={metadata["width"]}:{metadata["height"]}',
            'r': metadata.get("framerate", 30),
            'x264-params': 'nal-hrd=cbr:force-cfr=1',
            'preset': 'medium'
        }

        # 比特率控制
        if metadata.get("bitrate"):
            target_bitrate = metadata["bitrate"]
            video_args.update({
                'b:v': f'{target_bitrate}k',
                'maxrate': f'{target_bitrate}k',
                'minrate': f'{target_bitrate}k',
                'bufsize': f'{target_bitrate}k'
            })
        
        # 执行转码
        (
            ffmpeg.input(input_path)
            .output(output_path, **video_args, **audio_args)
            .overwrite_output()
            .run()
        )
        
        print(f"sucess: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        print("FFmpeg wrong:", e.stderr.decode())
    except Exception as e:
        print("fail:", str(e))
    
    return input_path

#打开输出文件夹 
def open_output_folder():
    try:
        output_dir = os.path.abspath("result")
        print(output_dir) 
        # 确保文件夹存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 更可靠的Windows打开方式
        if platform.system() == "Windows":
            # 方法1：使用explorer.exe（最可靠）
            subprocess.Popen(f'explorer "{output_dir}"', shell=True)
            
            # # 方法2：备用方案（如果方法1失败）
            # try:
            #     os.startfile(output_dir)
            # except:
            #     subprocess.run(['start', output_dir], shell=True)
                
            return f"成功打开文件夹：{output_dir}"
    except Exception as e:
        return f"打开失败：{str(e)}"

    
#加载选中的视频
def load_selected_video(folder_name):
    if not folder_name:
        return None
    video_filename = f"{folder_name}.mp4"
    video_path = os.path.join(ROOT_DIR,"result", folder_name, video_filename)
    if os.path.exists(video_path):
        return os.path.abspath(video_path)
    return None

#调用接口合成视频
import re  # 确保导入re模块

def synthesize_video(video_path, audio_path):
    task_code = str(uuid.uuid4())
    print(f"音频路径: {audio_path}, 视频路径: {video_path}")
    
    payload = {
        "audio_url": audio_path,
        "video_url": video_path,
        "code": task_code,
        "chaofen": 0,
        "watermark_switch": 0,
        "pn": 1
    }
    response = requests.post("http://127.0.0.1:8383/easy/submit", json=payload)
   
    if response.status_code == 200:
        print(f"任务提交成功，任务代码: {task_code}")
        
        while True:
            progress_response = requests.get(f"http://127.0.0.1:8383/easy/query?code={task_code}")
            progress_data = progress_response.json()
            print(f"任务进度: {progress_data}")
            
            status = progress_data.get("data", {}).get("status")
            if status == 2:
                result_path = progress_data.get("data", {}).get("result")
                if result_path:
                    # 1. 处理返回的路径字符串
                    result_path = result_path.replace("\\", "/")  # 将反斜杠转为正斜杠
                    result_path = re.sub(r'//+', '/', result_path)  # 合并连续斜杠
                    
                    # 2. 处理相对路径前缀 "./"
                    if result_path.startswith("./"):
                        result_path = result_path[2:]  # 移除 "./" 前缀
                    
                    # 3. 拼接正确的URL
                    if result_path.startswith('/'):
                        video_url = f"http://127.0.0.1:8383{result_path}"
                    else:
                        video_url = f"http://127.0.0.1:8383/{result_path}"
                    
                    print(f"正确的视频URL: {video_url}")
                    return video_url  # 返回完整URL而非本地路径
                else:
                    return None
            elif status == 1:
                print(f"任务进行中，进度: {progress_data.get('data', {}).get('progress')}%")
            else:
                return None
            time.sleep(5)
    else:
        return None
    
#预处理视频和音频，将任务添加到队列
def save_files(video, audio_folder, audio=None):
    if video is None:
        return "请上传视频或选择视频文件", [],""
    
    video_name = Path(video).stem
    inputvideo_dir = os.path.join("result", video_name)
    os.makedirs(inputvideo_dir, exist_ok=True)
    
    vidoe_id = str(uuid.uuid4())
    video_filename = os.path.basename(video)
    video_dest = os.path.join(TEMP_DIR, f"{vidoe_id}.mp4")
    video_path = os.path.join(inputvideo_dir, video_filename)
    shutil.copy(video, video_dest)
    shutil.copy(video, video_path)

    if audio is not None:
        audio_id = str(uuid.uuid4())
        audio_dest = os.path.join(TEMP_DIR, f"{audio_id}.mp3")
        shutil.copy(audio, audio_dest)
        task_id = f"{vidoe_id}_{audio_id}"
        
        if lang=="en":
            task_status_dict[task_id] = "waiting"
        if lang=="zh-TW":
            task_status_dict[task_id] = "waiting"
        if lang=="zh-CN":
            task_status_dict[task_id] = "waiting"
        task_queue.put((vidoe_id, audio_id, inputvideo_dir))
    
    if audio_folder is not None and audio_folder.strip() != "":
        for audio_file in os.listdir(audio_folder):
            if audio_file.endswith(".mp3") or audio_file.endswith(".wav"):
                audio_id = str(uuid.uuid4())
                audio_dest = os.path.join(TEMP_DIR, f"{audio_id}.mp3")
                audio_path = os.path.join(audio_folder, audio_file)
                shutil.copy(audio_path, audio_dest)
                task_id = f"{vidoe_id}_{audio_id}"
                print("lang：",lang)
                if lang=="en":
                    task_status_dict[task_id] = "waiting"
                if lang=="zh-TW":
                    task_status_dict[task_id] = "waiting"
                if lang=="zh-CN":
                    task_status_dict[task_id] = "waiting"
                
                task_queue.put((vidoe_id, audio_id, inputvideo_dir))

    return "任务已添加到队列，请等待处理"
#按照队列排队生成

def process_queue():
    while True:
        if not task_queue.empty():
            vidoe_id, audio_id, inputvideo_dir = task_queue.get()
            task_id = f"{vidoe_id}_{audio_id}"
            # 设置任务状态为处理中
            for lang_code in ["en", "zh-TW", "zh-CN"]:
                if lang == lang_code:
                    task_status_dict[task_id] = "processing"
            
            # 记录任务创建时间
            if task_id not in task_creation_time:
                task_creation_time[task_id] = datetime.now()
            
            tempvideo_path = os.path.abspath(os.path.join(TEMP_DIR, f"{vidoe_id}.mp4"))
            tempaudio_path = os.path.abspath(os.path.join(TEMP_DIR, f"{audio_id}.mp3"))
            
            try:
                # 获取视频URL
                video_url = synthesize_video(tempvideo_path, tempaudio_path)
                print("视频URL：", video_url)
                
                if not video_url:
                    task_status_dict[task_id] = "failed: 未获取到视频URL"
                    task_queue.task_done()
                    continue
                
                # 从URL下载视频到临时文件
                temp_result_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.mp4")
                response = requests.get(video_url, stream=True, timeout=60)
                response.raise_for_status()  # 检查请求是否成功
                
                with open(temp_result_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 处理下载的视频文件
                result_path = os.path.join(inputvideo_dir, f"{audio_id}_output.mp4")
                print("结果文件夹：", result_path)
                
                # 复制结果文件
                shutil.copy(temp_result_path, result_path)
                
                # 后处理视频
                result_path_final = reprocess_video(result_path, tempvideo_path)
                
                # 添加到生成的视频列表
                if not hasattr(app, "generated_videos"):
                    app.generated_videos = []
                app.generated_videos.append(result_path_final)
                print(app.generated_videos)
                
                # 更新任务状态
                if not hasattr(app, "task_status"):
                    app.task_status = ""
                status_msg = f"task done：{result_path_final}\n"
                for lang_code in ["en", "zh-TW", "zh-CN"]:
                    if lang == lang_code:
                        app.task_status += status_msg
                
                print(app.task_status)
                
                # 删除临时文件
                for file_path in [tempvideo_path, tempaudio_path, temp_result_path, result_path]:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"删除临时文件 {file_path} 失败: {str(e)}")
                
                # 更新任务状态为完成
                for lang_code in ["en", "zh-TW", "zh-CN"]:
                    if lang == lang_code:
                        task_status_dict[task_id] = "Done"
                
            except requests.exceptions.RequestException as e:
                task_status_dict[task_id] = f"failed: 视频下载失败 - {str(e)}"
                print(f"视频下载出错: {str(e)}")
            except Exception as e:
                task_status_dict[task_id] = f"failed: {str(e)}"
                print(f"任务处理出错: {str(e)}")
            finally:
                task_queue.task_done()
        else:
            time.sleep(1)

def cleanup_temp_files(path):
    """清理所有临时音视频文件"""
    TEMP_DIR1=path
    print(TEMP_DIR1)
    try:
        if os.path.exists(TEMP_DIR1):
            for filename in os.listdir(TEMP_DIR1):
                file_path = os.path.join(TEMP_DIR1, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"删除临时文件失败 {file_path}: {e}")
        print("临时文件清理完成")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")

def get_task_status():
    status_html = """
    <style>
        .task-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .task-table th, .task-table td {
            border: 1px solid #444;
            padding: 8px 12px;
            text-align: left;
        }
        .task-table th {
            background-color: #333;
        }
        .task-waiting {
            color: #FFA500;
        }
        .task-processing {
            color: #1E90FF;
        }
        .task-completed {
            color: #32CD32;
        }
        .task-failed {
            color: #FF4500;
        }
    </style>
    <table class="task-table">
        <tr>
            <th>任务ID</th>
            <th>状态</th>
            <th>创建时间</th>
        </tr>
    """
    
    for task_id, status in task_status_dict.items():
        status_class = f"task-{status.lower()}"
        short_id = task_id[:8] + "..." + task_id[-4:]  # 缩短显示的ID
        
        # 获取固定的创建时间，如果没有记录则使用当前时间（兼容旧任务）
        create_time = task_creation_time.get(task_id, datetime.now())
        
        status_html += f"""
        <tr>
            <td>{short_id}</td>
            <td class="{status_class}">{status}</td>
            <td>{create_time.strftime('%Y-%m-%d %H:%M:%S')}</td>
        </tr>
        """
    
    status_html += "</table>"
    return status_html
# 启动任务处理线程
threading.Thread(target=process_queue, daemon=True).start()
# 在 with gr.Blocks() 之前添加自定义 CSS
def get_task_status_en():
    status_html = """
    <style>
        .task-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .task-table th, .task-table td {
            border: 1px solid #444;
            padding: 8px 12px;
            text-align: left;
        }
        .task-table th {
            background-color: #333;
        }
        .task-waiting {
            color: #FFA500;
        }
        .task-processing {
            color: #1E90FF;
        }
        .task-completed {
            color: #32CD32;
        }
        .task-failed {
            color: #FF4500;
        }
    </style>
    <table class="task-table">
        <tr>
            <th>Task ID</th>
            <th>Status</th>
            <th>Created At</th>
        </tr>
    """
    
    for task_id, status in task_status_dict.items():
        status_class = f"task-{status.lower()}"
        short_id = task_id[:8] + "..." + task_id[-4:]  # Shorten displayed ID
        
        # Get creation time, use current time if not recorded (for backward compatibility)
        create_time = task_creation_time.get(task_id, datetime.now())
        
        # Translate status if needed (optional)
        status_text = {
            "waiting": "Waiting",
            "processing": "Processing",
            "completed": "Completed",
            "failed": "Failed"
        }.get(status.lower(), status)
        
        status_html += f"""
        <tr>
            <td>{short_id}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{create_time.strftime('%Y-%m-%d %H:%M:%S')}</td>
        </tr>
        """
    
    status_html += "</table>"
    return status_html

custom_css="""
/* 隐藏 Gradio 页脚 */
footer {
    display: none !important;
}
:root {
    /* 基础颜色 */
    --body-background-fill: #252525 !important;  /* 最底层背景 */
    --block-background-fill: #252525 !important;  /* 卡片背景 */
    --input-background-fill: #252525 !important;  /* 输入框背景 */
    
    /* 文字颜色 */
    --body-text-color: #888888 !important;
    --block-title-text-color: #888888 !important;
    --label-text-color: #888888 !important;
    
    /* 边框和交互元素 */
    --border-color-primary: #3a3a3a !important;
    --button-primary-background-fill: #4a8cff !important;
    --slider-color: #4a8cff !important;
    
    /* 特殊组件 */
    --checkbox-label-text-color: #c0c0c0 !important;
    --label-text-color: #c0c0c0 !important;  /* 主标签颜色 */
    --block-label-text-color: #252525 !important;  /* 区块标签颜色 */
    --primary-btn-color: #6a75ff;
    --primary-btn-hover: #5d68f0;
}
/* ===== 强制覆盖所有标签类型 ===== */
.gr-form > .gr-form-group > label,          /* 常规输入标签 */
.gr-input > label,                          /* 输入框标签 */
.gr-slider > label,                         /* 滑块标签 */
.gr-radio > label,                          /* 单选标签 */
.gr-checkbox > label,                       /* 多选标签 */
.gr-file > label,                           /* 文件上传标签 */
.gr-audio > label,                          /* 音频标签 */
.gr-video > label,                          /* 视频标签 */
.gr-image > label,                          /* 图片标签 */
.gr-plot > label,                           /* 图表标签 */
.gr-dataframe > label,                      /* 数据框标签 */
.gr-json > label,                           /* JSON标签 */
.gr-html > label,                           /* HTML标签 */
.gr-markdown > .label {                     /* Markdown区域标签 */
    color: #1a1a1a !important;
    font-weight: 500;
}
button.primary {
    background: var(--primary-btn-color) !important;
    border-color: var(--primary-btn-color) !important;
}

button.primary:hover {
    background: var(--primary-btn-hover) !important;
}

button.primary:active {
    filter: brightness(90%);
}

/* 表格标签 */
.gr-table th {
    color: #c0c0c0 !important;
}
/* 音频组件定制 */
.audio-container {
    background: linear-gradient(180deg, #1a1a1a 0%, #333333 100%) !important;
    border-radius: 8px !important;
}

/* 视频组件定制 */
.video-container {
    background: linear-gradient(180deg, #1a1a1a 0%, #333333 100%) !important;
}

/* 标签组 */
.gr-group {
    background: #252525 !important;
    border: 1px solid #3a3a3a !important;
}
.gr-audio label,
.gr-video label {
    display: none !important;
}
/* 覆盖所有文本 */
* {
    color: #e0e0e0 !important;
}
.custom-btn {
    background: #6a75ff !important;
    border-color: #6a75ff !important;
    color: white !important;
}

.custom-btn:hover {
    background: #5d68f0 !important;
    border-color: #5d68f0!important;
}

.custom-btn:active {
    background: #2f35c7 !important;
}
/* 下拉菜单展开后的容器背景 */


.custom-dropdown li {
    transition: background-color 0.3s ease;
}
.custom-textbox{
    background: linear-gradient(180deg, #1a1a1a 0%, #333333 100%) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.gradio-dropdown .gradio-dropdown-options .gradio-dropdown-option {
    color: #000 !important; /* 设置为黑色 */
}
.tabs {
    padding: 0px !important; /* 设置选项卡的内边距 */
    font-size: 16px !important; /* 设置选项卡的字体大小 */
}
.tab_buttun{
    font-size: 20px !important; 
}
/* 下拉输入框 */
[role="combobox"], 
[role="listbox"] {
  background: #252525 !important;
  border-color: var(--border-color-primary) !important;
}

/* 下拉选项面板 */
[role="listbox"] > div {
  background: #2525FF !important;
  border: 1px solid var(--border-color-primary) !important;
  box-shadow: var(--shadow-drop-lg) !important;
}

/* 单个选项 */
[role="option"] {
  color: var(--body-text-color) !important;
  padding: 8px 12px !important;
}

/* 悬停选项 */
[role="option"]:hover {
  background: #252525 !important;
  color: white !important;
}

/* 选中选项 */
[role="option"][aria-selected="true"] {
  background: #252525 !important;
}
.tab-container[role="tablist"] {
    gap: 30px !important; /* 调整标签间距 */
  
}
.tab-container[role="tablist"] button.svelte-1tcem6n {
    font-family: "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important; /* 中等加粗 */
    letter-spacing: 0.5px !important; /* 轻微字距 */
    text-transform: uppercase !important; /* 英文大写 */
    padding: 12px 24px !important;
    color:#888888 !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
/* 悬停状态的tab标签 */
.tab-container[role="tablist"] button.svelte-1tcem6n:hover {
    background: #2d2d2d !important;

}
.tab-container[role="tablist"] button.svelte-1tcem6n.selected {
    background: transparent !important;
    color: inherit !important;
}
.tab-container[role="tablist"] button.svelte-1tcem6n.selected::after {
    content: "";
    position: absolute;
    bottom: -15px; /* 对齐容器底部内边距 */
    left: 0;
    right: 0;
    height: 8px;
    background: #6a75ff !important; /* 灰色横条 */
    border-radius: 2px;
}
/* 隐藏倍速按钮 */
.custom-audio-preview button.playback.icon.svelte-ije4bl {
    display: none !important;
}
.scroll[part="scroll"] {
  overflow-x: hidden !important; /* 禁用水平滚动 */

}
.gradio-container {
    background: #252525 !important;
}
.styler.svelte-1nguped {
    background-color: #252525 !important;
    /* 其他样式... */
}
.form {
    gap: 10px !important;
}

.gr-group {
    gap: 10px !important;
}

.column.gap {
    gap: 12px !important;
}

.row.unequal-height {
    gap: 16px !important;
}

.gradio-container * {
    border-radius: 5px !important;
}
/* 修改所有block元素的边框为隐藏 */
.block.svelte-5y6bt2 {
    border-style: none !important;
}

/* 修改特定组件的边框为隐藏 */
.gr-group.svelte-1nguped .styler {
    --block-border-width: 0px !important;
}

/* 修改表单元素的边框为隐藏 */
.form.svelte-633qhp {
    border-style: none !important;
}

/* 修改标签容器的边框为隐藏 */
.label.svelte-p5q82i {
    border-style: none !important;
}
.gradio-container.gradio-container-5-4-0 .contain .gr-group  {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.tabitem.svelte-tcemt9 {
    padding-top: 10px !important;
}
#component-42.block.custom-audio.svelte-5y6bt2 {
    background-color: #303030 !important;
}
#component-54.block.svelte-5y6bt2.padded {
    background-color: #303030 !important;
}
#component-49.block.custom-dropdown.svelte-5y6bt2 {
    background-color: linear-gradient(180deg, #1a1a1a 0%, #333333 100%) !important;

}
#component-68.block.custom-gallery.svelte-5y6bt2 {
    background-color: #303030 !important;
}
#component-134.block.svelte-5y6bt2.padded {
    background-color: #303030 !important;
}
#component-122.block.custom-audio.svelte-5y6bt2 {
    background-color: #303030 !important;
}
#component-148.block.custom-gallery.svelte-5y6bt2 {
    background-color: #303030 !important;
}
#component-214.block.svelte-5y6bt2.padded {
    background-color: #303030 !important;
}
#component-202.block.custom-audio.svelte-5y6bt2 {
    background-color: #303030 !important;
}
#component-228.block.custom-gallery.svelte-5y6bt2 {
    background-color: #303030 !important;
}
/* 容器样式 */
#component-169 .wrap.svelte-12ioyct {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  position: relative;
  font-size: 0;
}

/* 图标样式（正确层级） */
#component-169 .wrap.svelte-12ioyct > .icon-wrap.svelte-12ioyct {
  font-size: initial;
  margin-bottom: 8px;
}

/* 三行文本（使用::after） */
#component-169 .wrap.svelte-12ioyct::after {
  content: "Drop audio here\A - or -\A Click to upload";
  white-space: pre;
  font-size: 20px;
  line-height: 1.5;
  text-align: center;
  display: block;
  margin-top: 4px;
}

/* 隐藏原始文本 */
#component-169 .wrap.svelte-12ioyct > :not(.icon-wrap),
#component-211 .or.svelte-12ioyct {
  display: none;
}

/* 容器样式 */
#component-211 .wrap.svelte-12ioyct {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  position: relative;
  font-size: 0;
}

/* 图标样式（正确层级） */
#component-211 .wrap.svelte-12ioyct > .icon-wrap.svelte-12ioyct {
  font-size: initial;
  margin-bottom: 8px;
}

/* 三行文本（使用::after） */
#component-211 .wrap.svelte-12ioyct::after {
  content: "Drop video here\A - or -\A Click to upload";
  white-space: pre;
  font-size: 20px;
  line-height: 1.5;
  text-align: center;
  display: block;
  margin-top: 4px;
}

/* 隐藏原始文本 */
#component-211 .wrap.svelte-12ioyct > :not(.icon-wrap),
#component-211 .or.svelte-12ioyct {
  display: none;
}

/* 容器样式 */
#component-217 .wrap.svelte-12ioyct {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  position: relative;
  font-size: 0;
}

/* 图标样式（正确层级） */
#component-217 .wrap.svelte-12ioyct > .icon-wrap.svelte-12ioyct {
  font-size: initial;
  margin-bottom: 8px;
}

/* 三行文本（使用::after） */
#component-217 .wrap.svelte-12ioyct::after {
  content: "Drop audio here\A - or -\A Click to upload";
  white-space: pre;
  font-size: 20px;
  line-height: 1.5;
  text-align: center;
  display: block;
  margin-top: 4px;
}

/* 隐藏原始文本 */
#component-217 .wrap.svelte-12ioyct > :not(.icon-wrap),
#component-217 .or.svelte-12ioyct {
  display: none;
}
/* 情感控制区域整体样式 */
.emotion-control-section {
    margin: 15px 0;
}

/* Radio组件样式 */
#component-emo-control-method {
    background: #252525;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
}

#component-emo-control-method label {
    color: #e0e0e0 !important;
    font-size: 15px !important;
    margin-bottom: 8px !important;
    display: block !important;
}

#component-emo-control-method .gr-radio-group {
    display: flex !important;
    gap: 15px !important;
    flex-wrap: wrap !important;
}

#component-emo-control-method input[type="radio"] {
    margin-right: 6px !important;
    accent-color: #6a75ff !important;
}

#component-emo-control-method .gr-radio-label {
    color: #d0d0d0 !important;
    font-size: 14px !important;
    cursor: pointer !important;
}

/* 情感向量组容器样式 */
.gr-group:has(> .markdown-body h3) {
    background: #252525 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 6px !important;
    padding: 15px !important;
    margin: 10px 0 !important;
}

/* 情感向量标题样式 */
.markdown-body h3:contains("情感向量调节") {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #e0e0e0 !important;
    margin: 0 0 12px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #3a3a3a !important;
}

/* 滑块通用样式 */
.gr-slider {
    margin: 10px 0 !important;
}

.gr-slider label {
    color: #e0e0e0 !important;
    font-size: 14px !important;
    margin-bottom: 5px !important;
    display: block !important;
}

.gr-slider input[type="range"] {
    width: 100% !important;
    background: #3a3a3a !important;
    height: 6px !important;
    border-radius: 3px !important;
}

.gr-slider input[type="range"]::-webkit-slider-thumb {
    background: #6a75ff !important;
    border: none !important;
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
    cursor: pointer !important;
}

/* 情感权重滑块容器 */
.gr-row:has(> .gr-slider label:contains("情感权重")) {
    background: #252525 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 6px !important;
    padding: 12px 15px !important;
    margin: 5px 0 !important;
}
/* 情感控制选项背景透明 */


/* 情感向量调节区域缩小 */
.emotion-vector-section {
    margin: 10px 0 !important;
    padding: 8px !important;
}

/* 情感向量标题缩小 */
.emotion-vector-title {
    font-size: 14px !important;
    margin: 0 0 8px 0 !important;
    padding-bottom: 5px !important;
}
label.selected.svelte-1bx8sav.svelte-1bx8sav.svelte-1bx8sav {
    /* background: var(--checkbox-label-background-fill-selected); */
    color: var(--checkbox-label-text-color-selected);
    border-color: var(--checkbox-label-border-color-selected);
}
label.svelte-1bx8sav.svelte-1bx8sav.svelte-1bx8sav {
    display: flex
;
    align-items: center;
    transition: var(--button-transition);
    cursor: pointer;
    box-shadow: var(--checkbox-label-shadow);
    border: var(--checkbox-label-border-width) solid var(--checkbox-label-border-color);
    border-radius: var(--checkbox-border-radius);
    background: transparent;
    padding: var(--checkbox-label-padding);
    color: var(--checkbox-label-text-color);
    font-weight: var(--checkbox-label-text-weight);
    font-size: var(--checkbox-label-text-size);
    line-height: var(--line-md);
}
"""


    


# ------------------------- 主界面 -------------------------
def create_chinese_simplified_block():
    with gr.Blocks(title="CosyVoice 数字人系统") as demo:
        with gr.Tabs(elem_classes="tab_buttun") as tabs:
            # 第一页：音色定制
            with gr.TabItem("🎙️ 音色定制", id="tab1",elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传参考音频",elem_classes="Markdown")
                        gr.Markdown("音频时长35s以内",elem_classes="Markdown")
                        prompt_wav = gr.Audio(
                            show_label=False,
                            type="filepath",
                            interactive=True,
                            elem_classes="custom-audio"  # 添加 CSS 类
                        )
                        gr.Markdown("### 设置音色参数")
                        with gr.Group(elem_classes="custom-group"):
                            speaker_name = gr.Textbox(
                                label="音色名称", 
                                placeholder="为您的音色起个名字",
                                info="建议使用英文命名",
                                elem_classes="custom-textbox"  # 添加 CSS 类
                            )
                        customize_btn = gr.Button(
                            "✨ 开始定制音色", 
                            variant="primary",
                            elem_classes="custom-btn"  # 添加 CSS 类
                        )

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 定制结果")
                        customize_output = gr.Textbox(
                            label="状态信息",
                            interactive=False,
                            placeholder="等待音色定制...",
                            elem_classes="custom-textbox"  # 添加 CSS 类
                        )

            # 第二页：语音合成
            with gr.TabItem("🔊 语音合成", id="tab2",elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 选择音色与文本")
                        with gr.Row():
                            voice_dir = os.path.join(ROOT_DIR, "voices")
                            voice_files = []
                            if os.path.exists(voice_dir):
                                voice_files = [f.replace(".pt", "") for f in os.listdir(voice_dir) if f.endswith(".pt")]
                            with gr.Column(scale=1):
                                with gr.Row():
                                    speaker = gr.Dropdown(
                                        label="选择定制音色", 
                                        choices=voice_files,
                                        value=voice_files[0] if voice_files else None,
                                        interactive=True,
                                        elem_classes="custom-dropdown"  # 添加 CSS 类
                                    )
                                    preview_audio = gr.Audio(
                                        show_label=False,
                                        interactive=False,
                                        visible=False,
                                        elem_classes="custom-audio-preview" 
                                    )
                                    with gr.Column():
                                        refresh_btn = gr.Button("🔄 刷新列表", size="sm", elem_classes="custom-btn")  # 添加 CSS 类
                                        delete_voice_btn = gr.Button("🗑️ 删除音色", size="sm", variant="stop", elem_classes="custom-btn") 

                        tts_text = gr.Textbox(
                            label="输入合成文本", 
                            placeholder="请输入要转换为语音的文字内容...",
                            lines=14,
                            elem_classes="custom-textbox"  # 添加 CSS 类
                        )
                        # -------------------------- 新增：情感向量控制UI --------------------------
                        # 1. 情感控制方式选择（只保留「与音色参考音频相同」「使用情感向量控制」两个选项）
                        emo_control_method = gr.Radio(
                            label="情感控制方式",
                            choices=["与音色参考音频相同", "使用情感向量控制"],
                            value="与音色参考音频相同",  # 默认不启用情感向量
                            interactive=True
                        )

                        # 2. 情感向量滑块组（默认隐藏，仅当选择「使用情感向量控制」时显示）
                        # 替换原有的情感向量相关代码
                        with gr.Group(visible=False) as emotion_vector_group:
                            gr.Markdown("### 情感向量调节（8维度）", elem_classes="emotion-vector-title")
                            with gr.Row(elem_classes="emotion-vector-section"):
                                with gr.Column(scale=1):
                                    vec1 = gr.Slider(label="喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec2 = gr.Slider(label="怒", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec3 = gr.Slider(label="哀", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec4 = gr.Slider(label="惧", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                with gr.Column(scale=1):
                                    vec5 = gr.Slider(label="厌恶", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec6 = gr.Slider(label="低落", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec7 = gr.Slider(label="惊喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec8 = gr.Slider(label="平静", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

                        # 3. 情感权重（默认隐藏，控制情感向量影响程度）
                        with gr.Row(visible=False) as emo_weight_group:
                            emo_weight = gr.Slider(label="情感权重", minimum=0.0, maximum=1.0, value=0.65, step=0.01)
                        # --------------------------------------------------------------------------
                        with gr.Row():
                            generate_btn = gr.Button("🎵 生成音频", variant="primary")  # 添加 CSS 类
                            go_to_digital_human_btn = gr.Button("➡️ 前往数字人合成", variant="secondary", elem_classes="custom-btn")  # 添加 CSS 类

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 生成结果")
                        output_audio = gr.Audio(
                            show_label=False,
                            interactive=False,
                            waveform_options={
                                "waveform_progress_color": "#4a8cff"
                            },
                            elem_classes="custom-audio",  # 添加 CSS 类
                            show_download_button=True
                        )
                        gr.Examples(
                            examples=["你好，欢迎使用语音克隆系统", "今天天气真好"],
                            inputs=[tts_text],
                            label="试试示例文本",
                        
                        )

            # 第三页：数字人合成
            with gr.TabItem("🎬 数字人合成", id="tab3",elem_classes="tabs", visible=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 选择素材")
                        with gr.Group(elem_classes="custom-group"):
                            model_dir = os.path.join(ROOT_DIR, "result")
                            #print(model_dir)
                            folders=get_result_folders()
                            
                            dropdown = gr.Dropdown(
                                choices= folders, 
                                label="选择已有视频模板",
                                info="或上传新视频",
                                elem_classes="custom-dropdown"  # 添加 CSS 类
                            )
                            delete_video_btn = gr.Button("🗑️ 删除模板", variant="stop", elem_classes="custom-btn")  # 新增删除按钮
                            video_input = gr.Video(
                                show_label=False,
                                sources=["upload"],
                                format="mp4",
                                elem_classes="custom-video"  # 添加 CSS 类
                            )
                        
                        gr.Markdown("### 添加音频")
                        with gr.Accordion("批量生成选项", open=False):
                            audio_folder_input = gr.Textbox(
                                label="音频文件夹路径",
                                placeholder="输入包含多个音频文件的文件夹路径",
                                elem_classes="custom-textbox"  # 添加 CSS 类
                            )
                        single_audio_input = gr.Audio(
                            show_label=False,
                            type="filepath",
                            elem_classes="custom-audio"  # 添加 CSS 类
                        )
                        save_button = gr.Button("🚀 生成数字人视频", variant="primary", elem_classes="custom-btn")  # 添加 CSS 类

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 生成结果")
                        with gr.Tab("状态信息"):
                            result_text = gr.Textbox(
                                visible=False,
                                label="处理进度",
                                interactive=False,
                                elem_classes="custom-textbox"  # 添加 CSS 类
                            )
                            task_status_html = gr.HTML(
                            value=get_task_status(),
                            label="当前任务状态"
                            )
                            
                            task_status_text = gr.Textbox(
                                label="详细日志", 
                                interactive=False,
                                lines=4,
                                elem_classes="custom-textbox"  # 添加 CSS 类
                            )
                        with gr.Tab("视频预览"):
                            video_gallery = gr.Gallery(
                                show_label=False,
                                columns=2,
                                height="auto",
                                object_fit="contain",
                                elem_classes="custom-gallery"  # 添加 CSS 类
                            )
                        open_folder_btn = gr.Button("📁 打开输出文件夹", elem_classes="custom-btn")  # 添加 CSS 类
                        gr.Markdown("### 实时资源利用率")
                        with gr.Row():
                            cpu_usage = gr.Textbox(label="CPU利用率", interactive=False)
                            gpu_usage = gr.Textbox(label="GPU利用率", interactive=False)
                        with gr.Row():
                            monitor_btn = gr.Button("📈 开始实时监控", variant="secondary", elem_classes="custom-btn")
                            stop_monitor_btn = gr.Button("⏹️ 停止监控", variant="stop", elem_classes="custom-btn")

        # 事件绑定（原有逻辑保持不变）
        
        refresh_btn.click(refresh_voice_list, outputs=speaker)
        customize_btn.click(customize_voice, inputs=[prompt_wav,speaker_name], outputs=customize_output)
        # -------------------------- 新增：情感控制方式切换逻辑 --------------------------
        def on_emo_method_change(emo_method):
            # 判断是否选择「使用情感向量控制」
            if emo_method == "使用情感向量控制":
                return (
                    gr.update(visible=True),  # 显示情感向量滑块组
                    gr.update(visible=True)   # 显示情感权重滑块
                )
            else:
                return (
                    gr.update(visible=False),  # 隐藏情感向量滑块组
                    gr.update(visible=False)   # 隐藏情感权重滑块
                )

        # 绑定Radio选择变化事件：当「情感控制方式」改变时，更新UI显示
        emo_control_method.change(
            fn=on_emo_method_change,
            inputs=[emo_control_method],
            outputs=[emotion_vector_group, emo_weight_group]
        )
        # --------------------------------------------------------------------------
       
        generate_btn.click(
            generate_audio, 
            inputs=[
                tts_text, speaker,  # 原有参数
                # 新增情感相关参数（需与generate_audio函数参数顺序一致）
                emo_control_method,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_weight
            ],
            outputs=[output_audio]
        )
        go_to_digital_human_btn.click(
            fn=lambda audio: (gr.Tabs(selected="tab3"), audio),
            inputs=output_audio,
            outputs=[tabs, single_audio_input]
        )
        dropdown.change(load_selected_video, inputs=dropdown, outputs=[video_input])
        open_folder_btn.click(
            fn=open_output_folder,
            outputs=gr.Textbox(visible=False)  # 无实际输出，仅触发动作
        )
        save_button.click(
            save_files,
            inputs=[video_input, audio_folder_input, single_audio_input],
            outputs=[result_text]
        )
        monitor_btn.click(
            fn=start_monitoring,
            outputs=[cpu_usage, gpu_usage]
        )
        delete_voice_btn.click(
            fn=delete_voice_model,
            inputs=[speaker],
            outputs=[result_text, speaker]
        )

        delete_video_btn.click(
            fn=delete_video_model,
            inputs=[dropdown],
            outputs=[result_text, dropdown]
        )
        speaker.change(
            load_preview_audio,
            inputs=speaker,
            outputs=[preview_audio]
        )
        stop_monitor_btn.click(
            fn=stop_monitoring,
            outputs=[cpu_usage, gpu_usage]
        )
        def auto_refresh_tasks():
            while True:
                time.sleep(1)
                yield get_task_status()
        demo.load(auto_refresh_tasks, outputs=task_status_html)
    
        # 自动刷新
        def update_interface():
            if hasattr(app, "generated_videos"):
                return app.generated_videos, app.task_status if hasattr(app, "task_status") else ""
            return [], ""

        def auto_refresh():
            while True:
                time.sleep(1)
                if hasattr(app, "generated_videos"):
                    yield update_interface()

        save_button.click(auto_refresh, inputs=None, outputs=[video_gallery, task_status_text])#点击生成后开始更新video_gallery, task_status_text
        return demo
    
def create_chinese_traditional_block():
    with gr.Blocks(title="CosyVoice 數字人系統", css=custom_css) as demo:
        with gr.Tabs(elem_classes="tab_buttun") as tabs:
            # 第一頁：音色定制
            with gr.TabItem("🎙️ 音色定制", id="tab1", elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上傳參考音頻", elem_classes="Markdown")
                        gr.Markdown("音頻時長35s以內", elem_classes="Markdown")
                        prompt_wav = gr.Audio(
                            show_label=False,
                            type="filepath",
                            interactive=True,
                            elem_classes="custom-audio"
                        )
                        gr.Markdown("### 設置音色參數")
                        with gr.Group(elem_classes="custom-group"):
                            prompt_text = gr.Textbox(
                                label="參考音頻文本（自動識別）",
                                placeholder="音頻識別結果將自動顯示在這裡",
                                lines=2,
                                elem_classes="custom-textbox"
                            )
                            speaker_name = gr.Textbox(
                                label="音色名稱", 
                                placeholder="為您的音色起個名字",
                                info="建議使用英文命名",
                                elem_classes="custom-textbox"
                            )
                        customize_btn = gr.Button(
                            "✨ 開始定制音色", 
                            variant="primary",
                            elem_classes="custom-btn"
                        )

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 定制結果")
                        customize_output = gr.Textbox(
                            label="狀態信息",
                            interactive=False,
                            placeholder="等待音色定制...",
                            elem_classes="custom-textbox"
                        )

            # 第二頁：語音合成
            with gr.TabItem("🔊 語音合成", id="tab2", elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 選擇音色與文本")
                        with gr.Row():
                            voice_dir = os.path.join(ROOT_DIR, "voices")
                            voice_files = []
                            if os.path.exists(voice_dir):
                                voice_files = [f.replace(".pt", "") for f in os.listdir(voice_dir) if f.endswith(".pt")]
                            with gr.Column(scale=1):
                                with gr.Row():
                                    speaker = gr.Dropdown(
                                        label="選擇定制音色", 
                                        choices=voice_files,
                                        value=voice_files[0] if voice_files else None,
                                        interactive=True,
                                        elem_classes="custom-dropdown"
                                    )
                                    preview_audio = gr.Audio(
                                        show_label=False,
                                        interactive=False,
                                        visible=False,
                                        elem_classes="custom-audio-preview" 
                                    )
                                    with gr.Column():
                                        refresh_btn = gr.Button("🔄 刷新列表", size="sm", elem_classes="custom-btn")
                                        delete_voice_btn = gr.Button("🗑️ 刪除音色", size="sm", variant="stop", elem_classes="custom-btn") 

                        tts_text = gr.Textbox(
                            label="輸入合成文本", 
                            placeholder="請輸入要轉換為語音的文字內容...",
                            lines=14,
                            elem_classes="custom-textbox"
                        )
                        # 情感控制方式选择
                        emo_control_method = gr.Radio(
                            label="情感控制方式",
                            choices=["與音色參考音頻相同", "使用情感向量控制"],
                            value="與音色參考音頻相同",
                            interactive=True
                        )

                        # 情感向量滑块组
                        with gr.Group(visible=False) as emotion_vector_group:
                            gr.Markdown("### 情感向量調節（8維度）", elem_classes="emotion-vector-title")
                            with gr.Row(elem_classes="emotion-vector-section"):
                                with gr.Column(scale=1):
                                    vec1 = gr.Slider(label="喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec2 = gr.Slider(label="怒", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec3 = gr.Slider(label="哀", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec4 = gr.Slider(label="懼", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                with gr.Column(scale=1):
                                    vec5 = gr.Slider(label="厭惡", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec6 = gr.Slider(label="低落", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec7 = gr.Slider(label="驚喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec8 = gr.Slider(label="平靜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

                        # 情感权重
                        with gr.Row(visible=False) as emo_weight_group:
                            emo_weight = gr.Slider(label="情感權重", minimum=0.0, maximum=1.0, value=0.65, step=0.01)


                        with gr.Row():
                            generate_btn = gr.Button("🎵 生成音頻", variant="primary")
                            go_to_digital_human_btn = gr.Button("➡️ 前往數字人合成", variant="secondary", elem_classes="custom-btn")

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 生成結果")
                        output_audio = gr.Audio(
                            show_label=False,
                            interactive=False,
                            waveform_options={
                                "waveform_progress_color": "#4a8cff"
                            },
                            elem_classes="custom-audio"
                        )
                        gr.Examples(
                            examples=["你好，歡迎使用語音克隆系統", "今天天氣真好"],
                            inputs=[tts_text],
                            label="試試示例文本",
                        )

            # 第三頁：數字人合成
            with gr.TabItem("🎬 數字人合成", id="tab3", elem_classes="tabs", visible=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 選擇素材")
                        with gr.Group(elem_classes="custom-group"):
                            model_dir = os.path.join(ROOT_DIR, "result")
                            folders = get_result_folders()
                            
                            dropdown = gr.Dropdown(
                                choices=folders, 
                                label="選擇已有視頻模板",
                                info="或上傳新視頻",
                                elem_classes="custom-dropdown"
                            )
                            delete_video_btn = gr.Button("🗑️ 刪除模板", variant="stop", elem_classes="custom-btn")
                            video_input = gr.Video(
                                show_label=False,
                                sources=["upload"],
                                format="mp4",
                                elem_classes="custom-video"
                            )
                        
                        gr.Markdown("### 添加音頻")
                        with gr.Accordion("批量生成選項", open=False):
                            audio_folder_input = gr.Textbox(
                                label="音頻文件夾路徑",
                                placeholder="輸入包含多個音頻文件的文件夾路徑",
                                elem_classes="custom-textbox"
                            )
                        single_audio_input = gr.Audio(
                            show_label=False,
                            type="filepath",
                            elem_classes="custom-audio"
                        )
                        save_button = gr.Button("🚀 生成數字人視頻", variant="primary", elem_classes="custom-btn")

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### 生成結果")
                        with gr.Tab("狀態信息"):
                            result_text = gr.Textbox(
                                visible=False,
                                label="處理進度",
                                interactive=False,
                                elem_classes="custom-textbox"
                            )
                            task_status_html = gr.HTML(
                                value=get_task_status(),
                                label="當前任務狀態"
                            )
                            
                            task_status_text = gr.Textbox(
                                label="詳細日誌", 
                                interactive=False,
                                lines=4,
                                elem_classes="custom-textbox"
                            )
                        with gr.Tab("視頻預覽"):
                            video_gallery = gr.Gallery(
                                show_label=False,
                                columns=2,
                                height="auto",
                                object_fit="contain",
                                elem_classes="custom-gallery"
                            )
                        open_folder_btn = gr.Button("📁 打開輸出文件夾", elem_classes="custom-btn")
                        gr.Markdown("### 實時資源利用率")
                        with gr.Row():
                            cpu_usage = gr.Textbox(label="CPU利用率", interactive=False)
                            gpu_usage = gr.Textbox(label="GPU利用率", interactive=False)
                        with gr.Row():
                            monitor_btn = gr.Button("📈 開始實時監控", variant="secondary", elem_classes="custom-btn")
                            stop_monitor_btn = gr.Button("⏹️ 停止監控", variant="stop", elem_classes="custom-btn")
        # 事件绑定（原有逻辑保持不变）
        # 添加情感控制切换逻辑
        def on_emo_method_change(emo_method):
            if emo_method == "使用情感向量控制":
                return (gr.update(visible=True), gr.update(visible=True))
            else:
                return (gr.update(visible=False), gr.update(visible=False))

        emo_control_method.change(
            fn=on_emo_method_change,
            inputs=[emo_control_method],
            outputs=[emotion_vector_group, emo_weight_group]
        )   
        generate_btn.click(
            generate_audio, 
            inputs=[
                tts_text, speaker,
                emo_control_method,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_weight
            ],
            outputs=output_audio
        )     
        refresh_btn.click(refresh_voice_list, outputs=speaker)
        customize_btn.click(customize_voice, inputs=[prompt_wav,  speaker_name], outputs=customize_output)
        go_to_digital_human_btn.click(
            fn=lambda audio: (gr.Tabs(selected="tab3"), audio),
            inputs=output_audio,
            outputs=[tabs, single_audio_input]
        )
        dropdown.change(load_selected_video, inputs=dropdown, outputs=[video_input])
        open_folder_btn.click(
            fn=open_output_folder,
            outputs=gr.Textbox(visible=False)  # 无实际输出，仅触发动作
        )
        save_button.click(
            save_files,
            inputs=[video_input, audio_folder_input, single_audio_input],
            outputs=[result_text]
        )
        monitor_btn.click(
        fn=start_monitoring,
        outputs=[cpu_usage, gpu_usage]
        )
        delete_voice_btn.click(
            fn=delete_voice_model,
            inputs=[speaker],
            outputs=[result_text, speaker]
        )

        delete_video_btn.click(
            fn=delete_video_model,
            inputs=[dropdown],
            outputs=[result_text, dropdown]
        )
        speaker.change(
        load_preview_audio,
        inputs=speaker,
        outputs=[preview_audio]
        )
        stop_monitor_btn.click(
            fn=stop_monitoring,
            outputs=[cpu_usage, gpu_usage]
        )
        def auto_refresh_tasks():
            while True:
                time.sleep(1)
                yield get_task_status()
        demo.load(auto_refresh_tasks, outputs=task_status_html)
    
        # 自动刷新
        def update_interface():
            if hasattr(app, "generated_videos"):
                return app.generated_videos, app.task_status if hasattr(app, "task_status") else ""
            return [], ""

        def auto_refresh():
            while True:
                time.sleep(1)
                if hasattr(app, "generated_videos"):
                
                    yield update_interface()

        save_button.click(auto_refresh, inputs=None, outputs=[video_gallery, task_status_text])#点击生成后开始更新video_gallery, task_status_text
        
        return demo

def create_english_block():
    with gr.Blocks(title="CosyVoice Digital Human System", css=custom_css) as demo:
        with gr.Tabs(elem_classes="tab_buttun") as tabs:
            # Tab 1: Voice Customization
            with gr.TabItem("🎙️ Voice Custom", id="tab1", elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Reference Audio", elem_classes="Markdown")
                        gr.Markdown("Audio duration within 35s", elem_classes="Markdown")
                        prompt_wav = gr.Audio(
                            show_label=False,
                            type="filepath",
                            interactive=True,
                            elem_classes="custom-audio"
                        )
                        gr.Markdown("### Voice Parameters")
                        with gr.Group(elem_classes="custom-group"):
                            prompt_text = gr.Textbox(
                                label="Reference Audio Text (Auto Recognized)",
                                placeholder="Audio recognition results will appear here",
                                lines=2,
                                elem_classes="custom-textbox"
                            )
                            speaker_name = gr.Textbox(
                                label="Voice Name", 
                                placeholder="Name your voice",
                                info="Recommended to use English names",
                                elem_classes="custom-textbox"
                            )
                        customize_btn = gr.Button(
                            "✨ Start Customization", 
                            variant="primary",
                            elem_classes="custom-btn"
                        )

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### Customization Result")
                        customize_output = gr.Textbox(
                            label="Status Information",
                            interactive=False,
                            placeholder="Waiting for voice customization...",
                            elem_classes="custom-textbox"
                        )

            # Tab 2: Speech Synthesis
            with gr.TabItem("🔊 TTS", id="tab2", elem_classes="tabs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Select Voice & Text")
                        with gr.Row():
                            voice_dir = os.path.join(ROOT_DIR, "voices")
                            voice_files = []
                            if os.path.exists(voice_dir):
                                voice_files = [f.replace(".pt", "") for f in os.listdir(voice_dir) if f.endswith(".pt")]
                            with gr.Column(scale=1):
                                with gr.Row():
                                    speaker = gr.Dropdown(
                                        label="Select Custom Voice", 
                                        choices=voice_files,
                                        value=voice_files[0] if voice_files else None,
                                        interactive=True,
                                        elem_classes="custom-dropdown"
                                    )
                                    preview_audio = gr.Audio(
                                        show_label=False,
                                        interactive=False,
                                        visible=False,
                                        elem_classes="custom-audio-preview" 
                                    )
                                    with gr.Column():
                                        refresh_btn = gr.Button("🔄 Refresh List", size="sm", elem_classes="custom-btn")
                                        delete_voice_btn = gr.Button("🗑️ Delete Voice", size="sm", variant="stop", elem_classes="custom-btn") 

                        tts_text = gr.Textbox(
                            label="Input Text", 
                            placeholder="Enter text to convert to speech...",
                            lines=14,
                            elem_classes="custom-textbox"
                        )
                        # 情感控制方式选择
                        emo_control_method = gr.Radio(
                            label="Emotion Control Method",
                            choices=["Same as Reference Audio", "Use Emotion Vector Control"],
                            value="Same as Reference Audio",
                            interactive=True
                        )

                        # 情感向量滑块组
                        with gr.Group(visible=False) as emotion_vector_group:
                            gr.Markdown("### Emotion Vector Adjustment (8 Dimensions)", elem_classes="emotion-vector-title")
                            with gr.Row(elem_classes="emotion-vector-section"):
                                with gr.Column(scale=1):
                                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec3 = gr.Slider(label="Sorrow", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                with gr.Column(scale=1):
                                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec6 = gr.Slider(label="Depression", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

                        # 情感权重
                        with gr.Row(visible=False) as emo_weight_group:
                            emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.0, value=0.65, step=0.01)


                        with gr.Row():
                            generate_btn = gr.Button("🎵 Generate Audio", variant="primary")
                            go_to_digital_human_btn = gr.Button("➡️ Go to Digital Human", variant="secondary", elem_classes="custom-btn")

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### Generation Result")
                        output_audio = gr.Audio(
                            show_label=False,
                            interactive=False,
                            waveform_options={
                                "waveform_progress_color": "#4a8cff"
                            },
                            elem_classes="custom-audio"
                        )
                        gr.Examples(
                            examples=["Hello, welcome to the voice cloning system", "The weather is nice today"],
                            inputs=[tts_text],
                            label="Try Example Texts",
                        )

            # Tab 3: Digital Human
            with gr.TabItem("🎬 Digital Human", id="tab3", elem_classes="tabs", visible=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Select Materials")
                        with gr.Group(elem_classes="custom-group"):
                            model_dir = os.path.join(ROOT_DIR, "result")
                            folders = get_result_folders()
                            
                            dropdown = gr.Dropdown(
                                choices=folders, 
                                label="Select Existing Video Template",
                                info="Or upload new video",
                                elem_classes="custom-dropdown"
                            )
                            delete_video_btn = gr.Button("🗑️ Delete Template", variant="stop", elem_classes="custom-btn")
                            video_input = gr.Video(
                                show_label=False,
                                sources=["upload"],
                                format="mp4",
                                elem_classes="custom-video"
                            )
                        
                        gr.Markdown("### Add Audio")
                        with gr.Accordion("Batch Generation Options", open=False):
                            audio_folder_input = gr.Textbox(
                                label="Audio Folder Path",
                                placeholder="Enter folder path containing multiple audio files",
                                elem_classes="custom-textbox"
                            )
                        single_audio_input = gr.Audio(
                            show_label=False,
                            type="filepath",
                            elem_classes="custom-audio"
                        )
                        save_button = gr.Button("🚀 Generate Digital Human Video", variant="primary", elem_classes="custom-btn")

                    with gr.Column(scale=1, elem_classes="output-section"):
                        gr.Markdown("### Generation Result")
                        with gr.Tab("Status Information"):
                            result_text = gr.Textbox(
                                visible=False,
                                label="Processing Progress",
                                interactive=False,
                                elem_classes="custom-textbox"
                            )
                            task_status_html = gr.HTML(
                                value=get_task_status_en(),
                                label="Current Task Status"
                            )
                            
                            task_status_text = gr.Textbox(
                                label="Detailed Logs", 
                                interactive=False,
                                lines=4,
                                elem_classes="custom-textbox"
                            )
                        with gr.Tab("Video Preview"):
                            video_gallery = gr.Gallery(
                                show_label=False,
                                columns=2,
                                height="auto",
                                object_fit="contain",
                                elem_classes="custom-gallery"
                            )
                        open_folder_btn = gr.Button("📁 Open Output Folder", elem_classes="custom-btn")
                        gr.Markdown("### Real-time Resource Usage")
                        with gr.Row():
                            cpu_usage = gr.Textbox(label="CPU Usage", interactive=False)
                            gpu_usage = gr.Textbox(label="GPU Usage", interactive=False)
                        with gr.Row():
                            monitor_btn = gr.Button("📈 Start Monitoring", variant="secondary", elem_classes="custom-btn")
                            stop_monitor_btn = gr.Button("⏹️ Stop Monitoring", variant="stop", elem_classes="custom-btn")
        # 事件绑定（原有逻辑保持不变）
        # 添加情感控制切换逻辑
        def on_emo_method_change(emo_method):
            if emo_method == "Use Emotion Vector Control":
                return (gr.update(visible=True), gr.update(visible=True))
            else:
                return (gr.update(visible=False), gr.update(visible=False))

        emo_control_method.change(
            fn=on_emo_method_change,
            inputs=[emo_control_method],
            outputs=[emotion_vector_group, emo_weight_group]
        )  
        generate_btn.click(
            generate_audio, 
            inputs=[
                tts_text, speaker,
                emo_control_method,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_weight
            ],
            outputs=output_audio
        )     
        refresh_btn.click(refresh_voice_list, outputs=speaker)
        customize_btn.click(customize_voice, inputs=[prompt_wav, speaker_name], outputs=customize_output)
        # 更新generate_btn的click事件，增加instruct_text输入

        go_to_digital_human_btn.click(
            fn=lambda audio: (gr.Tabs(selected="tab3"), audio),
            inputs=output_audio,
            outputs=[tabs, single_audio_input]
        )
        dropdown.change(load_selected_video, inputs=dropdown, outputs=[video_input])
        open_folder_btn.click(
            fn=open_output_folder,
            outputs=gr.Textbox(visible=False)  # 无实际输出，仅触发动作
        )
        save_button.click(
            save_files,
            inputs=[video_input, audio_folder_input, single_audio_input],
            outputs=[result_text]
        )
        monitor_btn.click(
        fn=start_monitoring,
        outputs=[cpu_usage, gpu_usage]
        )
        delete_voice_btn.click(
            fn=delete_voice_model,
            inputs=[speaker],
            outputs=[result_text, speaker]
        )

        delete_video_btn.click(
            fn=delete_video_model,
            inputs=[dropdown],
            outputs=[result_text, dropdown]
        )
        speaker.change(
        load_preview_audio,
        inputs=speaker,
        outputs=[preview_audio]
        )
        stop_monitor_btn.click(
            fn=stop_monitoring,
            outputs=[cpu_usage, gpu_usage]
        )
        def auto_refresh_tasks():
            while True:
                time.sleep(1)
                yield get_task_status_en()
        demo.load(auto_refresh_tasks, outputs=task_status_html)
    
        # 自动刷新
        def update_interface():
            if hasattr(app, "generated_videos"):
                return app.generated_videos, app.task_status if hasattr(app, "task_status") else ""
            return [], ""

        def auto_refresh():
            while True:
                time.sleep(1)
                if hasattr(app, "generated_videos"):
                    yield update_interface()

        save_button.click(auto_refresh, inputs=None, outputs=[video_gallery, task_status_text])#点击生成后开始更新video_gallery, task_status_text
        
        return demo




# 语言偏好文件路径
LANGUAGE_FILE = "language_preference.txt"

def get_saved_language():
    """读取保存的语言偏好，默认返回 'zh-CN'"""
    if os.path.exists(LANGUAGE_FILE):
        try:
            with open(LANGUAGE_FILE, "r", encoding="utf-8") as f:
                lang = f.read().strip()
                if lang in ["zh-CN", "zh-TW", "en"]:
                    return lang
        except Exception as e:
            print(f"读取语言偏好文件失败: {e}")
    return "zh-CN"  # 默认简体中文
lang=get_saved_language()
def save_language(lang):
    """保存语言偏好到文件"""
    try:
        with open(LANGUAGE_FILE, "w", encoding="utf-8") as f:
            f.write(lang)
    except Exception as e:
        print(f"保存语言偏好失败: {e}")

def create_main_app():
    # 获取上次保存的语言
    initial_lang = get_saved_language()

    with gr.Blocks(title="多语言应用", css=custom_css) as app:
        # 创建语言区块（初始状态由 initial_lang 决定）
        with gr.Group(visible=initial_lang == "zh-CN", elem_id="cn_block") as cn_block:
            cn_ui = create_chinese_simplified_block()
        
        with gr.Group(visible=initial_lang == "zh-TW", elem_id="tw_block") as tw_block:
            tw_ui = create_chinese_traditional_block()
            
        with gr.Group(visible=initial_lang == "en", elem_id="en_block") as en_block:
            en_ui = create_english_block()

        # 隐藏文本框，用于接收语言切换指令（初始值为保存的语言）
        lang_display = gr.Textbox(
            value=initial_lang,
            visible=False,
            interactive=False,
            elem_id="lang_display"
        )

        # 语言切换逻辑（保留原有逻辑 + 保存到文件）
        def switch_language(language):
            print(f"[DEBUG] 切换语言: {language}")
            lang=language
            save_language(lang)  # 新增：保存到文件
            return [
                gr.update(visible=lang == "zh-CN"),  # cn_block
                gr.update(visible=lang == "zh-TW"),  # tw_block
                gr.update(visible=lang == "en")      # en_block
            ]

        # 监听语言切换（原有逻辑不变）
        lang_display.input(
            fn=switch_language,
            inputs=lang_display,
            outputs=[cn_block, tw_block, en_block]
        )

        # 初始化时自动应用上次保存的语言（原有逻辑不变）
        app.load(
            None,
            None,
            None,
            js="""
            function() {
                console.log("[JS DEBUG] 初始化语言监听器...");
                
                // 1. 自动触发初始语言（由后端传递的 initial_lang 决定）
                const displayBox = document.getElementById('lang_display');
                if (displayBox) {
                    const textarea = displayBox.querySelector('textarea');
                    if (textarea) {
                        // 触发语言切换
                        const inputEvent = new Event('input', { bubbles: true });
                        textarea.dispatchEvent(inputEvent);
                    }
                }
                
                // 2. 保留原有的父窗口消息监听（兼容原有逻辑）
                window.addEventListener('message', (event) => {
                    console.log("[JS DEBUG] 收到消息:", event.data);
                    
                    if (event.data?.type === 'language-change') {
                        const lang = event.data.language;
                        console.log("[JS DEBUG] 处理语言切换:", lang);
                        
                        const displayBox = document.getElementById('lang_display');
                        if (displayBox) {
                            const textarea = displayBox.querySelector('textarea');
                            if (textarea) {
                                textarea.value = lang;
                                // 触发 input 事件
                                const inputEvent = new Event('input', { bubbles: true });
                                textarea.dispatchEvent(inputEvent);
                            }
                        }
                    }
                });
                
                return [];
            }
            """
        )
    
    return app

if __name__ == "__main__":

    app = create_main_app()
    app.launch(
        allowed_paths=[ROOT_DIR]
    )

