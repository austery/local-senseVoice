import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from funasr import AutoModel
import os
import time
import shutil
import torch
import re  # <--- 核心升级：引入正则库，专门用来洗掉那些标签

# === 1. M4 Pro 硬件配置 ===
# 检查是否支持 MPS (Apple Silicon 加速)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 正在初始化 SenseVoice-Small on device: {device}")

# === 2. 加载模型 (只加载一次) ===
# 注意：如果你之前切到了 Hugging Face 源，这里可能需要改回 "FunAudioLLM/SenseVoiceSmall"
# 如果用阿里源，就保持 "iic/SenseVoiceSmall"
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device=device,
    disable_update=True,
    log_level="ERROR"
)
print("✅ 模型加载完毕，等待调用...")

app = FastAPI()

# === 3. 清洗函数：把 SenseVoice 的富文本标签洗成纯文本 ===
def clean_sensevoice_tags(text):
    if not text:
        return ""
    # 1. 使用正则去掉所有 <|...|> 格式的标签
    # 比如 <|zh|>, <|NEUTRAL|>, <|Speech|>, <|withitn|> 等
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    
    # 2. 去掉多余的空格 (有时候标签去掉后会留下双空格)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

# === 4. 握手接口：为了应付软件的 "Test" 按钮 ===
@app.get("/v1/models")
async def list_models():
    print("🔎 收到客户端 Check 模型的请求 (Handshake)")
    return {
        "object": "list",
        "data": [{"id": "sensevoice", "object": "model", "created": int(time.time()), "owned_by": "alibaba"}]
    }

# === 5. 转录接口：OpenAI 兼容格式 ===
@app.post("/v1/audio/transcriptions")
async def openai_compatible_transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: str = Form(default="auto"),
    response_format: str = Form(default="json")
):
    """
    伪装成 OpenAI Whisper API 的 SenseVoice 接口
    """
    start_ts = time.time()
    print(f"🎤 收到音频处理请求: {file.filename}")
    
    # 临时文件处理
    temp_filename = f"temp_{int(start_ts)}_{file.filename}"
    
    try:
        # 1. 保存上传的音频
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 语言设置
        target_lang = language if language in ["zh", "en", "yue", "ja", "ko"] else "auto"
        
        # 3. 执行推理 (SenseVoice 核心)
        res = model.generate(
            input=temp_filename,
            cache={},
            language=target_lang,
            use_itn=True,       # 开启逆文本标准化 (一百 -> 100)
            batch_size_s=60,    # 批处理大小
            merge_vad=True,     # 合并短句
            merge_length_s=15
        )
        
        raw_text = res[0]["text"]
        
        # === 4. 调用清洗函数 (这一步把鬼画符去掉) ===
        clean_text = clean_sensevoice_tags(raw_text)

        # 计算耗时并打印预览
        duration = time.time() - start_ts
        print(f"⚡️ 处理完成: {clean_text[:30]}... (耗时: {duration:.2f}s)")

        return {
            "text": clean_text,
            "model": "SenseVoice-Small-M4-Clean",
            "object": "transcription"
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": True, "reason": str(e)})
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    # 启动服务，监听 50700 端口 (标准化端口)
    # 注意：host="0.0.0.0" 允许你用 127.0.0.1 访问
    print("🚀 服务正在启动，监听端口: 50070")
    uvicorn.run(app, host="0.0.0.0", port=50070)