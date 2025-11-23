from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 引入我们生成的所有组件
from src.core.engine import SenseVoiceEngine
from src.services.transcription import TranscriptionService
from src.api.routes import router as api_router

# === 全局配置 ===
# 可以从环境变量读取，这里硬编码作为 MVP
MODEL_ID = "iic/SenseVoiceSmall"
HOST = "0.0.0.0"
PORT = 50070  # 你的幸运端口

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    生命周期管理器 (The System Lifecycle)
    FastAPI 启动前执行 yield 前的代码，关闭后执行 yield 后的代码。
    """
    print("🌱 System starting up...")
    
    # 1. 初始化并加载引擎 (The Engine)
    # 这会触发模型下载和 MPS 预热
    engine = SenseVoiceEngine(model_id=MODEL_ID)
    engine.load()
    
    # 2. 初始化并启动服务 (The Service)
    # 此时队列建立，由于还未收到请求，队列为空
    service = TranscriptionService(engine=engine, max_queue_size=50)
    
    # 3. 启动后台消费者 (The Worker)
    # 这是一个死循环协程，必须用 create_task 扔到后台跑
    await service.start_worker()
    
    # 4. 依赖注入 (Dependency Injection)
    # 把 service 挂到 app.state 上，让路由层可以用
    app.state.service = service
    
    print("✅ System ready! Listening for requests...")
    
    yield  # --- 服务运行中 ---
    
    print("🛑 System shutting down...")
    # 可以在这里做清理工作，比如等待队列清空 (Graceful Shutdown)

# === 初始化 FastAPI ===
app = FastAPI(
    title="Local SenseVoice API",
    version="1.0.0",
    lifespan=lifespan  # 挂载生命周期
)

# 允许跨域 (方便前端调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)

# 简单的健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_ID}

if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(app, host=HOST, port=PORT)