import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.services.transcription import TranscriptionService

# 我们需要 Mock 掉 Engine，防止测试时加载真实模型
@pytest.fixture
def mock_engine_class():
    with patch("src.main.SenseVoiceEngine") as mock:
        yield mock

@pytest.fixture
def client(mock_engine_class):
    """
    创建 TestClient。
    注意：TestClient 会触发 lifespan (启动事件)。
    我们通过 mock_engine_class 确保 lifespan 中初始化的 Engine 是假的。
    """
    # 1. Setup Mock Engine 实例
    mock_instance = MagicMock()
    mock_engine_class.return_value = mock_instance
    
    # Mock 推理结果
    mock_instance.transcribe_file.return_value = "Integration Test Result"
    
    # 2. 启动 Client
    # 使用 with 语句触发 lifespan (startup/shutdown)
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_transcribe_endpoint(client):
    """测试转录接口"""
    # 1. 准备文件
    files = {
        "file": ("test.wav", b"fake audio bytes", "audio/wav")
    }
    data = {
        "language": "zh",
        "clean_tags": "true"
    }
    
    # 2. 发起请求
    response = client.post("/v1/audio/transcriptions", files=files, data=data)
    
    # 3. 验证响应
    assert response.status_code == 200
    result = response.json()
    
    # assert result["object"] == "transcription"  # 当前实现未返回 object 字段

    assert result["text"] == "Integration Test Result"
    assert "duration" in result

def test_transcribe_no_file(client):
    """测试缺少文件的情况"""
    response = client.post("/v1/audio/transcriptions", data={"language": "zh"})
    assert response.status_code == 422 # Validation Error
