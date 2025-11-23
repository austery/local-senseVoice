import re

def clean_sensevoice_tags(text: str, clean_tags: bool = True) -> str:
    """
    清洗 SenseVoice 输出的特殊标签。
    
    Args:
        text: 原始文本 (e.g. "<|zh|><|NEUTRAL|>你好")
        clean_tags: 是否执行清洗
        
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
        
    if not clean_tags:
        return text
        
    # 1. 使用正则去掉所有 <|...|> 格式的标签
    # 比如 <|zh|>, <|NEUTRAL|>, <|Speech|>, <|withitn|> 等
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    
    # 2. 去掉多余的空格 (有时候标签去掉后会留下双空格)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned