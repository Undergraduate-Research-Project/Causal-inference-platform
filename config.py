"""
应用配置文件
包含数据库配置、API密钥等
"""

import os
from datetime import timedelta

class Config:
    """基础配置"""
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_change_in_production'
    
    # 文件上传配置
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # 会话配置
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # 火伴API配置
    HUOBAN_API_HOST = "api.huoban.com"
    HUOBAN_TABLE_ID = "2100000066422526"
    HUOBAN_API_KEY = "9pTFg4AxdFRKsTb1y9667Rq1uoF2kCAtRsjXmVEe"
    
    # DeepSeek API配置（如果有的话）
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY') or ""
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True

# 根据环境变量选择配置
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
