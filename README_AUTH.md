# 因果推理平台 - 用户认证集成

这个项目已经集成了基于火伴API的用户认证系统。

## 功能特性

- ✅ 用户注册和登录
- ✅ 基于火伴API的数据库存储
- ✅ 会话管理和登录保护
- ✅ MD5密码加密
- ✅ 登录状态验证
- ✅ 用户登出功能

## 配置信息

### 火伴API配置
- **API主机**: api.huoban.com
- **表ID**: 2100000066422526
- **API密钥**: 9pTFg4AxdFRKsTb1y9667Rq1uoF2kCAtRsjXmVEe

## 安装和运行

### 1. 安装依赖
```bash
pip install flask pandas numpy networkx
```

### 2. 创建测试用户（可选）
```bash
python test_user_creation.py create
```

### 3. 测试用户登录
```bash
python test_user_creation.py test
```

### 4. 启动应用
```bash
python app.py
```

### 5. 访问应用
打开浏览器访问: http://localhost:5000

## 用户认证流程

### 注册流程
1. 用户在登录页面点击"注册"
2. 填写用户名、密码（和可选的邮箱）
3. 系统检查用户名是否已存在
4. 对密码进行MD5加密
5. 调用火伴API创建用户记录
6. 注册成功后自动切换到登录表单

### 登录流程
1. 用户输入用户名和密码
2. 系统对密码进行MD5加密
3. 调用火伴API验证用户凭据
4. 登录成功后设置会话状态
5. 重定向到主页面

### 登录保护
以下页面需要登录才能访问：
- `/index.html` - 主页
- `/data-upload.html` - 数据上传
- `/data-preparation.html` - 数据准备
- `/statistical-analysis.html` - 统计分析
- `/causal-analysis.html` - 因果分析
- `/big-model-analysis.html` - 大模型分析
- `/favorites.html` - 收藏夹

## 数据库字段说明

在火伴表中，用户数据包含以下字段：
- `username` - 用户名（唯一）
- `password_hash` - MD5加密的密码哈希
- `email` - 邮箱地址（可选）

## API接口

### POST /login
用户登录接口
```json
{
    "username": "用户名",
    "password": "密码"
}
```

### POST /register  
用户注册接口
```json
{
    "username": "用户名", 
    "password": "密码",
    "email": "邮箱（可选）"
}
```

### GET /logout
用户登出接口

## 错误处理

系统包含完整的错误处理机制：
- 用户名或密码为空的验证
- 用户名重复检查
- API请求失败处理
- 会话超时处理

## 安全特性

- 密码MD5加密存储
- 会话管理
- 登录状态验证
- CSRF保护（通过Flask内置机制）

## 配置管理

项目使用 `config.py` 进行配置管理，支持：
- 开发环境配置
- 生产环境配置  
- 测试环境配置

可以通过设置 `FLASK_ENV` 环境变量来切换配置。

## 测试账户

可以使用测试脚本创建以下测试账户：
- 用户名: `admin`, 密码: `password`
- 用户名: `test_user`, 密码: `test123`
