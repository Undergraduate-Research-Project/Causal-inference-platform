# 腾讯云服务器部署指南

这套仓库已经补齐了生产部署文件，推荐按 `Docker + Gunicorn + Nginx` 方式上线。这样比直接 `python app.py` 稳定很多，也更适合腾讯云服务器长期运行。

## 1. 先准备腾讯云服务器

推荐系统：`Ubuntu 22.04 LTS`

安全组至少放行这些端口：

- `22`：SSH
- `80`：HTTP
- `443`：HTTPS

如果你暂时还没有域名，也可以先只用公网 IP 跑通 HTTP。

## 2. 把项目传到服务器

推荐放到 `/opt/causal-platform`：

```bash
sudo mkdir -p /opt/causal-platform
sudo chown -R $USER:$USER /opt/causal-platform
```

如果你是从本地电脑上传，可以在你本地执行：

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' /Users/lvzufeng/Desktop/Causal-inference-platform/ ubuntu@<你的服务器公网IP>:/opt/causal-platform/
```

如果你已经把代码放到 Git 仓库，也可以直接在服务器上 `git clone`。

## 3. 在服务器安装 Docker 和 Nginx

在服务器执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg nginx
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 4. 配置环境变量

进入项目目录：

```bash
cd /opt/causal-platform
cp .env.example .env
```

生成一个新的 `SECRET_KEY`：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

然后编辑 `.env`：

```bash
nano .env
```

至少要改这些值：

- `SECRET_KEY=你刚生成的随机串`
- `DEEPSEEK_API_KEY=你的 DeepSeek Key`
- `HUOBAN_TABLE_ID=你的火伴表 ID`
- `HUOBAN_API_KEY=你的火伴 API Key`

说明：

- 如果 `HUOBAN_*` 暂时不填，登录注册相关能力可能不可用。
- DeepSeek Key 不填时，大模型分析相关接口会失败，但基础页面仍可运行。
- 如果你暂时只想先把站点跑起来，HTTP 阶段请保持 `SESSION_COOKIE_SECURE=false`。

## 5. 启动容器

在项目目录执行：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
docker compose logs -f
```

本地健康检查：

```bash
curl http://127.0.0.1:8000/healthz
```

返回 `{"status":"ok"}` 就说明应用已经起来了。

## 6. 配置 Nginx 反向代理

把仓库里的配置复制到 Nginx：

```bash
sudo cp deploy/nginx/causal-platform.conf /etc/nginx/sites-available/causal-platform
```

编辑里面的 `server_name`，替换成：

- 你的域名，或者
- 你的服务器公网 IP

启用站点：

```bash
sudo ln -sf /etc/nginx/sites-available/causal-platform /etc/nginx/sites-enabled/causal-platform
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

现在你应该已经可以通过下面任一地址访问：

- `http://你的域名`
- `http://你的服务器公网IP`

## 7. 配 HTTPS

如果你有域名，推荐立刻上 HTTPS：

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <你的域名>
```

证书成功后，把 `.env` 里的：

```bash
SESSION_COOKIE_SECURE=true
```

然后重启应用：

```bash
docker compose up -d
```

## 8. 常用运维命令

更新代码后重新部署：

```bash
cd /opt/causal-platform
docker compose up -d --build
```

查看实时日志：

```bash
docker compose logs -f web
```

重启服务：

```bash
docker compose restart web
```

停止服务：

```bash
docker compose down
```

## 9. 你现在这套代码上线时要注意的事

- 因果分析接口会调用 `PC算法/pc_easy.py` 和 `GIES算法/gies_easy.py`，所以不能只部署 Web 进程，必须带完整 Python 依赖运行。
- 这次已经把子进程调用改成了复用当前解释器，避免线上 Gunicorn 跑的是一个 Python、算法脚本却跑到另一个 Python 环境里。
- Gunicorn 和 Nginx 都已经把超时放大到 `600s`，避免长时间因果计算被提前中断。
- 上传文件保存在 `uploads/`，现在已经通过 `docker-compose.yml` 做了持久化映射。

## 10. 最后的上线自检

按这个顺序确认：

1. `docker compose ps` 里容器是 `running`
2. `curl http://127.0.0.1:8000/healthz` 返回 `ok`
3. `sudo nginx -t` 通过
4. 浏览器能打开首页
5. 能正常登录
6. 能上传 CSV
7. 能跑一次 PC 或 GIES 因果分析
8. 如果配置了 DeepSeek，能调用大模型分析

如果你部署时卡在某一步，直接把下面三样发出来，我可以继续帮你定位：

- `docker compose ps`
- `docker compose logs --tail=200`
- `sudo nginx -t`
