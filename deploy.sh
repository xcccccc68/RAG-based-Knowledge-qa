#!/bin/bash

# 部署脚本

# 1. 安装依赖
echo "正在安装依赖..."
pip install -r requirements.txt

# 2. 启动服务
echo "正在启动服务..."
python server.py
