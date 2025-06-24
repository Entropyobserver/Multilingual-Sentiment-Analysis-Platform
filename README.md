# 情感分析应用 - Azure 部署指南

基于 FastAPI + React 的情感分析应用，针对 Azure 免费学生账户优化。

## 项目架构

```
├── frontend/          → React 前端 (Azure Static Web Apps)
└── backend/           → FastAPI 后端 (Azure App Service)
```

## 特性

- 🚀 实时情感分析
- 💾 智能结果缓存
- 🔄 自动健康检查
- 📱 响应式设计
- 🔒 请求频率限制
- ⚡ 优化的资源使用

## 前置要求

- Azure 学生账户
- Node.js v14+
- Python 3.9+
- Git

## 本地开发

### 后端设置

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 前端设置

```bash
cd frontend
npm install
npm start
```

## Azure 部署步骤

### 1. 后端部署 (Azure App Service)

1. 登录 Azure Portal
2. 创建 App Service:
   - 创建资源 → App Service
   - 运行时栈: Python 3.9
   - 操作系统: Linux
   - 定价计划: F1 (免费)

3. 配置部署:
   ```bash
   az login
   az webapp up --name YOUR_BACKEND_NAME --resource-group YOUR_GROUP --runtime "PYTHON:3.9"
   ```

4. 环境变量设置:
   - `SCM_DO_BUILD_DURING_DEPLOYMENT=true`
   - `PYTHON_VERSION=3.9`

### 2. 前端部署 (Azure Static Web Apps)

1. 创建 Static Web App:
   - 创建资源 → Static Web App
   - 选择 GitHub 仓库
   - 构建预设: React
   - 应用代码位置: /frontend
   - 输出位置: build

2. 配置环境变量:
   ```env
   REACT_APP_API_URL=https://your-backend.azurewebsites.net
   ```

## 性能优化

后端优化:
- 使用轻量级模型 (DistilBERT)
- 实现请求缓存
- 限制并发请求
- 强制使用 CPU 以适应免费层

前端优化:
- 延迟加载
- 状态缓存
- 响应式设计
- 错误处理

## 监控和维护

1. 健康检查:
   - 访问 `/health` 端点
   - 前端自动监控

2. 日志查看:
   - Azure Portal → App Service → 日志流
   - Azure Portal → Static Web Apps → 监控

3. 性能监控:
   - 响应时间
   - 内存使用
   - 请求频率

## 常见问题

1. 冷启动延迟
   - 首次请求可能需要30-60秒
   - 使用健康检查预热

2. 内存限制
   - 已优化模型加载
   - 实现智能缓存

3. CORS 问题
   - 检查域名配置
   - 验证请求头

## 安全建议

1. 生产环境配置:
   - 限制 CORS 域名
   - 添加 API 密钥
   - 实现速率限制

2. 数据保护:
   - 输入验证
   - 长度限制
   - 错误处理

## 后续优化方向

1. 功能扩展:
   - 批量分析
   - 多语言支持
   - 详细分析报告

2. 性能提升:
   - 模型量化
   - 结果持久化
   - 负载均衡

3. 用户体验:
   - 分析历史
   - 导出功能
   - 自定义主题 