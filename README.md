# datasetMaker
利用AI多模态能力实现数据集自动标注

## 项目结构

```
datasetMaker/
├── ml_backend/          # Label Studio ML Backend (AI 适配器 / 网关层)
│   ├── app/             # FastAPI 应用
│   │   ├── config.py    # 环境变量配置
│   │   ├── main.py      # 应用入口
│   │   ├── api/routes.py          # /health /setup /predict 接口
│   │   └── services/
│   │       ├── minio_service.py   # MinIO/S3 图片下载 → Base64
│   │       ├── vllm_service.py    # vLLM Vision API 客户端
│   │       └── label_studio.py    # 坐标转换 & LS 结果格式封装
│   ├── tests/           # 单元测试
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md        # ML Backend 详细文档
└── docker-compose.yml   # 完整系统一键启动
```

## 系统架构

```
[ 管控层 ]
  Label Studio (8080) + PostgreSQL + MinIO (9000)
         ↕  HTTP Webhook
[ 网关层 (ML Backend) ]   ← ml_backend/  (9090)
         ↕  OpenAI Vision API
[ 算力层 ]
  vLLM API Server (8000) – Qwen2.5-VL-7B-Instruct
```

详见 [ml_backend/README.md](ml_backend/README.md)。
