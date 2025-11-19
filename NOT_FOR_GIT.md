# 📝 不上传到Git的文件说明

以下文件类型已在 `.gitignore` 中配置，**不会**被上传到GitHub：

## 🔐 敏感信息文件
- `config.json` - 包含Azure订阅ID、资源组等敏感配置
- `*.env` - 环境变量文件
- `.env.*` - 环境配置文件
- `*_config_local.json` - 本地配置文件
- Azure存储连接字符串等

## 📊 结果和输出文件
- `result/` - 所有推理结果目录
- `result2/` - 备份结果目录
- `outputs/` - 输出文件
- `*_results.json` - 推理结果JSON
- `*_progress.json` - 进度文件
- `*.json.bak` - 备份文件

## 📹 视频和媒体文件
- `*.mp4` - 视频文件
- `DADA-2000-videos/` - 视频数据集
- `audio/` - 音频文件
- `frames/` - 视频帧
- `transcriptions/` - 转录文件

## 🔧 开发环境文件
- `__pycache__/` - Python缓存
- `*.pyc` - Python编译文件
- `.vscode/` - VSCode配置
- `.idea/` - IDE配置
- `*.log` - 日志文件
- `logs/` - 日志目录

## 🤖 模型和大文件
- `*.pth` - PyTorch模型
- `*.bin` - 二进制模型文件
- `*.safetensors` - SafeTensors格式模型
- `models/` - 模型目录
- `checkpoints/` - 检查点文件

## 📦 依赖和构建文件
- `node_modules/` - Node.js依赖
- `env/` - Python虚拟环境
- `build/` - 构建输出
- `dist/` - 分发文件

## 💻 系统文件
- `.DS_Store` - macOS系统文件
- `*.tmp` - 临时文件
- `.cache/` - 缓存目录

## ✅ 已上传的重要文件
以下文件**已经**上传到GitHub供同事使用：

### 📚 文档
- `README_AZURE_ML.md` - 详细使用文档
- `QUICK_START.md` - 快速开始指南
- `config.json.example` - 配置示例文件

### 🔧 脚本
- `run_drivemm_azure.py` - 设置向导脚本
- `setup_drivemm_azure.py` - Azure ML SDK提交脚本
- `azure_drivemm_real_inference.py` - 推理脚本（如果存在）
- `azure_ml_drivemm_real_job.yml` - 作业配置（如果存在）
- `azure_drivemm_environment.yml` - 环境配置（如果存在）

### ⚙️ 配置
- `.gitignore` - Git忽略规则

## 🎯 使用建议

1. **首次使用**: 复制 `config.json.example` 为 `config.json` 并填入你的配置
2. **保护敏感信息**: 永远不要修改 `.gitignore` 来上传敏感文件
3. **结果分享**: 如需分享结果，建议使用Azure Storage或其他安全方式
4. **本地开发**: 所有临时文件和结果都会自动被忽略

---

💡 如有疑问，请查看 `.gitignore` 文件了解完整的忽略规则。
