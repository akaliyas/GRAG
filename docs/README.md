# GRAG 项目文档索引

本文档提供项目所有文档的索引和导航。

## 📚 核心模块文档

### 1. [API 层文档](./API_LAYER.md)

**内容**：
- FastAPI 应用入口和初始化
- RESTful API 路由定义
- HTTP Basic 认证机制
- 请求/响应处理
- 性能监控集成

**适用场景**：
- 了解 API 接口设计
- 集成第三方客户端
- 调试 API 问题

### 2. [Agent 层文档](./AGENT_LAYER.md)

**内容**：
- LangGraph Agent 实现
- 工作流和状态管理
- 意图识别机制
- 工具调用流程
- 条件路由逻辑

**适用场景**：
- 理解 Agent 工作流程
- 添加新工具
- 扩展意图类型

### 3. [模型层文档](./MODEL_LAYER.md)

**内容**：
- 模型管理器实现
- DeepSeek API 集成
- 本地模型（Ollama/vLLM）支持
- 动态切换机制
- 健康检查和自动回退

**适用场景**：
- 配置模型服务
- 切换模型提供商
- 调试模型调用问题

### 4. [知识存储层文档](./KNOWLEDGE_LAYER.md)

**内容**：
- LightRAG 封装实现
- 嵌入模型配置（SiliconFlow/Ollama/OpenAI）
- 文档导入和管理
- 双层检索机制（全局+局部）
- PostgreSQL 存储后端

**适用场景**：
- 配置嵌入模型
- 导入文档到知识库
- 理解检索机制

### 5. [存储层文档](./STORAGE_LAYER.md)

**内容**：
- 缓存管理器实现
- PostgreSQL 缓存表设计
- LRU 清理策略
- 质量评分机制
- 用户反馈处理

**适用场景**：
- 优化缓存策略
- 理解质量评分算法
- 调试缓存问题

### 6. [工具层文档](./TOOLS_LAYER.md)

**内容**：
- GitHub 文档提取工具
- Zero-Crawler 策略
- HTML 标签清理
- Notebook 清洗
- Frontmatter 提取

**适用场景**：
- 从 GitHub 提取文档
- 理解数据清洗流程
- 扩展数据源

### 7. [配置管理文档](./CONFIG_MANAGEMENT.md)

**内容**：
- 配置管理器实现
- YAML 配置文件结构
- 环境变量解析
- 配置访问接口
- 单例模式

**适用场景**：
- 配置系统参数
- 理解配置优先级
- 添加新配置项

### 8. [前端文档](./FRONTEND.md)

**内容**：
- Streamlit 前端实现
- 界面布局和功能
- API 集成
- 会话状态管理
- 用户交互流程

**适用场景**：
- 自定义前端界面
- 添加新功能
- 调试前端问题

## 🔧 专项文档

### 9. [API 认证文档](./API_AUTHENTICATION.md)

**内容**：
- HTTP Basic Authentication 机制
- 认证流程详解
- 配置方法
- 安全建议
- 常见问题

**适用场景**：
- 配置 API 认证
- 理解认证流程
- 安全加固

### 10. [PostgreSQL 配置文档](./POSTGRESQL_CONFIG.md)

**内容**：
- PostgreSQL 数据库配置
- 存储后端设置
- 连接池配置
- 性能优化建议

**适用场景**：
- 配置数据库
- 优化数据库性能
- 故障排查

## 📖 快速导航

### 按角色查找文档

#### 开发者
- **API 开发**：[API 层文档](./API_LAYER.md)、[API 认证文档](./API_AUTHENTICATION.md)
- **Agent 开发**：[Agent 层文档](./AGENT_LAYER.md)、[工具层文档](./TOOLS_LAYER.md)
- **模型集成**：[模型层文档](./MODEL_LAYER.md)
- **知识库管理**：[知识存储层文档](./KNOWLEDGE_LAYER.md)

#### 运维人员
- **系统配置**：[配置管理文档](./CONFIG_MANAGEMENT.md)
- **数据库管理**：[PostgreSQL 配置文档](./POSTGRESQL_CONFIG.md)
- **缓存优化**：[存储层文档](./STORAGE_LAYER.md)

#### 用户
- **使用指南**：[前端文档](./FRONTEND.md)
- **API 使用**：[API 层文档](./API_LAYER.md)

### 按任务查找文档

#### 配置系统
1. [配置管理文档](./CONFIG_MANAGEMENT.md) - 了解配置结构
2. [PostgreSQL 配置文档](./POSTGRESQL_CONFIG.md) - 配置数据库
3. [API 认证文档](./API_AUTHENTICATION.md) - 配置认证

#### 添加新功能
1. [Agent 层文档](./AGENT_LAYER.md) - 了解工作流
2. [工具层文档](./TOOLS_LAYER.md) - 添加新工具
3. [API 层文档](./API_LAYER.md) - 添加新接口

#### 调试问题
1. [模型层文档](./MODEL_LAYER.md) - 模型调用问题
2. [知识存储层文档](./KNOWLEDGE_LAYER.md) - 检索问题
3. [存储层文档](./STORAGE_LAYER.md) - 缓存问题

## 🔗 相关资源

- **主 README**：[../README.md](../README.md)
- **项目结构**：[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)
- **环境变量示例**：[../example.env](../example.env)

## 📝 文档更新

文档会随着项目发展持续更新。如果发现文档与代码不一致，请：

1. 提交 Issue 报告
2. 或直接提交 Pull Request 更新文档

## 💡 贡献指南

### 添加新文档

1. 在 `docs/` 目录创建新的 Markdown 文件
2. 遵循现有文档的格式和风格
3. 在本索引文件中添加链接
4. 更新主 README.md 的文档索引

### 文档规范

- **文件名**：使用大写字母和下划线（如 `API_LAYER.md`）
- **标题**：使用 H1 作为主标题
- **结构**：概述 → 模块结构 → 核心组件 → 使用示例 → 相关文档
- **代码示例**：提供完整的、可运行的示例
- **链接**：使用相对路径链接到相关文档

