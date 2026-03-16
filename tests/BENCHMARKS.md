# GRAG 真实场景基准测试

## 概述

本测试套件基于**真实用户需求**设计，用于评估GRAG系统解决实际问题的能力。测试覆盖**部署配置**、**API使用**、**故障排查**三大类场景，作为回归测试基准确保系统稳定性。

## 测试架构

```
tests/benchmarks/
├── __init__.py                      # 基准测试包
├── benchmark_base.py                # 基准测试基类
├── test_deployment.py               # 部署配置测试
├── test_api_usage.py                # API使用测试
├── test_troubleshooting.py          # 故障排查测试
└── scenarios/                       # 场景定义
    ├── kubernetes_scenarios.json    # Kubernetes场景（9个）
    ├── docker_scenarios.json        # Docker场景（9个）
    └── openclaw_scenarios.json      # OpenClaw场景（9个）
```

## 测试场景

### 场景分类

| 类别 | 工具 | 场景数 | 难度分布 |
|------|------|--------|----------|
| 部署配置 | Kubernetes | 9 | 3基础 + 3中级 + 3高级 |
| API使用 | Docker | 9 | 3基础 + 3中级 + 3高级 |
| 故障排查 | OpenClaw | 9 | 3基础 + 3中级 + 3高级 |
| **总计** | - | **27** | **9基础 + 9中级 + 9高级** |

### 场景示例

#### 部署配置类（Kubernetes）

**基础场景**:
- `k8s-deploy-001`: 创建Nginx Deployment
- `k8s-deploy-002`: 创建Service暴露应用
- `k8s-deploy-003`: 使用ConfigMap配置应用

**中级场景**:
- `k8s-deploy-004`: 配置HPA实现自动扩缩容
- `k8s-deploy-005`: 配置Ingress路由规则
- `k8s-deploy-006`: 使用PersistentVolume存储数据

**高级场景**:
- `k8s-deploy-007`: 配置StatefulSet部署有状态应用
- `k8s-deploy-008`: 配置DaemonSet在所有节点运行
- `k8s-deploy-009`: 创建和使用Helm Chart

#### API使用类（Docker）

**基础场景**:
- `docker-api-001`: Docker Engine API连接和认证
- `docker-api-002`: 列出和管理容器
- `docker-api-003`: 拉取和推送镜像

**中级场景**:
- `docker-api-004`: 使用Python SDK管理容器
- `docker-api-005`: 构建镜像并推送到Registry
- `docker-api-006`: 创建和管理网络

**高级场景**:
- `docker-api-007`: 实现自定义Docker插件
- `docker-api-008`: 监控容器资源使用
- `docker-api-009`: 实现服务发现机制

#### 故障排查类（OpenClaw）

**基础场景**:
- `claw-trouble-001`: 容器启动失败排查
- `claw-trouble-002`: 镜像构建失败
- `claw-trouble-003`: 网络连接问题

**中级场景**:
- `claw-trouble-004`: 内存溢出问题
- `claw-trouble-005`: 磁盘空间不足
- `claw-trouble-006`: 性能瓶颈分析

**高级场景**:
- `claw-trouble-007`: 跨节点通信问题
- `claw-trouble-008`: 数据持久化问题
- `claw-trouble-009`: 集群健康检查

## 快速开始

### 1. 准备知识库

首先摄取相关文档到知识库：

```bash
# Kubernetes文档
uv run scripts/pipeline_fetch.py \
  --repo https://github.com/kubernetes/website \
  --output artifacts/01_raw/kubernetes_docs.json \
  --file-types md \
  --max-files 500

uv run scripts/pipeline_clean.py \
  --input artifacts/01_raw/kubernetes_docs.json \
  --output artifacts/02_clean/kubernetes_docs.json

uv run scripts/pipeline_ingest.py \
  --input artifacts/02_clean/kubernetes_docs.json \
  --enable-bm25

# Docker文档
uv run scripts/pipeline_fetch.py \
  --repo https://github.com/docker/docs \
  --output artifacts/01_raw/docker_docs.json \
  --file-types md \
  --max-files 500

uv run scripts/pipeline_clean.py \
  --input artifacts/01_raw/docker_docs.json \
  --output artifacts/02_clean/docker_docs.json

uv run scripts/pipeline_ingest.py \
  --input artifacts/02_clean/docker_docs.json \
  --enable-bm25

# OpenClaw文档
uv run scripts/pipeline_fetch.py \
  --repo https://github.com/openclaw \
  --output artifacts/01_raw/openclaw_docs.json \
  --file-types md \
  --max-files 500

uv run scripts/pipeline_clean.py \
  --input artifacts/01_raw/openclaw_docs.json \
  --output artifacts/02_clean/openclaw_docs.json

uv run scripts/pipeline_ingest.py \
  --input artifacts/02_clean/openclaw_docs.json \
  --enable-bm25
```

### 2. 运行测试

```bash
# 运行所有测试
uv run tests/run_benchmark_tests.py --full

# 运行特定类别
uv run tests/run_benchmark_tests.py --category deployment
uv run tests/run_benchmark_tests.py --category api_usage
uv run tests/run_benchmark_tests.py --category troubleshooting

# 运行特定难度
uv run tests/run_benchmark_tests.py --difficulty basic
uv run tests/run_benchmark_tests.py --difficulty intermediate
uv run tests/run_benchmark_tests.py --difficulty advanced
```

### 3. 查看报告

测试报告保存在 `artifacts/test_results/benchmarks/` 目录：

```
artifacts/test_results/benchmarks/
├── deployment_report.json          # 部署配置测试报告
├── api_usage_report.json           # API使用测试报告
├── troubleshooting_report.json     # 故障排查测试报告
└── benchmark_report_YYYYMMDD_HHMMSS.json  # 综合报告
```

## 评估标准

### 评分维度

每个场景的答案根据以下维度评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| 查询成功 | 20分 | 查询是否成功执行 |
| 有答案 | 20分 | 是否返回了答案 |
| 有上下文 | 20分 | 是否检索到相关文档 |
| 引用数量 | 20分 | 引用数量是否达标 |
| 命令示例 | 10分 | 是否包含命令示例 |
| 代码示例 | 10分 | 是否包含代码/配置示例 |
| 前置条件 | 10分 | 是否提及前置条件 |
| 主题覆盖 | 10分 | 预期主题覆盖率 |
| **总分** | **100分** | **质量评分** |

### 召回率判断

场景被视为"召回"如果：
- 质量评分 ≥ 50分
- 至少50%的预期主题在答案中出现

### 引用要求

每个场景指定最小引用数量：
- 基础场景：至少1个引用
- 中级场景：至少1个引用
- 高级场景：至少1个引用

## 测试报告

### 报告内容

每个类别的测试报告包含：

```json
{
  "test_class": "DeploymentTest",
  "total_scenarios": 9,
  "success_count": 8,
  "recall_count": 7,
  "success_rate": 0.8889,
  "recall_rate": 0.7778,
  "avg_quality_score": 72.5,
  "avg_response_time": 1.234,
  "results": [...]
}
```

### 综合报告

综合报告包含所有类别的汇总：

```json
{
  "timestamp": "2026-03-16T12:34:56",
  "summary": {
    "total_scenarios": 27,
    "total_success": 24,
    "total_recall": 21,
    "overall_success_rate": 0.8889,
    "overall_recall_rate": 0.7778,
    "avg_quality_score": 72.5,
    "avg_response_time": 1.234
  },
  "categories": {...}
}
```

## 场景定义格式

每个场景遵循以下JSON格式：

```json
{
  "scenario_id": "k8s-deploy-001",
  "category": "deployment",
  "difficulty": "basic",
  "title": "创建Nginx Deployment",
  "description": "用户需要在Kubernetes集群中部署一个简单的Nginx应用",
  "user_query": "如何在Kubernetes中部署Nginx应用？",
  "expected_topics": ["Deployment", "kubectl apply", "容器镜像"],
  "min_citations": 1,
  "quality_criteria": {
    "contains_command": true,
    "contains_example": true,
    "mentions_prerequisites": true
  }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| scenario_id | string | 是 | 场景唯一标识 |
| category | string | 是 | 类别（deployment/api_usage/troubleshooting） |
| difficulty | string | 是 | 难度（basic/intermediate/advanced） |
| title | string | 是 | 场景标题 |
| description | string | 是 | 场景描述 |
| user_query | string | 是 | 用户查询（实际发送给系统的查询） |
| expected_topics | array | 是 | 预期答案应包含的主题 |
| min_citations | int | 是 | 最小引用数量 |
| quality_criteria | object | 否 | 质量标准（可选子字段） |

## 扩展指南

### 添加新场景

1. 在对应的场景文件中添加新场景定义
2. 确保scenario_id唯一
3. 验证JSON格式正确

### 添加新类别

1. 创建新的场景JSON文件
2. 创建对应的测试类继承`BenchmarkTest`
3. 在`run_benchmark_tests.py`中注册新类别

## 最佳实践

### 场景设计原则

1. **真实性**: 场景应基于真实用户需求
2. **可测试性**: 场景应有明确的成功标准
3. **代表性**: 场景应覆盖常见的使用模式
4. **层次性**: 场景应包含不同难度级别

### 查询设计原则

1. **自然语言**: 使用用户会使用的自然语言
2. **具体问题**: 避免过于宽泛的问题
3. **上下文完整**: 包含足够的上下文信息

### 评估标准设定

1. **合理预期**: 根据难度设定合理的评分标准
2. **多维度**: 从多个维度评估答案质量
3. **可量化**: 确保标准可以客观衡量

## 故障排查

### 问题1：场景运行失败

**症状**: 测试执行时报错

**解决方案**:
- 检查知识库是否已摄取相关文档
- 验证场景JSON格式是否正确
- 确认Agent和Model服务正常

### 问题2：召回率低

**症状**: 大量场景未被召回

**解决方案**:
- 检查检索模式配置（建议使用hybrid_bm25）
- 验证BM25索引是否正确构建
- 检查查询是否与文档内容匹配

### 问题3：评分不一致

**症状**: 同一场景多次运行评分差异大

**解决方案**:
- 检查模型温度设置（建议≤0.7）
- 验证缓存是否影响结果
- 确认知识库数据未变化

## 参考资料

- [CLAUDE.md](../CLAUDE.md) - 项目架构指南
- [README.md](../README.md) - 项目总览
- [tests/README.md](tests/README.md) - 测试总览
