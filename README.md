# Med-MARAG

基于毕业论文《面向医疗领域的 LLM 驱动型需求分析与 UML 辅助建模技术》实现的原型项目。系统围绕论文中的主线能力构建：

- 医疗领域 RAG 知识增强
- Analyst / Architect / Reviewer 多智能体闭环
- 自然语言需求到 EARS 标准化需求的转换
- 用例图、类图、序列图的 PlantUML 自动生成
- 测试用例与需求追溯矩阵自动产出

## 项目结构

- `workflow/ocean_graph.py`
  论文中的多智能体主工作流，支持 LangGraph 和无依赖顺序执行两种模式。
- `agents/`
  Analyst、Modeling、Review 三个智能体实现。
- `rag/knowledge_base.py`
  医疗与需求工程知识库，支持 Chroma 向量检索和本地关键字检索。
- `utils/requirement_utils.py`
  需求解析、EARS 生成、用例制品构造。
- `utils/plantuml_utils.py`
  用例图、类图、序列图 PlantUML 生成。
- `test_case_generator.py`
  基于需求自动生成测试用例。
- `traceability_matrix.py`
  生成需求-设计-测试追溯矩阵。
- `app.py`
  Streamlit 可视化前端。
- `run_pipeline.py`
  命令行运行入口。

## 快速开始

### 1. 命令行离线演示

```bash
.venv/bin/python run_pipeline.py "当患者预约挂号成功时，系统应发送确认短信给患者" --no-llm
```

如果想看完整 JSON：

```bash
.venv/bin/python run_pipeline.py "当护士完成分诊评估时，系统应生成分诊任务并更新排队状态" --no-llm --json
```

### 2. 交互式终端模式

```bash
.venv/bin/python main.py
```

### 3. Streamlit 前端

当前虚拟环境里如果还没有安装 `streamlit`，先补装依赖后再运行：

```bash
streamlit run app.py
```

### 4. 指定 Provider 运行

离线模式：

```bash
.venv/bin/python run_pipeline.py "当患者预约挂号成功时，系统应发送确认短信给患者" --provider offline
```

DeepSeek 模式：

```bash
.venv/bin/python run_pipeline.py "当患者预约挂号成功时，系统应发送确认短信给患者" --provider deepseek
```

OpenAI 模式：

```bash
.venv/bin/python run_pipeline.py "当患者预约挂号成功时，系统应发送确认短信给患者" --provider openai
```

### 5. 结构化需求输入集转换

项目支持将需求输入集从 `.xlsx` 或 `.csv` 转换为统一的结构化格式 `id/category/requirement`。当前也兼容你使用的双列表格形式：第一列为分类，第二列为需求句子，无表头。

```bash
conda run -n Med-MARAG-py311 python scripts/convert_requirements_csv.py \
  /Users/seaocean/Documents/毕业设计/reqirement_input.xlsx \
  --out-csv output/requirements_input.csv \
  --out-jsonl output/requirements_input.jsonl
```

## 环境变量

复制 `.env.example` 到 `.env` 后按需填写。推荐优先使用通用变量 `LLM_*`，同时兼容历史的 `DEEPSEEK_*` 配置：

```env
LLM_PROVIDER=offline
LLM_API_KEY=your_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1

PROJECT_ROOT=/Users/seaocean/projects/Python/Med-MARAG
```

没有配置 API Key、provider 非法、或在线客户端初始化失败时，系统会自动回退为离线规则模式，仍可完成论文答辩所需的主链路演示。

## 当前实现特点

- 优先保证离线可运行，避免因网络或模型不可用而阻塞演示
- 已支持 `offline / deepseek / openai` 三种 provider 选择，并在运行结果中显示实际执行来源
- 输出内容覆盖论文中的核心成果物：
  - EARS 标准化需求
  - 用例制品
  - UML PlantUML 代码
  - 测试用例
  - 追溯矩阵
  - 审查反馈

## 适合答辩的演示流程

1. 输入 1-3 条医疗场景需求。
2. 展示 EARS 标准化结果。
3. 展示用例图、类图、序列图代码。
4. 展示 Reviewer 审查结果和评分。
5. 展示自动生成的测试用例与追溯矩阵。
6. 说明系统支持 RAG、多智能体闭环与多 provider LLM 增强，并可在缺失密钥时自动回退到离线模式。
