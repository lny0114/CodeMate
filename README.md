# CodeMate —— C 语言编程陪练智能体

CodeMate 是一个面向 C 语言初学者的编程陪练 Demo：学生粘贴自己写的 C 代码、描述遇到的问题，
大模型会以“陪练教练”的方式给出诊断和引导，**只提示、不直接给答案**，帮助学生自己把问题改对。

基于 Python + Streamlit 构建，使用兼容 OpenAI 协议的大模型接口（DeepSeek、OpenAI、智谱 GLM、Moonshot 等均可）。

## 功能特性

- 网页化的代码输入框 + 问题描述输入框，点击“开始诊断”一键分析。
- 智能体严格按照七段式结构回答：
  - 【问题判断】
  - 【可能原因】
  - 【关联知识点】
  - 【排查步骤】
  - 【修改方向】
  - 【推荐练习】
  - 【学习评价】
- 内置三份 C 语言学习资料（知识点 / 常见错误 / 练习题库），随每次诊断提供给模型作为参考。
- 全程只做提示、分析和修改方向，**不会直接输出完整可运行代码**，避免学生抄答案。
- 支持流式输出，诊断内容边生成边显示。
- API Key 通过 `.env` 文件读取，也可在网页侧边栏临时覆盖。
- **本地模拟模式（兜底演示）**：未配置 API Key 或大模型调用失败时自动降级，
  不联网、不消耗额度，用内置规则识别「少分号 / 数组越界 / 循环边界」三类演示案例，
  仍按完整七段式输出；也可在侧边栏手动开启。

## 项目结构

```
zhiqi/
├── app.py                    # Streamlit 主程序（页面 + 大模型调用 + 提示词）
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板（复制为 .env 后填入 API Key）
├── README.md
└── knowledge/
    ├── C语言知识点.md         # C 语言核心知识点（初学者版）
    ├── C语言常见错误.md       # 28 个高频错误及自查方法
    └── C语言练习题库.md       # 27 道分级练习（只给提示，不给答案）
```

## 运行步骤（Windows）

1. 安装 Python 3.9 或更高版本（命令行输入 `python --version` 可确认）。

2. 进入项目目录，创建并激活虚拟环境（推荐）：

   ```bat
   cd c:\Users\15412\Documents\trae_projects\zhiqi
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. 安装依赖：

   ```bat
   pip install -r requirements.txt
   ```

4. 配置 API Key：复制 `.env.example` 为 `.env`，用文本编辑器打开，填入你的大模型 API Key：

   ```bat
   copy .env.example .env
   ```

   `.env` 中需要配置三项（默认使用 DeepSeek，可按需改成其他厂商）：

   ```env
   LLM_API_KEY=sk-你的真实APIKey
   LLM_BASE_URL=https://api.deepseek.com/v1
   LLM_MODEL=deepseek-chat
   ```

5. 启动应用：

   ```bat
   streamlit run app.py
   ```

   启动后浏览器会自动打开 http://localhost:8501 ；如果没有自动打开，手动访问该地址即可。

## 使用方法

1. 在“① 粘贴你的 C 语言代码”框中粘贴代码（问题描述选填）。
2. 点击“开始诊断”。
3. 阅读七段式诊断结果，按照【排查步骤】自己动手验证，按【修改方向】自行修改。
4. 修改后可以再次提交，形成“提交 → 诊断 → 修改 → 再诊断”的学习闭环。

## 本地模拟模式（无需 API Key 也能演示）

以下三种情况会进入本地模拟模式，页面会显示明显提示条，且结果下方标注“来自本地模拟模式”：

1. 侧边栏勾选「使用本地模拟模式」——手动开启，适合课堂/路演演示；
2. 未配置有效的 `LLM_API_KEY`（包括 `.env` 里仍是占位符）——自动降级，不再直接报错；
3. 已配置 Key 但调用失败（网络错误、Key 无效、限流等）——自动降级，并显示失败原因。

模拟模式通过简单规则识别三类内置演示案例，输出与真实大模型**完全相同的七段式结构**：

| 演示案例 | 识别方式（举例） |
| --- | --- |
| 少分号 | 变量定义 / 赋值 / return / printf 等语句行末缺少 `;`；或问题描述中提到“分号” |
| 数组越界 | `int a[5]` 后出现 `a[5]`；或遍历循环写成 `i <= 5` 且用 `a[i]`；问题中提到“越界/下标/数组” |
| 循环边界 | `for` 循环条件出现 `<=`（差一错误嫌疑）；或问题中提到“边界/差一/多一次/少一次” |

无法匹配时返回一份通用排查七段式模板。模拟结果是固定文案，仅用于演示完整流程；
配置有效 API Key 后即可获得针对真实代码的逐行分析。

## 支持的大模型接口

任何兼容 OpenAI Chat Completions 协议的服务都可以，只需修改 `.env` 中的地址与模型名：

| 厂商 | LLM_BASE_URL | LLM_MODEL 示例 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

> 安全提示：`.env` 中包含 API Key，请勿把它提交到公开代码仓库或分享给他人。

## 部署到 Streamlit Community Cloud（免费）

本项目已适配 Streamlit Community Cloud：依赖在 `requirements.txt` 中声明，
密钥支持通过平台的 **Secrets** 注入（无需上传 `.env`），未配置密钥时还有本地模拟模式兜底。

### 第一步：上传到 GitHub

1. 在 https://github.com 登录后新建一个仓库（例如 `CodeMate`），**不要**勾选 Add README/.gitignore（本地已经有了）。公开或私有均可（Streamlit Cloud 都支持）。
2. 在项目目录执行（把下方地址换成你自己仓库的地址）：

   ```bat
   cd c:\Users\15412\Documents\trae_projects\zhiqi
   git init
   git add .
   git status
   ```

3. **关键检查**：在 `git status` 输出中确认看不到 `.env`、`.venv/`、`__pycache__/`（它们已被 `.gitignore` 排除；应能看到 `.env.example`）。确认无误后提交并推送：

   ```bat
   git commit -m "Initial commit: CodeMate C language tutor"
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/CodeMate.git
   git push -u origin main
   ```

   > 如果曾经误提交过 `.env`，仅删除文件再提交是不够的（历史记录里仍有密钥），
   > 需清理 git 历史，并**立即到对应大模型平台吊销（revoke）该 Key 后重新生成**。

### 第二步：在 Streamlit Cloud 部署

1. 打开 https://share.streamlit.io ，用 GitHub 账号登录。
2. 点击 **Create app** → **Deploy a public app from GitHub**，选择：
   - Repository：你刚上传的 `CodeMate` 仓库；
   - Branch：`main`；
   - Main file path：`app.py`。
3. 点开 **Advanced settings**，在 **Secrets** 框中粘贴以下 TOML（把 Key 换成你自己的真实 Key）：

   ```toml
   LLM_API_KEY = "sk-你的真实APIKey"
   LLM_BASE_URL = "https://api.deepseek.com/v1"
   LLM_MODEL = "deepseek-chat"
   ```

4. 点击 **Deploy**，等待 1~2 分钟即可获得一个公开访问地址（形如 `https://<你的用户名>-codemate.streamlit.app`）。

注意事项：

- 密钥只填在 Streamlit Cloud 网页的 Secrets 里，**不要**写进代码或提交到 GitHub；以后可随时在 App 的 ⋮ 菜单 → Settings → Secrets 中修改。
- 即使暂不配置 Secrets，应用也能正常启动并自动使用本地模拟模式演示三类案例。
- 以后推送代码到 GitHub 的 `main` 分支，云端应用会自动重新部署；也可在网页菜单里手动 **Reboot**。
- 免费版资源有限且应用一段时间无人访问会休眠（再次访问会自动唤醒，需等待几十秒），适合 Demo 与教学使用。

## 自定义学习资料

`knowledge/` 目录下的三份 Markdown 文件会在每次诊断时自动加载进模型的上下文。
直接编辑这些文件即可扩充知识点、错误案例或练习题，无需改动代码；
新增资料文件后，在 `app.py` 的 `KNOWLEDGE_FILES` 列表中追加文件名即可生效。

## 常见问题

- **没有 API Key 能用吗？** 可以：直接启动并点击“开始诊断”即自动进入本地模拟模式，也可在侧边栏手动勾选。
- **想确认为什么走了模拟模式**：查看诊断结果上方的提示条——手动开启显示蓝色，无 Key/调用失败显示黄色并附带原因。
- **调用大模型失败**：检查 Key 是否有效、是否欠费、`LLM_BASE_URL` 和 `LLM_MODEL` 是否匹配，以及网络能否访问该接口；当前即使失败也会自动降级到模拟模式，不会中断演示。
- **端口被占用**：可换端口启动：`streamlit run app.py --server.port 8502`。
