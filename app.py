# -*- coding: utf-8 -*-
"""
CodeMate —— C 语言编程陪练智能体
技术栈：Python + Streamlit + OpenAI 兼容大模型 API（DeepSeek / OpenAI / 智谱 / Moonshot 等）

运行方式：
    streamlit run app.py
"""

import os
import re
import traceback
import unicodedata
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 路径与环境变量
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_FILES = [
    "C语言知识点.md",
    "C语言常见错误.md",
    "C语言练习题库.md",
]

# 从项目根目录的 .env 文件读取配置（本地开发用；Streamlit Cloud 上没有该文件时会静默跳过）
load_dotenv(BASE_DIR / ".env")

# Streamlit Community Cloud：平台通过 st.secrets（在 App 设置的 Secrets 中以 TOML 配置）
# 注入密钥。这里把 st.secrets 桥接到环境变量，使后续统一用 os.getenv 读取，
# 本地与云端代码路径保持一致；已由 .env / 系统环境变量提供的值优先，不覆盖。
try:
    for _secret_key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if not os.getenv(_secret_key) and _secret_key in st.secrets:
            os.environ[_secret_key] = str(st.secrets[_secret_key])
except Exception:
    # 本地未配置任何 secrets 时 st.secrets 可能不可用，忽略即可
    pass


# ---------------------------------------------------------------------------
# API Key 清洗与校验
# ---------------------------------------------------------------------------
# 零宽字符、BOM 等不可见字符会导致鉴权莫名其妙失败，这里统一清理。
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e]")


def sanitize_api_key(value: str) -> tuple[str, dict]:
    """清洗 API Key，并返回 (清洗后的 key, 元信息)。

    元信息包含：is_ascii、starts_with_sk、length、masked（前 3 位 + 长度），
    不包含完整 key，可安全展示。
    """
    if value is None:
        value = ""
    # 1) unicode 规范化（NFKC 可让全角字符变半角等）
    value = unicodedata.normalize("NFKC", value)
    # 2) 去掉零宽字符与 BOM
    value = _ZERO_WIDTH_RE.sub("", value)
    # 3) 去掉首尾空白（含空格、换行、制表符）
    value = value.strip()
    # 4) 去掉首尾成对的引号（英文单双引号、中文引号）
    while len(value) >= 2 and value[0] in "\"'\u201c\u2018\u300c\u300e" and value[-1] in "\"'\u201d\u2019\u300d\u300f":
        value = value[1:-1]
    # 5) 用户误填 "Bearer sk-xxx" 时去掉 Bearer 前缀
    if value.lower().startswith("bearer "):
        value = value[len("Bearer "):].strip()
    # 去掉前缀后再次去引号
    while len(value) >= 2 and value[0] in "\"'\u201c\u2018\u300c\u300e" and value[-1] in "\"'\u201d\u2019\u300d\u300f":
        value = value[1:-1]

    length = len(value)
    masked = f"{value[:3]}... (长度 {length})" if length else "(空)"
    meta = {
        "is_ascii": all(ord(c) < 128 for c in value),
        "starts_with_sk": value.startswith("sk-"),
        "length": length,
        "masked": masked,
    }
    return value, meta


def is_placeholder_key(api_key: str) -> bool:
    """识别 .env 模板里未替换的占位符假 Key。"""
    cleaned, _ = sanitize_api_key(api_key)
    return (not cleaned) or ("在这里" in cleaned) or ("替换" in cleaned)


# ---------------------------------------------------------------------------
# 智能体系统提示词（规定角色、七段式回答格式、禁止直接给完整代码）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是 CodeMate，一位耐心、专业的 C 语言编程陪练老师，对话对象是 C 语言初学者。

你的任务：阅读学生提交的 C 语言代码和他描述的问题，像陪练教练一样引导学生自己发现问题、解决问题。

【最高原则】
1. 绝对不要直接给出完整的、可以原样提交运行的修正代码。你只能做这些事：
   - 指出问题所在（可以引用学生代码中的极短片段或具体行，并说明错在哪）；
   - 讲解原理、分析可能原因；
   - 给出提示、排查思路和修改方向；
   - 必要时给出 1~3 行的关键语法小例子（用于说明知识点，不能是整道题的完整答案）。
2. 语言要通俗、鼓励人，避免堆砌术语；必须使用术语时顺手用一句话解释。
3. 充分参考下方“学习资料库”中的内容组织回答，并按学生水平控制难度。

【回答格式 —— 必须严格遵守】
只能依次输出以下七个小标题，标题文字、顺序都不能改变、不能省略、不能合并，每个标题下写对应内容：

【问题判断】
（用 1~3 句话点出最核心的问题：是编译错误、运行结果错误、逻辑错误，还是概念疑问）

【可能原因】
（分条列出导致该问题的可能原因，尽量对应到代码中的具体位置）

【关联知识点】
（列出相关的 C 语言知识点，每个知识点附一句通俗解释）

【排查步骤】
（给出学生可以自己动手做的检查动作，例如打印某个变量的值、检查分号/括号配对、检查 scanf 的 & 等）

【修改方向】
（只描述“哪里需要改、往什么方向改、为什么”，禁止给出完整可运行代码）

【推荐练习】
（从题库中推荐 1~3 个相关练习，写清练习名称和做这个练习能巩固什么）

【学习评价】
（用 1~2 句话评价学生当前的掌握情况，给出鼓励和下一步学习建议）

==================== 学习资料库 ====================
{knowledge}
==================== 资料结束 ====================
"""


# ---------------------------------------------------------------------------
# 知识加载（缓存，避免每次交互重复读盘）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_knowledge() -> str:
    """读取 knowledge 目录下的三份学习资料，拼接到系统提示词中。"""
    chunks = []
    for filename in KNOWLEDGE_FILES:
        path = KNOWLEDGE_DIR / filename
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            chunks.append(f"## 资料：{filename}\n\n{content}")
        else:
            chunks.append(f"## 资料：{filename}\n\n（该资料文件缺失，请提醒学生补充）")
    return "\n\n---\n\n".join(chunks)


def build_user_message(code: str, question: str) -> str:
    """把学生代码和问题组装成发送给大模型的用户消息。"""
    parts = [
        "学生提交的 C 语言代码如下：",
        "",
        "```c",
        code.strip(),
        "```",
    ]
    if question.strip():
        parts += ["", "学生补充描述的问题 / 疑问：", question.strip()]
    else:
        parts += ["", "学生没有额外描述问题，请你根据代码主动判断最可能存在的问题。"]
    return "\n".join(parts)


def diagnose(api_key: str, base_url: str, model: str, code: str, question: str):
    """流式调用大模型，逐段产出诊断文本。

    传入的 api_key 会先经过 sanitize_api_key 清洗（去零宽字符、去引号、去 Bearer 前缀等），
    保证不会因为复制粘贴带入的不可见字符导致鉴权失败。
    """
    cleaned_key, _ = sanitize_api_key(api_key)
    client = OpenAI(api_key=cleaned_key, base_url=base_url.strip())
    stream = client.chat.completions.create(
        model=model.strip(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(knowledge=load_knowledge())},
            {"role": "user", "content": build_user_message(code, question)},
        ],
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content


# ---------------------------------------------------------------------------
# 本地模拟模式
# 当没有 API Key、调用大模型失败或用户手动开启时使用：
# 不联网、不消耗额度，用规则识别三类演示案例并返回固定七段式诊断。
# ---------------------------------------------------------------------------
SECTION_TITLES = [
    "【问题判断】",
    "【可能原因】",
    "【关联知识点】",
    "【排查步骤】",
    "【修改方向】",
    "【推荐练习】",
    "【学习评价】",
]

C_TYPES = r"(?:int|long|short|char|float|double|unsigned|signed|const)"


def _strip_comments(code: str) -> str:
    """去掉 C 代码中的块注释和行注释，避免注释内容干扰规则识别。"""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return code


def _detect_missing_semicolon(code: str):
    """识别“少分号”：找到疑似漏写分号的语句行，返回 (True, 该行内容)。"""
    control_header = re.compile(r"^(?:else\s+)?(?:if|for|while|switch)\b")
    type_keyword = re.compile(r"^" + C_TYPES + r"\b")
    plain_assign = re.compile(r"(?<![=!<>+\-*/%&|^])=(?!=)")          # a = b
    compound_assign = re.compile(r"(?:\+=|-=|\*=|/=|%=)")             # a += b
    for raw in _strip_comments(code).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 正常结尾 / 续行 / 分支标签，跳过
        if line.endswith((";", "{", "}", ":", ",", "\\", "&&", "||")):
            continue
        # if/for/while/switch（含 else if）的控制行，以 ) 结尾是正常的
        if control_header.match(line) and line.endswith(")"):
            continue
        # 函数定义/调用以 ) 结尾通常也正常；但 return/printf/scanf 除外
        if line.endswith(")") and not re.match(r"^(?:return|printf|scanf)\b", line):
            continue
        looks_like_statement = (
            type_keyword.match(line)
            or plain_assign.search(line)
            or compound_assign.search(line)
            or re.match(r"^(?:return|printf|scanf)\b", line)
        )
        if looks_like_statement:
            return True, line
    return False, None


def _detect_array_out_of_bounds(code: str):
    """识别“数组越界”，返回 (案例描述, None) 或 None。

    两种命中方式：
    1. 字面量下标越界，如 int a[5] 后出现 a[5]；
    2. 循环条件 <= 数组长度，且用该循环变量访问数组。
    """
    src = _strip_comments(code)
    decl_pattern = re.compile(r"\b" + C_TYPES + r"\s+(\w+)\s*\[\s*(\d+)\s*\]")
    arrays = {name: int(size) for name, size in decl_pattern.findall(src)}
    if not arrays:
        return None

    # 去掉声明语句，避免把 a[5] 声明本身误判成访问
    src_without_decl = decl_pattern.sub(" ", src)

    # 1) 字面量下标
    for name, size in arrays.items():
        for idx in re.findall(rf"\b{name}\s*\[\s*(\d+)\s*\]", src_without_decl):
            if int(idx) >= size:
                return (
                    f"检测到数组 `{name}` 的长度为 {size}（合法下标为 0~{size - 1}），"
                    f"但代码中出现了 `{name}[{idx}]`，已经越过最后一个元素。"
                )

    # 2) for 循环条件用 <= 数组长度，且循环体里用该变量做下标
    for match in re.finditer(r"for\s*\(([^)]*)\)", src):
        header = match.group(1)
        bound_match = re.search(r"(\w+)\s*<=\s*(\d+)", header)
        if not bound_match:
            continue
        var, bound = bound_match.group(1), int(bound_match.group(2))
        for name, size in arrays.items():
            if size == bound and re.search(rf"\b{name}\s*\[\s*{var}\s*\]", src):
                return (
                    f"检测到数组 `{name}` 的长度为 {size}，但遍历它的循环条件写成了"
                    f"`{var} <= {size}`，最后一轮会访问不存在的 `{name}[{size}]`。"
                )
    return None


def _detect_loop_boundary(code: str) -> bool:
    """识别“循环边界”：for 循环中出现 <=，存在差一错误嫌疑。"""
    return re.search(r"for\s*\([^;]*;[^;]*<=\s*[^;]+;", _strip_comments(code)) is not None


def _extract_block_body(src: str, open_brace_idx: int) -> str:
    """从 src[open_brace_idx] 处的 '{' 开始，提取到大括号块内部的内容。"""
    depth = 0
    i = open_brace_idx
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace_idx + 1:i]
        i += 1
    return src[open_brace_idx + 1:]


def _detect_init_position(code: str):
    """识别“累计变量 / max / min 初始化位置或初始值不当”。

    两种子问题：
    1. sum/count/max/min 在循环体内被重复初始化（每轮重置）；
    2. max = 0（而非用数组首元素初始化），全负数输入时会得到错误结果。

    返回 (detail, None) 或 None。
    """
    src = _strip_comments(code)

    # 子问题 1：在 for/while 循环体内部出现 sum/count/max/min = 0 或 int xxx = 0
    loop_vars = ("sum", "count", "max", "min")
    init_inside = []
    for m in re.finditer(r"\b(?:for|while)\s*\([^)]*\)\s*\{", src):
        body = _extract_block_body(src, m.end() - 1)
        for var in loop_vars:
            # 匹配 var = 0 或 var = 其它字面量 / int var = ...
            if re.search(rf"\b{var}\s*=\s*(?:0|\d+)", body) or \
               re.search(rf"\bint\s+{var}\s*=", body) or \
               re.search(rf"\b{var}\s*\+=\s*0", body):
                init_inside.append(var)

    if init_inside:
        return (
            f"检测到累计变量 `{'`、`'.join(init_inside)}` 在循环体内部被初始化（如 `= 0`），"
            "这会导致它每一轮都被重置，累加/求最值的结果自然不对。"
        )

    # 子问题 2：max 被初始化为 0，但代码看起来在求数组/序列的最大值（全负数时 0 会“伪胜利”）
    if re.search(r"\bmax\s*=\s*0\b", src) or re.search(r"\bint\s+max\s*=\s*0\b", src):
        has_array_compare = re.search(r"a\w*\s*\[\s*\w+\s*\]\s*>\s*max", src) or \
                            re.search(r"max\s*=\s*\w+\s*\[\s*\w+\s*\]", src)
        if has_array_compare or re.search(r"\bfor\b", src):
            return (
                "检测到求最大值时把 `max` 初始化成了 `0`。如果输入数据全是负数，"
                "0 会一直“赢”，最终得到错误的最大值 0。"
            )

    return None


def detect_demo_case(code: str, question: str):
    """综合问题描述关键词与代码规则，判断演示案例类型。

    返回 (case_key, detail)：case_key ∈ semicolon / array / loop / init / generic。
    """
    text = question or ""
    # 学生在问题里直接点名了问题类型时，优先采信
    if re.search(r"分号", text):
        return "semicolon", None
    if re.search(r"越界|下标|数组", text):
        return "array", None
    if re.search(r"边界|差一|多一次|少一次|多一圈|少一圈", text):
        return "loop", None
    if re.search(r"初始化|最大值|最小值|max|min|累加|累计", text, re.IGNORECASE):
        return "init", None

    array_detail = _detect_array_out_of_bounds(code)
    if array_detail:
        return "array", array_detail
    init_detail = _detect_init_position(code)
    if init_detail:
        return "init", init_detail
    missing, _ = _detect_missing_semicolon(code)
    if missing:
        return "semicolon", None
    if _detect_loop_boundary(code):
        return "loop", None
    return "generic", None


MOCK_TEMPLATES = {
    "semicolon": [
        # 【问题判断】
        "这是一个**编译错误**：你的代码里很可能有一条语句在结尾处漏掉了分号 `;`，导致程序无法编译通过。\n\n"
        "提示：C 编译器报告的行号常常比真正出错的位置**晚一行**——往报错行的上一行看，往往就能找到元凶。",
        # 【可能原因】
        "- 变量定义或赋值语句（如 `int a = 0`、`sum = sum + i`）末尾漏写分号；\n"
        "- `return`、`printf`、`scanf` 等语句末尾漏写分号；\n"
        "- 从上一行复制语句后只改了内容、忘了补 `;`；\n"
        "- 注意：`#include`、`#define` 这类预处理指令后面**不需要**分号，别补错地方。",
        # 【关联知识点】
        "- **分号是语句的结束标志**：它告诉编译器“一句话说完了”，而不是“一行写完了”；\n"
        "- **编译错误**：语法不合规时编译器拒绝生成程序，必须先修掉才能运行；\n"
        "- **报错定位技巧**：优先看第一条 error，并检查它指向行号的上一行。",
        # 【排查步骤】
        "1. 找到编译器的第一条 error，记下它给出的行号；\n"
        "2. 从该行**往上**逐行检查：变量定义、赋值、return、printf/scanf 是否都以 `;` 结尾；\n"
        "3. 用有符号高亮的编辑器逐行核对，重点看刚改过或复制来的行；\n"
        "4. 编译时加上 `-Wall` 参数（如 `gcc -Wall hello.c -o hello`），让编译器把可疑处都提示出来。",
        # 【修改方向】
        "在漏写的那条语句末尾补一个**英文半角分号 `;`**。注意只加在普通语句末尾：函数体的大括号 `{}`、"
        "`if/for/while` 的控制行、`#include` 行都不要加分号。请你自己根据上面的排查方法定位具体行并修改，"
        "我不直接替你改——自己找到一次，下次就不会再漏。",
        # 【推荐练习】
        "- **练习 1：个人名片**（★）——熟悉 printf 语句与分号的基本书写；\n"
        "- **练习 2：两数之和**（★）——练习变量定义、scanf 与赋值语句的规范收尾。\n"
        "做完后刻意全文扫一遍：是不是每条语句都有分号？",
        # 【学习评价】
        "漏分号是所有 C 语言初学者都会经历的“入门第一课”，这不是粗心，而是还没建立对 C 语法边界的感觉。"
        "建议以后每写两三行就扫一眼行尾，很快就会形成肌肉记忆，加油！",
    ],
    "array": [
        "这是一个典型的**数组下标越界**问题（可能表现为运行崩溃、结果随机，也可能只有警告甚至“看起来正常”）："
        "代码访问了数组合法范围之外的元素。",
        "- 长度为 n 的数组，合法下标范围是 **0 ~ n-1**，并不存在“第 n 个元素”；\n"
        "- 遍历时循环条件写成了“下标 <= n”，最后一轮就会越界；\n"
        "- 也可能是直接用字面量访问，如对长度为 5 的数组写 `a[5]`；\n"
        "- C 语言通常**不会自动检查越界**：程序不报错，不代表没出错，它可能已悄悄改掉了别的数据。",
        "- **数组下标从 0 开始**：`int a[5]` 的五个元素是 `a[0]` 到 `a[4]`；\n"
        "- **越界访问的后果**：读到的是别的内存里的垃圾值；往里写则可能覆盖无关变量；\n"
        "- **遍历标准形式**：下标从 0 开始、循环条件用“小于长度”，两者必须配套。",
        "1. 先确认数组定义时的长度，写出它的合法下标范围；\n"
        "2. 在循环体里临时 `printf` 打印下标变量的值，观察它最后一次取到几；\n"
        "3. 检查循环条件用的是 `<` 还是 `<=`，和下标起始值（0 还是 1）是否配套；\n"
        "4. 逐个检查代码里的 `数组名[...]`，确认方括号中的最大值不超过“长度 - 1”。",
        "两个方向任选其一，请你自己动手：① 保持下标从 0 开始，把循环终止条件从“小于等于长度”"
        "调整为“小于长度”；② 如果业务上坚持从 1 开始计数，访问数组时就要用“计数变量 - 1”。"
        "改完后再次打印下标，验证最后一轮停在合法位置。",
        "- **练习 16：成绩统计**（★）——巩固数组遍历与循环边界；\n"
        "- **练习 17：数组逆序输出**（★★）——两端下标向中间靠拢，最容易暴露越界问题；\n"
        "- **练习 18：冒泡排序**（★★）——练习嵌套循环中的边界控制。",
        "你已经学到数组这一关，说明基础语法在稳步推进。越界是 C 语言最“隐蔽”的错误之一，"
        "养成“写循环先想首尾下标”的习惯后，这类问题会越来越少，继续加油！",
    ],
    "loop": [
        "这是一个疑似**循环边界（差一）错误**：循环体实际执行的次数比你想要的多一次或少一次，"
        "属于逻辑错误——程序能编译运行，但结果不对。",
        "- 起点与终止条件不配套：下标从 0 开始却用 `<= n`，或从 1 开始却用 `< n`；\n"
        "- 混淆了“让循环体执行 n 次”和“访问下标 0~n-1”两种计数方式；\n"
        "- 更新语句的方向、步长与条件不一致时，也可能多转、少转甚至死循环。",
        "- **for 的执行流程**：初始化只做 1 次 → 判断条件 → 执行循环体 → 执行更新 → 再判断；\n"
        "- **差一错误（Off-by-one）**：边界只差 1，是编程中最常见的逻辑错误之一；\n"
        "- **首尾代入法**：把循环变量第一次和最后一次的取值分别代入推演，是检查边界的利器。",
        "1. 先想清楚你希望循环体到底执行多少次；\n"
        "2. 在纸上代入循环变量的**第一次**取值，确认这一轮应该执行；\n"
        "3. 再代入你以为的**最后一次**取值，以及“再多走一次”的值，确认该在哪里停下；\n"
        "4. 可在循环体内打印循环变量和一个自增计数器，用真实输出核对执行次数。",
        "统一“起点 + 条件”的搭配规则：下标从 0 开始遍历 n 个元素时，条件应表达“小于 n”；"
        "从 1 开始计数到 n 时，才使用“小于等于 n”。请根据你代码的真实语义（数组下标还是纯计数），"
        "自行调整起始值或条件中的不等号。",
        "- **练习 10：1 到 100 求和**（★）——最适合体会循环次数与累加器初始化；\n"
        "- **练习 13：判断素数**（★★）——练习用标记变量控制循环提前结束；\n"
        "- **练习 18：冒泡排序**（★★）——综合训练嵌套循环边界。",
        "边界感是循环学习中最值得打磨的基本功。你已经开始怀疑“循环次数”这一层，"
        "说明正在像程序员一样思考问题，多用首尾代入法练几题就会越来越稳！",
    ],
    "init": [
        "问题属于**初始化位置或初始值选择不当**：程序能编译运行，但最大值、最小值、求和、计数等结果不对。"
        "这类错误很隐蔽——没有报错，只是数值差一点，容易让人怀疑编译器。",
        "- 累计变量（sum、count）被放在了循环体内部初始化，每一轮都被重置成 0，最终只剩最后一轮的值；\n"
        "- 求最大值时把 `max` 初始化为 `0`，如果输入全是负数，0 永远比任何输入都大，结果就错了；\n"
        "- 同理 `min = 0` 在全正数输入时也会出问题；\n"
        "- 变量作用域没理清：以为“定义在循环外”，实际写成了“循环内的局部变量”。",
        "- **变量作用域**：在 `{}` 内定义的变量只在该块内有效，循环每次进入都会重新创建；\n"
        "- **累计变量的标准位置**：`sum = 0`、`count = 0` 必须写在循环**之前**；\n"
        "- **最大值/最小值的初始值选择**：不知道数据范围时，用**第一个元素**初始化最安全，比用 0 更通用。",
        "1. 找到 `sum` / `count` / `max` / `min` 被赋值为 0（或其它初值）的那一行，确认它在循环的**外面还是里面**；\n"
        "2. 在循环体里临时 `printf` 打印每一轮结束后该变量的值，看它是不是每轮都被重置；\n"
        "3. 如果是 max/min 问题，构造一组全负数（求最大值）或全正数（求最小值）的测试数据跑一遍，看结果是否正确；\n"
        "4. 在纸上手动跟踪两轮循环，记录变量值的变化，问题通常一眼就能看出来。",
        "两个修改方向，请你自己动手：① 把 `sum=0` / `count=0` 这类初始化语句**移到循环之前**；"
        "② 求最大值/最小值时，把 `max=0` 改成**用数组第一个元素初始化**（如 `max = a[0]`，循环从下标 1 开始），"
        "这样无论输入正负都不会出错。改完后用全负数测试再验证一次。",
        "- **练习 10：1 到 100 求和**（★）——最直接的累加器初始化练习；\n"
        "- **练习 16：成绩统计**（★）——同时练 sum、max、min 三个累计变量的初始化位置；\n"
        "- **练习：找出数组中的最大值与最小值**（★★）——重点练习用首元素初始化 max/min，而非写死 0。",
        "初始化位置是循环和数组这一关的核心概念，你已经能定位到“结果不对”这一层，说明调试直觉在进步。"
        "记住一个口诀：**累计变量在循环外初始化，循环内只更新不重置**，多练两题就能形成肌肉记忆。",
    ],
    "generic": [
        "本地模拟模式暂时无法确定你代码中的具体问题（演示版内置识别四类案例：少分号、数组越界、循环边界、初始化位置）。"
        "下面给出一份通用排查指引，按顺序走一遍，通常能定位大部分入门问题。",
        "- 先看编译器的**第一条**报错信息及其行号（必要时往上看一行）；\n"
        "- 高频原因：漏分号、括号不配对、变量未定义、`scanf` 漏写 `&`、占位符与类型不匹配；\n"
        "- 能编译但结果不对时，多为逻辑问题：循环次数、累加器初始化位置、整数除法、"
        "条件里把 `==` 误写成 `=`。",
        "- **编译错误 / 运行错误 / 逻辑错误**：先分清类别，三类问题的排查方向完全不同；\n"
        "- **printf 调试法**：在关键位置打印中间变量，是最简单有效的排查手段；\n"
        "- **最小化问题**：把可疑代码单独抽成一个小程序反复验证。",
        "1. 重新编译并加上 `-Wall`，通读全部警告和错误；\n"
        "2. 从第一条报错开始修，**每修一处就重新编译一次**（后面的报错可能只是连锁反应）；\n"
        "3. 能运行后，在关键计算前后打印变量，和你的预期值逐项对比；\n"
        "4. 仍无法定位时，把问题缩小到最小代码片段再分析。",
        "请按上面的定位结果自行修改：每次只改一处，改完立即重新编译验证，"
        "避免同时改多处而无法判断哪一处真正生效。配置有效 API Key 后，"
        "即可获得针对你这份代码的逐行详细分析。",
        "- **练习 2：两数之和**（★）——检查输入输出与基础语句；\n"
        "- **练习 10：1 到 100 求和**（★）——检查循环与累加器初始化；\n"
        "- **练习 16：成绩统计**（★）——检查数组遍历边界。",
        "主动提交代码寻求诊断本身就是很好的学习习惯！本地模拟模式只能覆盖三类演示案例，"
        "配上 API Key 后我就能针对你的真实代码做详细分析，继续保持多练多问。",
    ],
}


def build_mock_diagnosis(code: str, question: str) -> str:
    """按七段式固定格式组装一份本地模拟诊断结果。"""
    case, detail = detect_demo_case(code, question)
    bodies = MOCK_TEMPLATES[case][:]
    if detail:
        # 把规则检测到的具体证据插到【可能原因】最前面
        bodies[1] = detail + "\n\n" + bodies[1]
    sections = [f"{title}\n\n{body}" for title, body in zip(SECTION_TITLES, bodies)]
    return "\n\n".join(sections)


def mock_diagnose(code: str, question: str):
    """模拟诊断的流式接口：逐行产出，与大模型流式路径保持同构。"""
    for line in build_mock_diagnosis(code, question).split("\n"):
        yield line + "\n"


# ---------------------------------------------------------------------------
# Streamlit 页面
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeMate C语言编程陪练智能体",
    page_icon="🩺",
    layout="centered",
)

st.title("CodeMate C语言编程陪练智能体")
st.caption("粘贴你的 C 语言代码，说说你遇到的问题，我会陪你一起排查 —— 但不会直接替你写出答案。")

# ---------------- 侧边栏：模型配置（默认值来自 .env） ----------------
with st.sidebar:
    st.header("模型配置")
    st.caption("配置读取顺序：系统环境变量 / Streamlit Secrets → 本地 .env 文件；也可在下方临时覆盖。")

    api_key = st.text_input(
        "LLM_API_KEY（大模型 API Key）",
        value=os.getenv("LLM_API_KEY", ""),
        type="password",
        placeholder="sk-xxxxxxxx",
    )
    base_url = st.text_input(
        "LLM_BASE_URL（接口地址）",
        value=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )
    model = st.text_input(
        "LLM_MODEL（模型名称）",
        value=os.getenv("LLM_MODEL", "deepseek-chat"),
    )

    # 对当前输入的 key 做一次清洗与校验，仅展示掩码信息，绝不打印完整 key
    cleaned_key, key_meta = sanitize_api_key(api_key)

    st.divider()
    force_mock = st.checkbox(
        "使用本地模拟模式（无需 API Key）",
        value=False,
        help="开启后不联网、不消耗额度，使用内置规则对“少分号 / 数组越界 / 循环边界 / 初始化位置”四类演示案例返回固定诊断。",
    )
    debug_mode = st.checkbox(
        "显示调试信息",
        value=False,
        help="开启后在结果区显示 base_url、model、key 掩码、是否 ASCII、长度及异常 traceback。",
    )

    if force_mock:
        st.info("已开启本地模拟模式：可直接点击“开始诊断”体验完整流程。")
    elif not is_placeholder_key(api_key) and key_meta["starts_with_sk"] and key_meta["is_ascii"]:
        st.success(f"已读取到有效 API Key（{key_meta['masked']}），将调用真实大模型诊断。")
    else:
        hint = "未配置有效 API Key：诊断时将自动使用本地模拟模式。"
        if cleaned_key and not key_meta["starts_with_sk"]:
            hint += "（注意：Key 需以 sk- 开头）"
        if cleaned_key and not key_meta["is_ascii"]:
            hint += "（注意：Key 包含非 ASCII 字符，通常是复制时带入了隐藏字符）"
        st.warning(hint)

    with st.expander("支持哪些大模型？"):
        st.markdown(
            "任何兼容 OpenAI 接口协议的服务都可以，例如：\n\n"
            "- **DeepSeek（推荐）**：`https://api.deepseek.com` 或兼容 `https://api.deepseek.com/v1`，模型 `deepseek-chat`\n"
            "- OpenAI：`https://api.openai.com/v1`，模型 `gpt-4o-mini`\n"
            "- 智谱 GLM：`https://open.bigmodel.cn/api/paas/v4`，模型 `glm-4-flash`\n"
            "- Moonshot：`https://api.moonshot.cn/v1`，模型 `moonshot-v1-8k`"
        )

# ---------------- 主区域：代码输入 + 问题输入 ----------------
code = st.text_area(
    "① 粘贴你的 C 语言代码",
    height=320,
    placeholder="#include <stdio.h>\n\nint main(void) {\n    // 把你的代码粘贴到这里\n    return 0;\n}",
)

question = st.text_area(
    "② 描述你遇到的问题（选填）",
    height=120,
    placeholder="例如：编译报错不知道什么意思 / 运行结果和预期不一样 / 不知道错在哪里……",
)

start_clicked = st.button("开始诊断", type="primary", use_container_width=True)

# ---------------- 诊断逻辑 ----------------
if start_clicked:
    if not code.strip():
        st.warning("请先粘贴需要诊断的 C 语言代码，再点击“开始诊断”。")
        st.stop()

    st.divider()
    st.subheader("诊断结果")

    # 每次结果顶部的来源标识，方便录视频时一眼看出是不是大模型真的在工作
    source_badge = st.empty()
    answer_placeholder = st.empty()
    full_answer = ""

    # 决定本次诊断走哪条路径：手动模拟 > 真实大模型；无有效 Key 时直接模拟
    use_mock = force_mock or is_placeholder_key(api_key)
    if force_mock:
        mock_reason = "已手动开启本地模拟模式"
        banner_level = "info"
    elif is_placeholder_key(api_key):
        mock_reason = "未配置有效的 API Key"
        banner_level = "warning"
    else:
        mock_reason = ""
        banner_level = None

    error_traceback = None

    if not use_mock:
        source_badge.info("⏳ 当前结果来源：正在调用大模型……")
        # 优先尝试真实大模型；任何异常都自动降级到本地模拟模式
        try:
            with st.spinner("CodeMate 正在阅读你的代码，请稍候……"):
                for piece in diagnose(
                    api_key=cleaned_key,        # 使用清洗后的 key
                    base_url=base_url,
                    model=model,
                    code=code,
                    question=question,
                ):
                    full_answer += piece
                    answer_placeholder.markdown(full_answer)
        except Exception as exc:  # 网络错误、Key 错误、限流等：不直接报错，降级
            # 1) 在服务端控制台打印完整 traceback，便于排查
            print("[CodeMate] 大模型调用异常：")
            print(traceback.format_exc())
            # 2) 页面只显示简洁错误类型，不泄露敏感信息
            use_mock = True
            error_traceback = traceback.format_exc()
            mock_reason = f"大模型调用失败，已切换到本地规则模式。错误类型：{type(exc).__name__}"
            banner_level = "warning"
            full_answer = ""
        else:
            source_badge.success("✅ 当前结果来源：大模型诊断")
            st.session_state["last_answer"] = full_answer
            st.session_state["last_mode"] = "llm"
            st.success("诊断完成！先按提示自己动手改一改，改完可以再来诊断一次。")

    if use_mock:
        source_badge.warning("⚠️ 当前结果来源：本地规则兜底")
        banner = (
            f"{mock_reason}，已切换到**本地规则模式**（不联网、不消耗额度）。"
            "以下为内置规则生成的演示诊断，当前可识别：少分号、数组越界、循环边界、初始化位置。"
        )
        if banner_level == "warning":
            st.warning(banner)
        else:
            st.info(banner)

        with st.spinner("CodeMate（本地规则）正在分析你的代码……"):
            for piece in mock_diagnose(code, question):
                full_answer += piece
                answer_placeholder.markdown(full_answer)

        st.session_state["last_answer"] = full_answer
        st.session_state["last_mode"] = "mock"
        st.success("规则诊断完成！配置有效 API Key 后，可获得针对你代码的真实逐行分析。")

    # 调试信息：仅在开启调试模式时显示，绝不含完整 API Key
    if debug_mode:
        with st.expander("🔧 调试信息", expanded=True):
            st.code(
                f"base_url     = {base_url}\n"
                f"model        = {model}\n"
                f"key 掩码     = {key_meta['masked']}\n"
                f"key 长度     = {key_meta['length']}\n"
                f"key 是否ASCII = {key_meta['is_ascii']}\n"
                f"key 以sk-开头 = {key_meta['starts_with_sk']}\n"
                f"本次走模拟模式 = {use_mock}",
                language="text",
            )
            if error_traceback:
                st.caption("异常 traceback：")
                st.code(error_traceback, language="text")

elif st.session_state.get("last_answer"):
    st.divider()
    st.subheader("诊断结果")
    if st.session_state.get("last_mode") == "mock":
        st.warning("⚠️ 当前结果来源：本地规则兜底")
    else:
        st.success("✅ 当前结果来源：大模型诊断")
    st.markdown(st.session_state["last_answer"])
    if st.session_state.get("last_mode") == "mock":
        st.caption("（以上结果来自本地规则模式，未调用大模型）")
