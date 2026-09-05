# -*- coding: utf-8 -*-
"""生成《AI-PI 整定汇报》PPTX（16:9，约 22 页）。

内容定位：被控对象简述；重点讲清 7 种整定算法各自"怎么落地"
（拿什么数据 → 怎么整定/训练 → 产什么参数 → 怎么部署、怎么兜底）。
数据口径：reports/实验报告_AI-PI整定.md / 实验报告_AI自动整定.md / docs/ALGORITHM_GUIDE.md；
图片复用 reports/figures/ 已有 PNG，不重跑实验、不改封存产物。

用法：python reports/make_slides.py
输出：reports/AI-PI整定_汇报slides.pptx
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "AI-PI整定_汇报slides.pptx"

EMU_W, EMU_H = 12192000, 6858000  # 13.333 x 7.5 in (16:9)
IN = Inches

FONT = "微软雅黑"
C_PRIMARY = RGBColor(0x1F, 0x4E, 0x79)
C_PRIMARY_DK = RGBColor(0x17, 0x3A, 0x5C)
C_ACCENT = RGBColor(0xE0, 0x7B, 0x2C)
C_TEXT = RGBColor(0x2B, 0x2B, 0x2B)
C_MUTED = RGBColor(0x6E, 0x6E, 0x6E)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT = RGBColor(0xEE, 0xF3, 0xF9)
C_BAND = RGBColor(0xF4, 0xF7, 0xFB)
C_SKY = RGBColor(0xC9, 0xDA, 0xEC)
C_BORDER = RGBColor(0xC5, 0xD5, 0xE8)


# ---------------------------------------------------------------- 基础助手
def _set_font(run, size=13, bold=False, color=C_TEXT, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)


def L(text, size=12, bold=False, color=C_TEXT, space_after=4, space_before=0,
      indent=0, align=PP_ALIGN.LEFT):
    return {"text": text, "size": size, "bold": bold, "color": color,
            "space_after": space_after, "space_before": space_before,
            "indent": indent, "align": align}


def add_text(slide, x, y, w, h, lines, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        p.space_before = Pt(ln.get("space_before", 0))
        p.space_after = Pt(ln.get("space_after", 4))
        run = p.add_run()
        run.text = ("　" * ln.get("indent", 0)) + ln["text"]
        _set_font(run, size=ln.get("size", 12), bold=ln.get("bold", False),
                  color=ln.get("color", C_TEXT))
    return box


def _rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, line_w=0.75):
    sp = slide.shapes.add_shape(shape, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    return sp


def title_bar(slide, title, subtitle=None, page=None):
    _rect(slide, 0, 0, EMU_W, IN(1.02), C_PRIMARY)
    _rect(slide, 0, IN(1.02), EMU_W, IN(0.045), C_ACCENT)
    add_text(slide, IN(0.45), IN(0.10), IN(12.0), IN(0.62),
             [L(title, size=24, bold=True, color=C_WHITE, space_after=0)],
             anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, IN(0.47), IN(0.70), IN(12.4), IN(0.32),
                 [L(subtitle, size=12, color=C_SKY, space_after=0)])
    if page is not None:
        add_text(slide, IN(12.30), IN(7.10), IN(0.85), IN(0.30),
                 [L(str(page), size=10, color=C_MUTED, space_after=0, align=PP_ALIGN.RIGHT)])


def image_fit(slide, path, x, y, w, h):
    with PILImage.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    px, py = int(x + (w - nw) / 2), int(y + (h - nh) / 2)
    return slide.shapes.add_picture(str(path), px, py, nw, nh)


def fig_block(slide, label, png, caption, x, y, w, h, cap_h=IN(0.30)):
    add_text(slide, x, y, w, IN(0.32),
             [L(label, size=13.5, bold=True, color=C_PRIMARY, space_after=0)])
    image_fit(slide, FIG / png, x, y + IN(0.36), w, h - IN(0.36) - cap_h)
    if caption:
        add_text(slide, x, y + h - cap_h + IN(0.02), w, cap_h,
                 [L(caption, size=10, color=C_MUTED, space_after=0)])


def add_table(slide, data, x, y, w, h, ratios=None, font=10.5, hfont=10.5):
    rows, cols = len(data), len(data[0])
    gf = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    if ratios:
        total = float(sum(ratios))
        for i, r in enumerate(ratios):
            tbl.columns[i].width = Emu(int(w * r / total))
    for ri in range(rows):
        tbl.rows[ri].height = Emu(int(h / rows))
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Emu(45720)
            cell.margin_top = cell.margin_bottom = Emu(13716)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = C_PRIMARY
            else:
                cell.fill.fore_color.rgb = C_WHITE if ri % 2 == 1 else C_BAND
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            _set_font(run, size=hfont if ri == 0 else font,
                      bold=(ri == 0), color=C_WHITE if ri == 0 else C_TEXT)
    return tbl


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------- 算法页模板
def algo_slide(prs, page, num_label, name, tagline, steps, facts, png, fig_caption):
    s = new_slide(prs)
    title_bar(s, f"{num_label}｜{name}：怎么落地", tagline, page=page)
    # 左栏：落地步骤
    add_text(s, IN(0.45), IN(1.20), IN(5.95), IN(0.34),
             [L("落地步骤（数据 → 整定/训练 → 部署）", size=13.5, bold=True,
                color=C_PRIMARY, space_after=0)])
    add_text(s, IN(0.45), IN(1.60), IN(5.95), IN(5.62), steps)
    # 右栏：整定过程图
    fig_block(s, "整定/训练过程（数据来自封存产物）", png, fig_caption,
              IN(6.70), IN(1.20), IN(6.15), IN(3.72))
    # 右栏下方：关键事实框
    box = _rect(s, IN(6.55), IN(5.02), IN(6.42), IN(2.24), C_LIGHT,
                line=C_BORDER, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text(s, IN(6.75), IN(5.14), IN(6.05), IN(2.02), facts)
    return s


# ---------------------------------------------------------------- 幻灯片构建
def build():
    prs = Presentation()
    prs.slide_width = Emu(EMU_W)
    prs.slide_height = Emu(EMU_H)

    # ---- 1 封面 ----
    s = new_slide(prs)
    _rect(s, 0, 0, EMU_W, EMU_H, C_PRIMARY_DK)
    _rect(s, 0, IN(4.72), EMU_W, IN(0.05), C_ACCENT)
    add_text(s, IN(0.9), IN(1.85), IN(11.5), IN(1.15),
             [L("变频精密/机柜空调 AI-PI 自动整定", size=34, bold=True,
                color=C_WHITE, space_after=0)])
    add_text(s, IN(0.9), IN(3.05), IN(11.5), IN(0.7),
             [L("七种整定算法的工程落地方法", size=20, color=C_SKY, space_after=0)])
    add_text(s, IN(0.9), IN(4.95), IN(11.6), IN(1.9), [
        L("项目：hvac-ai-pid-demo ｜ 数据来源：v3 封存验收（80 场景）· 整定耗时基准 · 七算法统一演示", size=12.5, color=C_SKY, space_after=6),
        L("汇报重点：每个算法怎么落地——拿什么数据、怎么整定/训练、产什么参数、怎么部署与兜底", size=12.5, color=C_SKY, space_after=6),
        L("诚信声明：全部结论限于仿真与 SIL 层；真机验收（串口曲线 / WCET / HIL）未开始", size=12.5, color=C_ACCENT, bold=True, space_after=6),
        L("2026-09", size=12, color=C_SKY, space_after=0),
    ])

    # ---- 2 大纲 ----
    s = new_slide(prs)
    title_bar(s, "汇报框架", "重点在第三部分：七个算法逐一讲清落地路径", page=2)
    items = [
        ("①", "被控对象与控制架构（简述）", "3R2C 热对象 + 执行器约束；100 ms PI 内环 + 2 s AI 增益调度层 + 四层安全"),
        ("②", "验收协议：怎么算“调好了”", "48/16/16 划分、80 封存场景、统一目标 J、部署验收门"),
        ("③", "七种算法逐一拆解", "Z-N / IMC-λ / 贝叶斯 BO / 安全 BO / LLM Agent / FNN / RL——每页：落地步骤 + 整定过程图 + 关键事实"),
        ("④", "落地成本与控制性能对比", "PC 秒级 vs 等效对象数千小时；80 场景性能排名"),
        ("⑤", "从 PC 到 ESP32 的部署链路", "策略冻结 → C++ 导出 → 一致性校验 → SIL 演示"),
        ("⑥", "选型建议 · 安全设计 · 局限", "什么约束选什么算法；三重安全机制；如实声明边界"),
    ]
    y = 1.35
    for num, head, desc in items:
        add_text(s, IN(0.75), IN(y), IN(0.6), IN(0.5),
                 [L(num, size=20, bold=True, color=C_ACCENT, space_after=0)])
        add_text(s, IN(1.45), IN(y - 0.02), IN(11.2), IN(0.9), [
            L(head, size=15, bold=True, color=C_PRIMARY, space_after=2),
            L(desc, size=11.5, color=C_MUTED, space_after=0),
        ])
        y += 0.98

    # ---- 3 被控对象（简述） ----
    s = new_slide(prs)
    title_bar(s, "被控对象（简述）", "所有算法共用同一对象与约束——对比才公平；建模细节见实验报告附录 D", page=3)
    add_text(s, IN(0.55), IN(1.30), IN(12.2), IN(5.9), [
        L("控制任务（一句话）", size=14, bold=True, color=C_PRIMARY, space_after=4),
        L("单区机柜空调送风温度闭环：被控量 T_zone（区域空气温度），控制量 u∈[0,1]（压缩机归一化容量指令）；e = T_zone − T_sp > 0 表示偏热、需增制冷。", size=12.5, space_after=10),
        L("对象模型", size=14, bold=True, color=C_PRIMARY, space_after=4),
        L("3R2C 集中参数热模型（两热容 + 三热阻）＋ 纯延迟 L（名义 5 min，场景随机 2–12 min）＋ 一阶执行器滞后 τu≈4 min；名义最大制冷量 7000 W（场景随机 4500–10500 W）；仿真步长 1 min。", size=12.5, space_after=10),
        L("六条执行器/传感器约束（全部算法完全相同）", size=14, bold=True, color=C_PRIMARY, space_after=4),
        L("最小制冷容量 25%（低于即停机）｜指令量化 1%｜变频斜率 5 个百分点/min（刻意保守）", size=12.5, space_after=2),
        L("最短运行 5 min / 最短停机 3 min（防短循环）｜纯延迟 + 一阶惯性｜测量噪声 σ=0.04 °C + 滤波 τ=2 min", size=12.5, space_after=10),
        L("为什么这些约束重要", size=14, bold=True, color=C_PRIMARY, space_after=4),
        L("它们使真机试凑既昂贵又不安全（斜率/驻留限制下试验周期长、过冲风险高）——这是后文“AI 层必须限幅 + 回退”与“试错全部搬进离线仿真”的直接原因。", size=12.5, space_after=10),
        L("模型可信度（已做，不展开）", size=14, bold=True, color=C_PRIMARY, space_after=4),
        L("数值层（DOP853 交叉验证）· 降阶层（FOPDT vs 全模型 ≈12%）· 物理层（OpenModelica vs Python，RMSE 7.16×10⁻⁶ °C）三层交叉验证。", size=12.5, space_after=0),
    ])

    # ---- 4 分层控制架构 ----
    s = new_slide(prs)
    title_bar(s, "分层控制架构：所有算法共同的落地底座", "快内环 + 慢调度——AI 只调增益，不碰压缩机", page=4)
    add_text(s, IN(0.50), IN(1.25), IN(6.05), IN(5.95), [
        L("内环（100 ms 周期）· PI 基础控制器", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("e = T_z − T_sp；I ← I + e·Δt；u = clamp(Kp·e + Ki·I, 0, 1)", size=12, space_after=3),
        L("条件积分抗饱和：输出已饱和且误差继续同向时冻结积分——防长时间 100% 制冷后积分累积、严重过冷", size=12, space_after=9),
        L("外环（2 s 周期）· AI 增益调度层", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("FNN/RL 只调整 Kp/Ki，不直接控制压缩机；AI 推理微秒~毫秒级时延不影响内环执行", size=12, space_after=9),
        L("四层安全保护（在线算法共用）", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("① AI 每 2 s 才更新一次，把 PI 留给 100 ms 执行", size=12, space_after=3),
        L("② 增益界：Kp ∈ [0.002, 1.5]，Ki ∈ [1e-5, 0.08]", size=12, space_after=3),
        L("③ 单次增益变化 ≤ 当前值的 25%", size=12, space_after=3),
        L("④ 输入/模型异常 → 立即恢复 IMC 的 fallback_gains", size=12, space_after=9),
        L("为什么要这样分层", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("执行器约束决定控制量必须缓慢、受限地变化；任何 AI 失效情形下，系统退化为已验证的 IMC 闭环——这是后续所有算法能够安全落地的前提。", size=12, space_after=0),
    ])
    fig_block(s, "系统结构框图", "fig1_系统结构框图.png",
              "图1｜信号流：Tsp 与 Tz 求误差 → PI 内环（AI 调增益）→ 执行器约束环节 → 3R2C 对象 → 传感反馈",
              IN(6.75), IN(1.25), IN(6.15), IN(5.95))

    # ---- 5 验收协议 ----
    s = new_slide(prs)
    title_bar(s, "验收协议：怎么算“调好了”", "防背题、防挑指标、防“上线才翻车”", page=5)
    add_text(s, IN(0.50), IN(1.25), IN(6.05), IN(5.95), [
        L("数据集", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("48 训练 / 16 验证 / 16 测试；制冷量 4500–10500 W、延迟 2–12 min 随机化", size=12, space_after=3),
        L("7 类工况族：热/冷启动、设定值升/降阶跃、开门热脉冲、持续负荷、低温降阶跃", size=12, space_after=3),
        L("逐场景 SHA-256 写入 dataset_manifest.csv，执行数据泄漏/重复拒绝检查", size=12, space_after=9),
        L("封存验收（防“背题”）", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("5 个验收种子 × 16 场景 = 80 个全新场景；算法冻结后才“开封”，只考一次取平均，压掉运气成分", size=12, space_after=9),
        L("统一目标 J（越小越好）", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("J = 2·IAE + 1.5·ITAE + 10·舒适区违规 + 3·最大过冷 + 0.35·调节时间 + 0.08·控制变化量 + 0.7·指令方差 + 能耗项", size=12, space_after=3),
        L("不稳定或越界轨迹罚 1e6；全部算法同一指标集（hvac_pid/metrics.py）", size=12, space_after=9),
        L("部署验收门", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("密封测试均值比（算法 / IMC）的 95% 置信上界 ≤ 阈值才允许部署——FNN/RL 均按此门判定 PASSED", size=12, space_after=0),
    ])
    fig_block(s, "实验流程", "fig2_实验流程框图.png",
              "图2｜场景划分 → 各算法整定/训练 → 统一闭环仿真 → 指标计算 → 部署验收门 → 封存对比",
              IN(6.75), IN(1.25), IN(6.15), IN(5.95))

    # ---- 6 七算法总览 ----
    s = new_slide(prs)
    title_bar(s, "七种整定方法总览", "按“参数从哪来”分三类：人工公式 / 离线自动搜索 / 在线自整定", page=6)
    data = [
        ["算法", "整定方式", "在线?", "输入", "输出", "PC 整定/训练耗时", "80 场景 holdout"],
        ["Z-N 反应曲线法", "人工经验公式", "否", "FOPDT 的 K/L/τ", "固定 Kp/Ti", "0.163 s / 等效 168 h", "58.19"],
        ["IMC-λ", "离线自动整定（λ 扫描）", "否", "FOPDT + λ", "固定 Kp/Ti", "3.03 s / 等效 1008 h", "39.53"],
        ["贝叶斯 BO", "离线自动整定（GP+EI）", "否", "工况 + 目标函数 J", "固定 Kp/Ki", "1.42 s / 等效 528 h", "39.72"],
        ["安全 BO（RaGoOSE 式）", "离线自动整定（风险感知）", "否", "同上 + 风险项", "固定 Kp/Ki", "1.41 s（7 次评估）", "33.58*"],
        ["LLM Agent", "离线自动整定（工具调用）", "否", "工况信息 + 指标工具", "Kp/Ki 修正决策", "真实调用 30.4 s", "见注†"],
        ["FNN 自整定", "在线自整定（查规则表）", "是（每 2 s）", "e, ė（模糊化）", "Kp/Ki 增益", "训练 7.65 s / 等效 4181 h", "39.26 PASSED"],
        ["RL 自整定", "在线自整定（查 Q 表）", "是（每 2 s）", "热状态 + 容量模式", "相对 IMC 的增益目标", "训练 16.22 s / 等效 4018 h", "37.61 最优"],
    ]
    add_table(s, data, IN(0.45), IN(1.30), IN(12.45), IN(4.60),
              ratios=[1.55, 2.05, 0.95, 1.85, 1.75, 2.15, 1.55], font=10.5)
    add_text(s, IN(0.45), IN(6.10), IN(12.4), IN(1.2), [
        L("* 安全 BO 的 33.58 是高级整定基准（独立留出集）上的风险目标，与 80 场景口径不同，不可直接混比；该基准内排序：安全 BO 33.58 < 启发式 34.18 < IMC 36.29 < 普通 BO 37.96。", size=10.5, color=C_MUTED, space_after=3),
        L("† LLM Agent 为真实 API 调用（阿里云百炼 qwen-max）：七算法统一演示综合目标 45.04（第三）、开门恢复 38 min（次快）、过冷 0.81 °C 最小；候选经安全门接受后部署。30.4 s 为监督式基准口径。", size=10.5, color=C_MUTED, space_after=3),
        L("耗时来自 archive/outputs_tuning_benchmark（基准问题：31.5→24 °C、7600 W、延迟 6 min、8 个离线工况）；holdout 来自 v3 封存验收。", size=10.5, color=C_MUTED, space_after=0),
    ])

    # ---- 7 Z-N ----
    algo_slide(prs, 7, "算法①", "Z-N 反应曲线法",
        "人工经验公式 · 非自整定 · 激进经典基线",
        [
            L("① 做阶跃试验：稳定工况给压缩机一个小阶跃，记录温度响应直到稳态（真机需等效 168 对象小时；PC 仿真约 200× 加速）", size=12, space_after=5),
            L("② 辨识 FOPDT：有界最小二乘同时拟合温降幅度 A、时间常数 τ、延迟 L，增益 K = A/Δu；不要用“末点当 t63”的老办法（旧版 12 h 截尾问题，已修正）", size=12, space_after=5),
            L("③ 代入 Z-N 公式：Kp = 0.9τ/(K·L)，Ti = 3.33L，Ki = Kp/Ti", size=12, space_after=5),
            L("④ 工程限幅（np.clip，不属 Z-N 原公式）→ 冻结写入控制器，投运后不再变化", size=12, space_after=8),
            L("落地要点：公式本身免费，贵的是安全辨识采数——真机上 168 h 的阶跃试验才是主要成本。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("输入/输出：FOPDT (K, τ, L) → 固定 Kp/Ti（v3 原值 2.97/0.178 被限幅为 1.5/0.08）", size=11, space_after=3),
            L("耗时：公式 0.163 s；辨识采数等效 168 对象小时", size=11, space_after=3),
            L("表现：80 场景 holdout 58.19（全场最差）；宽工况偏振荡，过冷 1.46–2.21 °C、指令方差最大", size=11, space_after=3),
            L("定位：激进基线；无自适应；ESP32 演示中该候选被验收门拒绝、回退 IMC", size=11, space_after=0),
        ],
        "fig03_zn_fopdt辨识.png",
        "图3｜168 h 阶跃响应（灰实线）与 FOPDT 拟合（橙虚线）：围护热容长尾是单时间常数近似的固有误差")

    # ---- 8 IMC ----
    algo_slide(prs, 8, "算法②", "IMC-λ（内模控制）",
        "离线自动整定 · 稳健基线 · 全项目安全回退基准",
        [
            L("① 复用同一次 FOPDT 辨识结果（与 Z-N 共享数据基础，不重做试验）", size=12, space_after=5),
            L("② 扫描 λ：只在训练工况上评估 21 个候选 λ（0.88 s），选训练集平均目标最小者 λ* = 26.9 min，随后冻结；验证/测试集不再回调", size=12, space_after=5),
            L("③ 代入 IMC 公式：Kp = τ/[K(λ+L)]，Ti = min[τ, 4(λ+L)]，Ki = Kp/Ti", size=12, space_after=5),
            L("④ 另保留保守公式值 λ = max(τ/3, 3L, 12 min) 作为故障回退增益——它是全项目所有 AI 方法的兜底基准", size=12, space_after=8),
            L("落地要点：这样整定能公平回答“IMC 是结构不合适，还是 λ 没选对”——答案是后者。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("输入/输出：FOPDT + λ → 固定 Kp/Ti（Kp/Ti 仍受 IMC 公式约束）", size=11, space_after=3),
            L("耗时：PC 3.03 s（含 λ 扫描）/ 等效 1008 对象小时", size=11, space_after=3),
            L("表现：holdout 39.53；全部工况稳定；单组固定增益无法随工况自适应", size=11, space_after=3),
            L("关键发现：保守回退 λ=303.7 min 的目标值比 λ*=26.9 min 高近 6 倍", size=11, space_after=0),
        ],
        "fig04_imc_lambda扫描.png",
        "图4｜λ 扫描曲线：21 候选中 λ*=26.9 min（蓝圈）最优；保守回退值 303.7 min（橙叉）差近 6 倍")

    # ---- 9 贝叶斯 BO ----
    algo_slide(prs, 9, "算法③", "贝叶斯 BO（固定增益）",
        "离线自动整定 · GP + 期望改进 · 运行时零学习负担",
        [
            L("① 定义搜索空间：log(Kp)、log(Ki) 两维——对数域处理 Ki 跨多个数量级的问题", size=12, space_after=5),
            L("② 定义目标 J：多指标加权综合（跟踪/舒适区/过冷/平稳性/能耗），不稳定或越界轨迹罚 1e6；每个候选都跑完整闭环仿真", size=12, space_after=5),
            L("③ 迭代搜索：Matérn-5/2 高斯过程近似“增益 → J”，EI（期望改进）选下一个试验点；IMC/Z-N/随机点初始化", size=12, space_after=5),
            L("④ 收敛：以多个训练工况的平均 J 为目标 → 输出一套全局固定 Kp/Ki → 写入控制器，运行时零学习负担", size=12, space_after=8),
            L("落地要点：全部搜索离线完成，部署后不再探索——最适合“只允许固定参数”的低成本 MCU。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("输入/输出：工况 + 目标函数 → 全局固定 Kp/Ki（v3 部署 Kp=0.5355、Ki=0.0063）", size=11, space_after=3),
            L("耗时：PC 1.42 s（10 次评估）/ 等效 528 对象小时", size=11, space_after=3),
            L("表现：holdout 39.72 ≈ IMC 39.53——单组固定增益已接近其上限", size=11, space_after=3),
            L("安全：纯离线搜索，部署后不探索，无运行时风险", size=11, space_after=0),
        ],
        "fig05_bo搜索轨迹.png",
        "图5｜GP+EI 搜索轨迹：7 个初始化点（灰）后，EI 迭代（橙圈）把目标从 ~242 压到 33 附近并收敛")

    # ---- 10 安全 BO ----
    algo_slide(prs, 10, "算法④", "风险感知安全 BO（RaGoOSE 式）",
        "离线自动整定 · 优化“均值 + 0.75σ” · 跨工况最稳",
        [
            L("① 在 BO 框架上加重复噪声仿真：每个候选在每个场景重复 2 次，得到跨工况的均值与方差", size=12, space_after=5),
            L("② 风险目标：J_risk = 均值 J̄ + 0.75 × 标准差 σ_J——主动避开“平均好但波动大”的候选", size=12, space_after=5),
            L("③ 安全机制内建于搜索：β=2 安全 GP 抑制危险试验点；安全条件：稳定、斜率无违规、无最低运行频率违规、最大冷偏差 ≤4 °C、调参风险分 ≤ IMC 基线 1.25 倍", size=12, space_after=5),
            L("④ 输出固定 Kp/Ki（0.3838 / 0.004441）→ 部署方式同普通 BO；安全裕量逐候选记录（预测/实测）", size=12, space_after=8),
            L("落地要点：对“最差工况也不能失控”的精密空调，风险感知口径比均值口径更合适。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("耗时：PC 1.41 s（7 次评估）；7 训练 + 4 留出场景", size=11, space_after=3),
            L("表现：风险目标 33.58 为该基准最优（IMC 36.29、普通 BO 37.96）", size=11, space_after=3),
            L("更稳：objective_std 20.93 vs 普通 BO 29.18；七算法演示综合目标 43.23 亦最优", size=11, space_after=3),
            L("注意：33.58 是高级整定基准口径，与 80 场景 holdout 不可混比", size=11, space_after=0),
        ],
        "fig06_安全bo风险搜索.png",
        "图6｜7 次候选评估：误差棒为跨工况标准差，风险目标（均值+0.75σ）引导搜索避开高方差候选")

    # ---- 11 LLM Agent ----
    algo_slide(prs, 11, "算法⑤", "LLM Agent（工具调用整定）",
        "离线自动整定 · 只提建议 · 安全门与仿真器拥有最终决定权",
        [
            L("① 定位：低频离线诊断与候选生成——LLM 不进控制回路，只提 Kp/Ki 建议", size=12, space_after=5),
            L("② 工具调用闭环：Agent 自主选择工具——查看历史 → 提交候选 Kp/Ki → 运行候选仿真 → 读取指标", size=12, space_after=5),
            L("③ 确定性安全门：每条建议先限幅，再做“风险调整改进”判定；改进不足即拒绝（本次真实会话：首候选被拒，第二个被接受）", size=12, space_after=5),
            L("④ 接受的最优安全候选冻结部署（真实调用：0.4057 / 0.002769）", size=12, space_after=8),
            L("落地要点：真实调用（百炼 qwen-max）已完成单批次验证；启发式 dry-run 只验证安全流水线，不能作为 LLM 效果证据。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("真实调用（qwen-max）：6 步、2 次候选试验，第二个候选风险目标 32.01→30.68（改善 4.1%）被接受", size=11, space_after=3),
            L("七算法演示：综合目标 45.04（第三）；开门恢复 38 min（次快）；过冷 0.81 °C 最小", size=11, space_after=3),
            L("实测耗时：监督式 3 轮 30.4 s（比经典方法高 1–2 个数量级）；另一次独立会话候选全被拒、回退 IMC——输出非确定但零失控", size=11, space_after=3),
            L("定位：低频离线诊断工具，不追求单点指标最优", size=11, space_after=0),
        ],
        "fig07_llm工具调用轨迹.png",
        "图7｜真实调用轨迹：查看历史（基线 32.0）→ 候选①被拒 → 候选②（小幅减小 Kp/Ki）30.68 被安全门接受 → 主动终止")

    # ---- 12 FNN ----
    algo_slide(prs, 12, "算法⑥", "FNN 在线自整定（5×5 TSK 规则面）",
        "在线自整定 · 每 2 s 查规则表 · 面向 MCU 的零阶模糊神经",
        [
            L("① 造标签：对 48 个训练场景各做一轮完整 BO → 每工况最优 Kp/Ki（4464 样本；等效 4181 对象小时）", size=12, space_after=5),
            L("② 定输入：e = Tz−Tsp（中心点 [−3,−0.75,0,1.5,5] °C）× ė = Δe/Δt（中心点 [−0.30,−0.05,0,0.05,0.30] °C/min）——用变化率而非增量，PC 5 min 训练节拍与 MCU 秒级节拍物理含义一致", size=12, space_after=5),
            L("③ 学规则面：按 (e, ė) 模糊格学 5×5 = 25 条零阶 TSK 规则后件 → fnn_rule_table.npy", size=12, space_after=5),
            L("④ 过验收门：规则覆盖率 ≥80% 且平均目标不劣于 IMC 才部署；不过则自动写 IMC 回退表（候选另存 *_candidate.npy 供审计）", size=12, space_after=5),
            L("⑤ 在线：每 2 s 找相邻 2×2 = 4 条规则加权平均 → 公共限幅 + 变化率限制（推理 78 µs/次）", size=12, space_after=8),
            L("落地要点：把“什么状态下哪组旋钮更好”记进 25 格规则表，运行时只查表——不是大规模深度网络。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("训练：PC 7.65 s；log-RMSE 1.41→1.059（多工况标签冲突致回升）", size=11, space_after=3),
            L("表现：holdout 39.26；相对 IMC 比值 0.9979（95% 上界 1.0067）PASSED；规则覆盖 100%", size=11, space_after=3),
            L("注意：ESP32 演示中该批次 FNN 候选被验收门拦截，影子模式 + IMC 回退（与 v3 封存验收不同批次）", size=11, space_after=0),
        ],
        "fig08_fnn训练收敛.png",
        "图8｜(a) log-RMSE 随训练工况数先降后升（标签冲突）；(b) 规则覆盖率 100%，远高于 80% 验收门")

    # ---- 13 RL ----
    algo_slide(prs, 13, "算法⑦", "RL 在线自整定（表格式 Q 学习）",
        "在线自整定 · 离线训练 + 在线查表 · holdout 全场最优",
        [
            L("① 定义 MDP：状态 = bin(e,5) × bin(ė,5) × bin(上次实际容量,3) 共 75 个；动作 = 相对 IMC 的 Kp/Ki 绝对目标比例共 9 个；Q 表 5×5×3×9 = 675 项", size=12, space_after=5),
            L("② 定奖励：r = −|e| − 1.5·舒适带超限 − 0.08|Δu| − 0.02u − 0.08|增益变化| − 0.015|目标偏离 1|", size=12, space_after=5),
            L("③ 离线训练：同一 3R2C 虚拟环境 750 回合（每回合 ≤240 min、5 min 决策、36,000 转移）；ε 0.25→0.03；验证集早停选表（31.95 < IMC 基线 32.91）", size=12, space_after=5),
            L("④ 过部署门：平均目标差于 IMC 2% 以上或 25 热状态覆盖 <80% → 整表拒绝、回退 IMC；不安全动作在线屏蔽", size=12, space_after=5),
            L("⑤ 在线：分箱 → 屏蔽不安全动作 → argmax(Q) → ±10% 安全变化层；真机只查冻结表、零随机探索（88 µs/次）", size=12, space_after=8),
            L("落地要点：动作是“相对 IMC 的绝对目标比例”，修复旧版增益递归相乘的非马尔可夫问题——这是能安全上板的关键设计。", size=12, bold=True, color=C_PRIMARY, space_after=0),
        ],
        [
            L("关键事实", size=12.5, bold=True, color=C_ACCENT, space_after=4),
            L("训练：PC 16.22 s / 等效 4018 对象小时；产物 rl_q_table.npy", size=11, space_after=3),
            L("表现：holdout 37.61 全场最优；比值 0.9595（上界 0.9743）PASSED；三标准工况全面领先", size=11, space_after=3),
            L("残留缺陷 C4：积分、限制器与负荷仍是安全上下文记录（部分闭环）", size=11, space_after=0),
        ],
        "fig09_rl训练收敛.png",
        "图9｜(a) 回合奖励波动但 ε 按计划 0.25→0.03 衰减；(b) 检查点验证目标下降，早停选中 31.95")

    # ---- 14 成本对比 ----
    s = new_slide(prs)
    title_bar(s, "落地成本：PC 秒级 vs 真机数千对象小时", "等效对象时间 = 若在真实压缩机上串行完成同样整定试验，设备需实际运行的时长", page=14)
    fig_block(s, "整定耗时双口径对比", "fig10_整定耗时对比.png",
              "图10｜(a) PC 墙钟时间（秒级）；(b) 等效虚拟对象时间（百到数千小时）",
              IN(0.45), IN(1.25), IN(7.35), IN(5.95))
    add_text(s, IN(8.05), IN(1.35), IN(4.85), IN(5.8), [
        L("PC 口径（纯 CPU，无 GPU / 无深度学习框架）", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("Z-N 0.163 s｜IMC 3.03 s｜BO 1.42 s｜安全 BO 1.41 s｜FNN 7.65 s｜RL 16.22 s", size=11.5, space_after=8),
        L("等效对象口径", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("Z-N 168 h｜IMC 1008 h｜BO 528 h｜FNN 4181 h｜RL 4018 h", size=11.5, space_after=8),
        L("成本三构成", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("FOPDT 安全辨识采数 168 h/工况；每组候选评估约 5 h；FNN/RL 标签需每工况完整搜索 → 4000+ h", size=11.5, space_after=8),
        L("工程结论", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("这些试验在真机上受执行器安全约束根本不可行；PC 加速仿真（~200×）把全部试错压到秒级", size=11.5, space_after=3),
        L("真机只部署结果、不做试验——这是 FNN/RL 路线的核心工程价值", size=11.5, bold=True, space_after=0),
    ])

    # ---- 15 性能对比 ----
    s = new_slide(prs)
    title_bar(s, "控制性能：80 场景封存验收", "算法冻结后才开封的全新场景，只考一次取平均", page=15)
    fig_block(s, "80 场景 holdout 性能对比", "fig11_控制性能对比.png",
              "图11｜RL 37.61 < FNN 39.26 ≈ IMC 39.53 ≈ BO 39.72 < Z-N 58.19",
              IN(0.45), IN(1.25), IN(7.35), IN(5.95))
    add_text(s, IN(8.05), IN(1.35), IN(4.85), IN(5.8), [
        L("80 场景封存验收（v3，最权威口径）", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("RL 37.61 < FNN 39.26 ≈ IMC 39.53 ≈ BO 39.72 < Z-N 58.19", size=11.5, space_after=8),
        L("部署验收门判定", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("RL：比值 0.9595，95% 上界 0.9743 → PASSED", size=11.5, space_after=3),
        L("FNN：比值 0.9979，95% 上界 1.0067 → PASSED", size=11.5, space_after=8),
        L("留出工况平均（五算法）", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("ITAE：RL 5.850 最优；最大过冷：RL 1.666 °C 最优；稳定率全部 100%", size=11.5, space_after=8),
        L("读数提醒", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("不能只按单一目标排名；精密空调应同时看稳定率、ITAE、扰动恢复、指令方差与启停次数", size=11.5, space_after=3),
        L("高级整定基准（不同口径）：安全 BO 33.58 同时取得最优均值与最小方差", size=11.5, space_after=0),
    ])

    # ---- 16 三标准工况 ----
    s = new_slide(prs)
    title_bar(s, "三标准工况：算法差异都在进入舒适带之后", "前 1 h 各算法均满容量降温；拉开差距的是带内稳定与扰动恢复", page=16)
    data = [
        ["算法", "工况一 ITAE", "工况一 调节(h)", "工况一 过冷(°C)", "工况二 ITAE", "工况二 过冷(°C)", "工况三 ITAE", "工况三 过冷(°C)"],
        ["Z-N", "7.054", "4.000", "1.456", "11.20", "1.765", "18.75", "2.209"],
        ["IMC-λ", "3.887", "2.517", "0.684", "6.846", "1.177", "15.63", "1.202"],
        ["贝叶斯 BO", "3.666", "2.500", "0.806", "7.113", "1.297", "15.29", "1.547"],
        ["FNN", "3.626", "2.517", "0.759", "6.936", "1.214", "15.29", "1.335"],
        ["RL", "3.484", "1.733", "0.229", "6.603", "0.849", "15.21", "1.251"],
    ]
    add_table(s, data, IN(0.45), IN(1.35), IN(12.45), IN(3.30),
              ratios=[1.35, 1.35, 1.45, 1.55, 1.35, 1.55, 1.35, 1.55], font=11)
    add_text(s, IN(0.45), IN(4.95), IN(12.4), IN(2.2), [
        L("工况一＝初次快速降温 31.5→24 °C；工况二＝1 h 时设定值 25→23 °C；工况三＝外温 37±4.5 °C + 3 h 开门 12 min（+2600 W 热脉冲）。", size=11.5, color=C_MUTED, space_after=6),
        L("RL 三工况全面领先或并列领先：工况一调节时间 1.733 h（其余 2.5–4 h）、过冷 0.229 °C（次优 0.68–1.46）；工况二过冷 0.849 °C 最小；工况三指令方差 0.132 最平稳。", size=12, space_after=6),
        L("Z-N 在全部工况振幅最大且不衰减（指令方差 0.18–0.20）——固定激进增益 + 大延迟对象的典型表现。", size=12, space_after=0),
    ])

    # ---- 17 七算法统一演示 ----
    s = new_slide(prs)
    title_bar(s, "七算法统一演示（ESP32 温度闭环）", "30→24 °C、min 189 开门扰动——安全门 + 回退机制生效的实例", page=17)
    fig_block(s, "七算法闭环曲线", "fig27_七算法统一演示.png",
              "图27｜min 189 开门扰动（橙区）后 RL 与安全 BO 恢复最快；Z-N/FNN 曲线与 IMC 重合（实际部署 IMC 回退）",
              IN(0.45), IN(1.25), IN(7.55), IN(5.95))
    add_text(s, IN(8.25), IN(1.35), IN(4.65), IN(5.8), [
        L("关键数字（seven_algorithm_summary.csv）", size=12.5, bold=True, color=C_PRIMARY, space_after=4),
        L("开门恢复：RL 24 min 最快、安全 BO 25 min 次之、LLM Agent 38 min 第三（真实调用）；Z-N/IMC/FNN 42 min", size=11.5, space_after=6),
        L("综合目标：安全 BO 43.23 最优；RL 44.41 次之；LLM Agent 45.04 第三（过冷 0.81 °C 全算法最小）", size=11.5, space_after=6),
        L("全部七算法首次进带均 34 min——满容量降温阶段无差异，差异都在进入舒适带之后", size=11.5, space_after=8),
        L("验收门拦截实例", size=12.5, bold=True, color=C_ACCENT, space_after=4),
        L("该批次 Z-N 候选被可见验收门拒绝、FNN 候选进入影子模式，二者实际部署 IMC 回退——曲线完全重合即为证据", size=11.5, space_after=6),
        L("提示：与 v3 封存验收为不同批次运行，判定不一致问题见局限性页", size=11, color=C_MUTED, space_after=0),
    ])

    # ---- 18 嵌入式落地链路 ----
    s = new_slide(prs)
    title_bar(s, "从 PC 到 ESP32：策略怎么上板", "冻结 → 导出 → 校验 → SIL，真机验收如实标注“未开始”", page=18)
    chain = [
        ("离线训练产物", C_PRIMARY),
        ("策略冻结导出\nC++ 头文件", C_PRIMARY),
        ("Py/C++ 校验\n75 条向量", C_PRIMARY),
        ("ESP32 SIL 演示\n七算法闭环", C_PRIMARY),
        ("真机验收\n（未开始）", RGBColor(0x8C, 0x8C, 0x8C)),
    ]
    x, w_ch, gap = IN(0.45), IN(2.42), IN(0.10)
    for text, fill in chain:
        sp = _rect(s, x, IN(1.45), w_ch, IN(1.05), fill, shape=MSO_SHAPE.CHEVRON)
        tf = sp.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Emu(27432)
        tf.margin_top = tf.margin_bottom = Emu(13716)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        parts = text.split("\n")
        first = True
        for part in parts:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = part
            _set_font(run, size=11 if len(parts) == 1 or part == parts[0] else 10,
                      bold=True, color=C_WHITE)
        x += w_ch + gap
    add_text(s, IN(0.45), IN(2.90), IN(12.4), IN(4.3), [
        L("① 训练产物冻结：规则表 / Q 表（.npy）或固定增益 → 导出 C++ 头文件 generated_policy.hpp，附 CRC v3 校验——板上策略与训练时逐位一致", size=12.5, space_after=8),
        L("② 一致性校验：75 条 Py/C++ 奇偶校验向量，最大误差 2.98×10⁻⁸（policy_parity_vectors.csv）——Python 策略与板级代码逐点一致", size=12.5, space_after=8),
        L("③ 板级资源：ESP32 静态 RAM 22,036 B（6.7%）、Flash 296,169 B（22.6%），PlatformIO 编译通过；PC-SIL 回退 0/190", size=12.5, space_after=8),
        L("④ 在线节拍：PI 100 ms / AI 2 s；PC 推理 78–88 µs/次仅为本机测量——不是板级 WCET", size=12.5, space_after=8),
        L("⑤ 待办（如实声明）：真机 10 min 串口验收、DWT/GPIO 脉冲实测 WCET、HIL 均未开始", size=12.5, bold=True, color=C_ACCENT, space_after=0),
    ])

    # ---- 19 选型决策表 ----
    s = new_slide(prs)
    title_bar(s, "怎么选算法：落地决策表", "按“允许什么运行形态 + 追求什么”选路线", page=19)
    data = [
        ["落地约束 / 目标", "推荐", "理由与注意"],
        ["只允许固定参数、MCU 零负担", "IMC-λ", "一组增益写死，实现最简单、全工况稳定；同时兼任回退基准"],
        ["固定参数但要搜索最优", "贝叶斯 BO", "离线 1.42 s 出全局增益；需闭环仿真与目标函数基建"],
        ["最差工况也不能失控", "风险感知安全 BO", "风险目标 = 均值 + 0.75σ；均值与方差同时最优"],
        ["允许在线计算、追求最优性能", "RL Q 表", "holdout 最优；在线仅查表 88 µs、零探索，带部署门"],
        ["在线可算、要可解释/可审计", "FNN 规则面", "25 条规则可人工检查；4 邻格加权查表，78 µs"],
        ["快速起基线 / 无数据基建", "Z-N", "一次阶跃 + 公式即可；宽工况易振荡，仅作基线"],
        ["有工具调用基建、低频诊断", "LLM Agent", "真实调用已完成单批次验证（qwen-max）：候选经安全门接受、过冷最小；耗时比经典方法高 1–2 个数量级"],
    ]
    add_table(s, data, IN(0.45), IN(1.35), IN(12.45), IN(4.85),
              ratios=[3.1, 2.0, 7.35], font=11.5)
    add_text(s, IN(0.45), IN(6.45), IN(12.4), IN(0.8), [
        L("通用兜底：无论选哪条路线，都保留 IMC 回退增益 + 增益限幅 + 部署验收门——AI 失效时系统退化为已验证闭环。", size=12, bold=True, color=C_PRIMARY, space_after=0),
    ])

    # ---- 20 安全设计 ----
    s = new_slide(prs)
    title_bar(s, "安全设计：三重机制兜底", "限幅 → 回退 → 验收门，缺一不可", page=20)
    blocks = [
        ("① 限幅：把 AI 关进笼子", [
            "增益界 Kp∈[0.002,1.5]、Ki∈[1e-5,0.08]",
            "单次增益变化 ≤ 当前值 25%",
            "执行器斜率/量化/驻留在物理层再挡一道",
            "AI 2 s 节拍与 PI 100 ms 解耦",
        ]),
        ("② 回退：坏了怎么办", [
            "输入/模型异常 → 立即恢复 IMC 增益",
            "验收不过 → 候选自动写 IMC 回退表",
            "真机上永远有可用闭环，不会“无控制器”",
            "RL 回退率 0.19%，FNN 0%（v3 批次）",
        ]),
        ("③ 验收门：上线前把关", [
            "80 个封存场景冻结后才开封、只考一次",
            "均值比（算法/IMC）95% 上界 ≤ 阈值才部署",
            "FNN/RL 均按此门判定 PASSED",
            "不过门 = 不上线，无例外",
        ]),
    ]
    x = IN(0.45)
    for title, lines in blocks:
        _rect(s, x, IN(1.40), IN(4.00), IN(4.60), C_LIGHT, line=C_BORDER,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text(s, x + IN(0.22), IN(1.60), IN(3.56), IN(0.5),
                 [L(title, size=14, bold=True, color=C_PRIMARY, space_after=0)])
        add_text(s, x + IN(0.22), IN(2.15), IN(3.56), IN(3.7),
                 [L("· " + t, size=11.5, space_after=7) for t in lines])
        x += IN(4.22)
    add_text(s, IN(0.45), IN(6.30), IN(12.4), IN(0.9), [
        L("实证：七算法演示中 Z-N/FNN 候选被验收门拦截、回退 IMC；80 场景验收全部算法稳定率 100%、零失控。", size=12.5, bold=True, color=C_ACCENT, space_after=0),
    ])

    # ---- 21 局限性 ----
    s = new_slide(prs)
    title_bar(s, "局限性与后续工作（如实声明）", "全部结论限于仿真与 SIL 层", page=21)
    left = [
        L("证据边界", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("真机验收未开始：10 min 串口曲线、实测 WCET、HIL——全部耗时为 PC/仿真口径", size=12, space_after=5),
        L("LLM Agent 真实调用仅单批次（2 次会话）：输出非确定性未做统计表征；耗时/费用为单次实测", size=12, space_after=5),
        L("批次差异：耗时基准中 RL 验收列 False、ESP32 演示中 FNN 被拒、v3 封存中均 PASSED——正式发布前需重新封存复跑", size=12, space_after=5),
        L("等效对象时间为估算口径：衡量“搬到真机需多久”的工程换算，非实测", size=12, space_after=10),
        L("方法学残留", size=13.5, bold=True, color=C_PRIMARY, space_after=4),
        L("RL 马尔可夫性残留缺陷 C4：积分、限制器与负荷仍是安全上下文记录（部分闭环）", size=12, space_after=5),
        L("FNN 标签冲突：多场景同状态异标签，log-RMSE 1.059–1.85 限制规则面拟合精度", size=12, space_after=5),
        L("能耗与指令方差为代理量：∫Qc·dt 未含 COP 与风机/泵功耗；湿度与结霜未建模（仅显热）", size=12, space_after=10),
        L("后续工作：SIL → HIL → 只读影子模式 → 有限增益试运行 → 单机试点 → 多季节回归；LLM 多批次真实调用统计；C4 补齐；封存产物批次一致性复跑", size=12, bold=True, color=C_PRIMARY, space_after=0),
    ]
    add_text(s, IN(0.55), IN(1.30), IN(12.2), IN(5.9), left)

    # ---- 22 结论 ----
    s = new_slide(prs)
    title_bar(s, "结论：三问三答", "对应实验目的的三个工程问题", page=22)
    qa = [
        ("问题① 参数从哪来？", [
            "Z-N：人工公式代入 FOPDT 辨识结果",
            "IMC / BO / 安全 BO / LLM：投运前离线自动搜索（λ 扫描 / GP+EI / 风险 GP / 工具调用）",
            "FNN / RL：离线训练策略 + 在线每 2 s 查表自整定",
        ]),
        ("问题② 要多久？", [
            "PC 口径 0.16–16.22 s，全程纯 CPU、无 GPU / 深度学习框架",
            "等效对象口径 168–4181 h——真机上不可行，AI 路线用离线仿真承担全部试错",
        ]),
        ("问题③ 效果如何？", [
            "RL 37.61 全场最优（PASSED）；FNN 39.26 通过验收门；IMC/BO 相当（39.53/39.72）；Z-N 58.19 最差",
            "全部算法稳定率 100%、零失控；安全 BO 在其基准上均值与方差双优",
        ]),
    ]
    y = 1.30
    for title, lines in qa:
        add_text(s, IN(0.55), IN(y), IN(3.1), IN(0.6),
                 [L(title, size=14.5, bold=True, color=C_PRIMARY, space_after=0)])
        add_text(s, IN(3.85), IN(y - 0.03), IN(9.05), IN(1.5),
                 [L("· " + t, size=12, space_after=5) for t in lines])
        y += 1.52
    add_text(s, IN(0.55), IN(6.10), IN(12.25), IN(1.1), [
        L("AI 收益的本质：不在单点目标值的大幅领先（约 5%），而在免人工、免真机试错、跨工况自适应且带安全回退——以及把 4000+ 对象小时的整定成本压缩到 PC 秒级。", size=12.5, bold=True, color=C_ACCENT, space_after=5),
        L("诚信声明：以上结论限于仿真与 SIL 层；真机验收未开始。", size=11.5, color=C_MUTED, space_after=0),
    ])

    prs.save(OUT)
    print(f"saved: {OUT}")
    print(f"slides: {len(prs.slides._sldIdLst)}")


if __name__ == "__main__":
    build()
