"""将Markdown实验报告导出为提交用PDF，图片路径相对于Markdown文件解析。

样式都集中在下面的STYLE字典里，直接改默认值，或用命令行临时覆盖：
    python experiments/export_report.py --set h2_size=12pt --set cjk_font="Noto Sans CJK SC"
"""
import argparse
import re
import struct
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]

STYLE = {
    # ---- 字体：英文/数字先用latin_font，找不到的字（中文、全角标点）自动回退到cjk_font ----
    "latin_font": '"Noto Sans CJK SC"',  # 与安装新字体前PDF的实际正文用字一致
    "cjk_font": '"Noto Sans CJK SC"',
    "mono_font": '"DejaVu Sans Mono"',  # 显式使用安装新字体前的字体，避免系统回退随安装变化
    # ---- 字号与行距 ----
    "body_size": "10pt",
    "line_height": "1.3",          # 倍数，1.2紧凑，1.5宽松
    "h1_size": "14pt",
    "h2_size": "12pt",
    "h3_size": "10pt",
    "code_size": "9pt",            # 行内代码字号
    "table_size": "8.5pt",
    "caption_size": "8.5pt",
    # ---- 间距 ----
    "page_margin": "10mm",
    "h3_runin": "1",               # 1=三级标题与后面的正文同一行（紧跟，不换行）；0=标题单独一行
    "para_gap": "3pt",             # 段落上下间距
    "para_align": "justify",       # 正文对齐：justify（两端对齐）或left
    "strip_cjk_spaces": "1",       # 1=去掉中文与英文/代码之间的空格。两端对齐时多余宽度只会摊到空格上，这些空格会被拉得很宽
    "h2_gap_top": "3pt",           # 二级标题与上文的间距
    "h3_gap_top": "3pt",           # 三级标题与上文的间距
    "fig_float_below": "0.65",      # 图片显示宽度小于正文宽度的这个比例时，靠左浮动，右侧环绕正文；设为0则不浮动
    "fig_float_force": "1,4,6,7,8",      # 不论宽窄，强制靠左浮动的图编号（图注以“图N：”开头），逗号分隔
    "fig_block_force": "",            # 指定图片独占一行并居中，优先于自动浮动和强制浮动
    "fig_float_width": "105mm",     # 强制浮动的图最宽不超过这个宽度，留出右侧正文栏
    "fig_max_height": "48mm",      # 图片最大高度（图里文字会随之缩小，太小会看不清）
    # ---- 颜色 ----
    "text_color": "#000000",
    "h_color": "#1f2d6b",          # 一、二、三级标题统一用同一种颜色
    "table_head_bg": "#e9ecf8",
}

CSS = Template('''
@page { size: A4; margin: ${page_margin};
  @bottom-right { content: counter(page) " / " counter(pages); color:#64748b; font-size:8pt;
                  font-family:${latin_font}, ${cjk_font}, serif; }
}
body {font-family: ${latin_font}, ${cjk_font}, serif; font-size:${body_size};
      line-height:${line_height}; color:${text_color};}
h1, h2, h3 {font-family: ${latin_font}, ${cjk_font}, serif; font-weight:700; color:${h_color};
            line-height:1.25; break-after:avoid;}
h1 {font-size:${h1_size}; margin:0 0 2pt; text-align:center;}
h1 + p, h1 + p + p {text-align:center;}
h2 {font-size:${h2_size}; margin:${h2_gap_top} 0 3pt; padding-bottom:1.5pt; border-bottom:.6pt solid #c8cee6;}
h3 {font-size:${h3_size}; margin:${h3_gap_top} 0 2pt;}
p.runin {margin-top:${h3_gap_top};}
span.runin-title {font-family:${latin_font}, ${cjk_font}, serif; font-weight:700; font-size:${h3_size};
                  color:${h_color}; margin-right:.6em;}
p {margin:${para_gap} 0; text-align:${para_align}; overflow-wrap:anywhere;}
strong {font-weight:700;}
code {font-family: ${mono_font}, ${cjk_font}, monospace; font-size:${code_size}; color:#1f2d6b;
      background:#f1f3fa; padding:0 1.5pt; border-radius:1.5pt;
      white-space:normal; word-break:break-all; overflow-wrap:anywhere;  /* 任何行内代码都允许在任意位置换行 */
      letter-spacing:0; word-spacing:0; font-variant-ligatures:none;}
table {width:100%; border-collapse:collapse; font-size:${table_size}; margin:6pt 0; line-height:1.3;}
th {background:${table_head_bg}; color:${h_color}; font-weight:700; text-align:center;
    padding:4pt 3pt; border-bottom:1pt solid #c8cee6;}
td {padding:3pt; border-bottom:.4pt solid #dce3ed; text-align:center;}
tr:nth-child(even) td {background:#f7f9fc;}
figure {break-inside:avoid;}
tr {break-inside:avoid;}
thead {display:table-header-group;}  /* 表格跨页时每页重复表头；表格本身允许跨页，减少页底留白 */
figure {margin:5pt 0 4pt; text-align:center;}
figure.side {float:left; margin:2pt 4mm 2pt 0;}
figure.block {float:none; clear:both; width:100%;}
h2, table {clear:both;}   /* 标题和表格不与浮动图片并排 */
img {width:100%; height:auto; max-height:${fig_max_height}; object-fit:contain;}
figure.side img {max-height:none;}
figcaption {font-size:${caption_size}; color:#333; margin:2pt 0 0; text-align:center; line-height:1.3;}
''')

# 单独成行的 ![图注](路径) 先换成占位符，渲染完再替换成带图注的<figure>。
# （不直接用markdown-it的alt：它会丢掉 \\* 这类转义字符，图注单独用renderInline渲染。）
_FIG_LINE = re.compile(r'^!\[(.*)\]\((.*?)\)[ \t]*$', re.M)


_CJK = "\u3400-\u9fff\u3000-\u303f\uff00-\uffef"


def _strip_cjk_spaces(source):
    """去掉紧挨着中文字符（含全角标点）的半角空格；表格和标题不动。"""
    out = []
    for line in source.split("\n"):
        if line.lstrip().startswith(("|", "#")):   # 表格行、标题行不处理
            out.append(line)
            continue
        line = re.sub(f"(?<=[{_CJK}]) +(?=[^ |])", "", line)
        line = re.sub(f"(?<=[^ |]) +(?=[{_CJK}])", "", line)
        if re.search(f"[{_CJK}]", line):   # 英文词之间的单个空格改成窄空格，避免被两端对齐拉宽
            line = re.sub(r"(?<=[A-Za-z0-9*)]) (?=[A-Za-z0-9(*])", "\u2009", line)
        out.append(line)
    return "\n".join(out)


def build_html(source, style=None, base_dir=None):
    """使用标准Markdown解析器处理强调、转义、链接、图片和表格。"""
    from markdown_it import MarkdownIt

    merged = {**STYLE, **(style or {})}
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    figures = []

    base = Path(base_dir) if base_dir else ROOT / "report"
    page_mm = 210 - 2 * float(re.sub(r"[^\d.]", "", merged["page_margin"]))   # 正文宽度(mm)，边距按mm填写
    max_h = float(re.sub(r"[^\d.]", "", merged["fig_max_height"]))
    ratio = float(merged["fig_float_below"])

    def figure_html(path, caption):
        side, style = "", ""
        number = re.match(r"图(\d+)", re.sub(r"<[^>]+>", "", caption))
        block = bool(number) and number.group(1) in merged["fig_block_force"].replace(" ", "").split(",")
        if block:
            side = " block"
        try:
            header = (base / path).read_bytes()[:24]
            width, height = struct.unpack(">II", header[16:24])
            shown = min(page_mm, max_h * width / height)   # 受最大高度限制后的显示宽度
            forced = bool(number) and number.group(1) in merged["fig_float_force"].replace(" ", "").split(",")
            if forced:
                shown = min(shown, float(re.sub(r"[^\d.]", "", merged["fig_float_width"])))
            if not block and (forced or (ratio and shown < page_mm * ratio)):
                side, style = " side", f' style="width:{shown:.1f}mm"'
        except (OSError, struct.error):
            pass
        return (f'<figure class="fig{side}"{style}><img src="{path}" alt="">'
                f'<figcaption>{caption}</figcaption></figure>')

    def stash(match):
        figures.append((match.group(2), md.renderInline(match.group(1))))
        return f"FIGPLACEHOLDER{len(figures) - 1}END"

    if merged["strip_cjk_spaces"] == "1":
        source = _strip_cjk_spaces(source)
    body = md.render(_FIG_LINE.sub(stash, source))
    for index, (path, caption) in enumerate(figures):
        body = body.replace(
            f"<p>FIGPLACEHOLDER{index}END</p>",
            figure_html(path, caption))
    if merged["h3_runin"] == "1":   # 三级标题后面紧跟正文：<h3>标题</h3><p>正文 → <p><span>标题</span>正文
        body = re.sub(r'<h3>(.*?)</h3>\n<p>',
                      lambda m: '<p class="runin"><span class="runin-title">' + m.group(1).replace(" ", "\u2002") + '</span>',
                      body)   # 标题内的空格用固定宽度空格，避免被两端对齐拉宽
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>实验二报告</title><style>' + CSS.substitute(merged) + '</style><body>'
        + body + '</body></html>'
    )


def report_short_last_lines(document, limit):
    """遍历排版结果，列出段落末行字数不超过limit的段落（含页码、末行内容、段首文字）。"""
    from weasyprint.formatting_structure import boxes

    def text_of(box):
        return "".join(b.text for b in box.descendants() if isinstance(b, boxes.TextBox))

    found = 0
    for number, page in enumerate(document.pages, 1):
        for box in page._page_box.descendants():
            lines = [c for c in getattr(box, "children", []) if isinstance(c, boxes.LineBox)]
            if box.element_tag != "p" or len(lines) < 2 or lines[0] is not box.children[0]:   # 只看正文段落，跳过表格单元格
                continue
            last = text_of(lines[-1]).strip()
            if 0 < len(last) <= limit:
                found += 1
                print(f"第{number}页 末行「{last}」({len(last)}字) 段首：{text_of(lines[0]).strip()[:24]}")
    print(f"共{found}段末行不超过{limit}个字")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "report" / "report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "report_preview.pdf")
    parser.add_argument("--short-last-line", type=int, default=0, metavar="N",
                        help="列出最后一行不超过N个字的段落（用来压缩页数），不影响导出")
    parser.add_argument("--set", action="append", default=[], metavar="键=值",
                        help="覆盖STYLE中的某一项，可重复使用；可用的键：" + "、".join(STYLE))
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in STYLE:
            parser.error(f"未知样式项{key!r}，可用：{', '.join(STYLE)}")
        overrides[key] = value

    from weasyprint import HTML

    source = args.input.read_text(encoding="utf-8")
    document = HTML(string=build_html(source, overrides, args.input.parent), base_url=str(args.input.parent.resolve())).render()
    if args.short_last_line:
        report_short_last_lines(document, args.short_last_line)
    document.write_pdf(args.output)
    print(f"已导出{args.output}，共{len(document.pages)}页。")


if __name__ == "__main__":
    main()
