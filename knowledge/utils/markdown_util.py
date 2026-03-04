import re
from typing import List
from bs4 import BeautifulSoup


class MarkdownTableLinearizer:
    """
    解决：HTML复杂合并单元格、无表头KV表、左上角空置交叉表、原生MD表
    """

    HTML_TABLE_PATTERN = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    MD_TABLE_PATTERN = re.compile(
        r'((?:^[ \t]*\|.*\|[ \t]*\n)'
        r'(?:^[ \t]*\|[ \t]*[-:]+[-| :]*\|[ \t]*\n)'
        r'(?:^[ \t]*\|.*\|[ \t]*(?:\n|$))*)',
        re.MULTILINE
    )

    @classmethod
    def process(cls, content: str) -> str:
        if not content:
            return content

        if "<table" in content.lower():
            content = cls.HTML_TABLE_PATTERN.sub(cls._replace_html_table, content)

        if "|" in content:
            content = cls.MD_TABLE_PATTERN.sub(cls._replace_md_table, content)

        return content

    @classmethod
    def _replace_html_table(cls, match) -> str:
        html_content = match.group(0)
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table: return html_content

        rows = table.find_all("tr")
        if not rows: return html_content

        # 嗅探 HTML 中是否使用了标准的 <th> 表头标签
        has_th = len(table.find_all("th")) > 0

        grid = []
        for _ in range(len(rows)):
            grid.append([])

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                # 使用 separator 保证 <br> 换行能变成空格，不会粘连
                text = cell.get_text(separator=" ", strip=True)

                for r in range(row_idx, row_idx + rowspan):
                    while len(grid) <= r: grid.append([])
                    while len(grid[r]) < col_idx + colspan: grid[r].append(None)
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = text
                col_idx += colspan

        return cls._grid_to_text(grid, is_md=False, has_th=has_th)

    @classmethod
    def _replace_md_table(cls, match) -> str:
        md_text = match.group(0).strip()
        lines = md_text.split('\n')
        grid = []
        for line in lines:
            if re.match(r'^[ \t]*\|[ \t\-|:]+\|[ \t]*$', line): continue
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            grid.append(cells)
        # Markdown 表格天生自带表头结构
        return cls._grid_to_text(grid, is_md=True, has_th=False)

    @classmethod
    def _grid_to_text(cls, grid: List[List[str]], is_md: bool, has_th: bool) -> str:
        if not grid or not grid[0]: return ""

        cols_count = max(len(r) for r in grid)
        # 补齐不规则行的列数，防越界
        for r in grid:
            while len(r) < cols_count: r.append("")

        is_header_row = False
        if is_md or has_th:
            is_header_row = True
        # 重点防御：如果左上角是空的，这绝对是一个交叉表头！绝不能当废数据丢弃！
        elif grid[0][0] == "":
            is_header_row = True
        # 一般大于两列的表格，第一行基本都是表头
        elif cols_count > 2:
            is_header_row = True

        res = []

        if not is_header_row and cols_count == 2:
            # 【策略 1：纯 K-V 表格】
            for r in grid:
                k, v = r[0], r[1]
                if k or v:
                    k_str = k if k else "未知属性"
                    v_str = v if v else "无"
                    res.append(f"- 【{k_str}】：{v_str}。")
        else:
            # 【策略 2：带有表头定义的标准/交叉表格】
            headers = grid[0]
            for r in grid[1:]:
                # 跳过完全空的数据行
                if not any(r): continue

                subject = r[0] if r[0] else "未知项目"
                subject_header = headers[0] if headers[0] else ""

                props = []
                for c in range(1, cols_count):
                    head = headers[c] if headers[c] else f"属性{c}"
                    val = r[c] if r[c] else ""

                    if val and val not in ('-', '/', '\\', '无'):
                        props.append(f"{head}为{val}")

                if props:
                    prop_str = "，".join(props)
                    if subject_header:
                        res.append(f"- 【{subject}】(对应{subject_header})：{prop_str}。")
                    else:
                        # 针对左上角为空的情况，隐藏对应关系描述
                        res.append(f"- 【{subject}】：{prop_str}。")
                else:
                    if subject != "未知项目":
                        res.append(f"- 【{subject}】")

        return "\n\n" + "\n".join(res) + "\n\n"