def format_table(rows, columns):
    if not rows:
        return "(no data)"

    prepared_rows = []
    widths = [len(column["title"]) for column in columns]

    for row in rows:
        prepared_row = []
        for index, column in enumerate(columns):
            value = row[column["key"]]
            rendered = _format_cell(value)
            prepared_row.append(rendered)
            widths[index] = max(widths[index], len(rendered))
        prepared_rows.append(prepared_row)

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = _render_row([column["title"] for column in columns], widths, columns)
    body = [_render_row(row, widths, columns) for row in prepared_rows]

    return "\n".join([separator, header, separator, *body, separator])


def _format_cell(value):
    if isinstance(value, float):
        rendered = f"{value:.6f}".rstrip("0").rstrip(".")
        return rendered if rendered else "0"
    return str(value)


def _render_row(values, widths, columns):
    cells = []
    for value, width, column in zip(values, widths, columns):
        align = column.get("align", "left")
        if align == "right":
            cells.append(value.rjust(width))
        else:
            cells.append(value.ljust(width))
    return "| " + " | ".join(cells) + " |"
