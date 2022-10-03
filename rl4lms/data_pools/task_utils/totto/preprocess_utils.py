# Adapted from https://github.com/google-research/language/blob/master/language/totto/baseline_preprocessing/preprocess_data_main.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Baseline preprocessing utilities."""
import copy


def _add_adjusted_col_offsets(table):
    """Add adjusted column offsets to take into account multi-column cells."""
    adjusted_table = []
    for row in table:
        real_col_index = 0
        adjusted_row = []
        for cell in row:
            adjusted_cell = copy.deepcopy(cell)
            adjusted_cell["adjusted_col_start"] = real_col_index
            adjusted_cell["adjusted_col_end"] = (
                    adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"])
            real_col_index += adjusted_cell["column_span"]
            adjusted_row.append(adjusted_cell)
        adjusted_table.append(adjusted_row)
    return adjusted_table


def _get_heuristic_row_headers(adjusted_table, row_index, col_index):
    """Heuristic to find row headers."""
    row_headers = []
    row = adjusted_table[row_index]
    for i in range(0, col_index):
        if row[i]["is_header"]:
            row_headers.append(row[i])
    return row_headers


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
    """Heuristic to find column headers."""
    adjusted_cell = adjusted_table[row_index][col_index]
    adjusted_col_start = adjusted_cell["adjusted_col_start"]
    adjusted_col_end = adjusted_cell["adjusted_col_end"]
    col_headers = []
    for r in range(0, row_index):
        row = adjusted_table[r]
        for cell in row:
            if (cell["adjusted_col_start"] < adjusted_col_end and
                    cell["adjusted_col_end"] > adjusted_col_start):
                if cell["is_header"]:
                    col_headers.append(cell)

    return col_headers


def get_highlighted_subtable(table, cell_indices, with_heuristic_headers=False):
    """Extract out the highlighted part of a table."""
    highlighted_table = []

    adjusted_table = _add_adjusted_col_offsets(table)

    for (row_index, col_index) in cell_indices:
        cell = table[row_index][col_index]
        if with_heuristic_headers:
            row_headers = _get_heuristic_row_headers(adjusted_table, row_index,
                                                     col_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, row_index,
                                                     col_index)
        else:
            row_headers = []
            col_headers = []

        highlighted_cell = {
            "cell": cell,
            "row_headers": row_headers,
            "col_headers": col_headers
        }
        highlighted_table.append(highlighted_cell)

    return highlighted_table


def linearize_full_table(table, cell_indices, table_page_title,
                         table_section_title):
    """Linearize full table with localized headers and return a string."""
    table_str = ""
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "

    table_str += "<table> "
    adjusted_table = _add_adjusted_col_offsets(table)
    for r_index, row in enumerate(table):
        row_str = "<row> "
        for c_index, col in enumerate(row):

            row_headers = _get_heuristic_row_headers(adjusted_table, r_index, c_index)
            col_headers = _get_heuristic_col_headers(adjusted_table, r_index, c_index)

            # Distinguish between highlighted and non-highlighted cells.
            if [r_index, c_index] in cell_indices:
                start_cell_marker = "<highlighted_cell> "
                end_cell_marker = "</highlighted_cell> "
            else:
                start_cell_marker = "<cell> "
                end_cell_marker = "</cell> "

            # The value of the cell.
            item_str = start_cell_marker + col["value"] + " "

            # All the column headers associated with this cell.
            for col_header in col_headers:
                item_str += "<col_header> " + col_header["value"] + " </col_header> "

            # All the row headers associated with this cell.
            for row_header in row_headers:
                item_str += "<row_header> " + row_header["value"] + " </row_header> "

            item_str += end_cell_marker
            row_str += item_str

        row_str += "</row> "
        table_str += row_str

    table_str += "</table>"
    if cell_indices:
        assert "<highlighted_cell>" in table_str
    return table_str


def linearize_subtable(subtable, table_page_title, table_section_title):
    """Linearize the highlighted subtable and return a string of its contents."""
    table_str = ""
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "
    table_str += "<table> "

    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]

        # The value of the cell.
        item_str = "<cell> " + cell["value"] + " "

        # All the column headers associated with this cell.
        for col_header in col_headers:
            item_str += "<col_header> " + col_header["value"] + " </col_header> "

        # All the row headers associated with this cell.
        for row_header in row_headers:
            item_str += "<row_header> " + row_header["value"] + " </row_header> "

        item_str += "</cell> "
        table_str += item_str

    table_str += "</table>"
    return table_str
