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
"""Utilities for tables-to-text."""


def get_highlighted_subtable(table, cell_indices):
  """Extract out the highlighted part of a table."""
  highlighted_table = []
  for (row_index, col_index) in cell_indices:
    cell = table[row_index][col_index]
    highlighted_table.append(cell)

  return highlighted_table


def get_table_parent_format(table, table_page_title, table_section_title,
                            table_section_text):
  """Convert table to format required by PARENT."""
  table_parent_array = []

  # Table values.
  for row in table:
    for cell in row:
      if cell["is_header"]:
        attribute = "header"
      else:
        attribute = "cell"
      value = cell["value"].strip()
      if value:
        value = value.replace("|", "-")
        entry = "%s|||%s" % (attribute, value)
        table_parent_array.append(entry)

  # Page title.
  if table_page_title:
    table_page_title = table_page_title.replace("|", "-")
    entry = "%s|||%s" % ("page_title", table_page_title)
    table_parent_array.append(entry)

  # Section title.
  if table_section_title:
    table_section_title = table_section_title.replace("|", "-")
    entry = "%s|||%s" % ("section_title", table_section_title)
    table_parent_array.append(entry)

  # Section text.
  if table_section_text:
    table_section_text = table_section_text.replace("|", "-")
    entry = "%s|||%s" % ("section_text", table_section_text)
    table_parent_array.append(entry)

  table_parent_str = "\t".join(table_parent_array)
  return table_parent_str


def get_subtable_parent_format(subtable, table_page_title, table_section_title):
  """Convert subtable to PARENT format. Do not include section text."""
  table_parent_array = []
  # Table values.
  for cell in subtable:
    if cell["is_header"]:
      attribute = "header"
    else:
      attribute = "cell"
    value = cell["value"].strip()
    if value:
      value = value.replace("|", "-")
      entry = "%s|||%s" % (attribute, value)
      table_parent_array.append(entry)

  # Page title.
  if table_page_title:
    table_page_title = table_page_title.replace("|", "-")
    entry = "%s|||%s" % ("page_title", table_page_title)
    table_parent_array.append(entry)

  # Section title.
  if table_section_title:
    table_section_title = table_section_title.replace("|", "-")
    entry = "%s|||%s" % ("section_title", table_section_title)
    table_parent_array.append(entry)

  table_parent_str = "\t".join(table_parent_array)
  return table_parent_str
