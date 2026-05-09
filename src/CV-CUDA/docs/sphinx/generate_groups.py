# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


if __name__ == "__main__":
    c_api_dir = Path(sys.argv[1])
    cpp_api_dir = Path(sys.argv[2])
    xmlRoot = sys.argv[3]

    os.makedirs(c_api_dir, exist_ok=True)
    os.makedirs(cpp_api_dir, exist_ok=True)

    for i in os.listdir(xmlRoot):
        group_path = os.path.join(xmlRoot, i)
        if os.path.isfile(group_path) and "group__" in i:
            tree = ET.parse(group_path)
            root = tree.getroot()
            for compounddef in root.iter("compounddef"):
                group_name = compounddef.attrib["id"]
                group_label = compounddef.find("compoundname").text
                group_title = compounddef.find("title").text

                # Determine output directory based on group name
                if "group__NVCV__CPP__" in group_name:
                    outdir = cpp_api_dir
                    is_cpp = True
                elif "group__NVCV__C__" in group_name:
                    outdir = c_api_dir
                    is_cpp = False
                else:
                    # Default to C API for any other groups
                    outdir = c_api_dir
                    is_cpp = False

                outfile = outdir / (group_name + ".rst")
                output = ":orphan:\n\n"
                output += group_title + "\n"
                output += "=" * len(group_title) + "\n\n"

                # For C++ API, add a note pointing to the corresponding C API documentation
                if is_cpp:
                    # Generate the corresponding C API group name and file
                    c_group_name = group_name.replace("__CPP__", "__C__")
                    # Sphinx :doc: directive expects path without .rst extension
                    c_group_file = f"../_c_api/{c_group_name}"

                    output += ".. note::\n"
                    output += (
                        "   The C++ API provides RAII wrappers around the C API.\n"
                    )
                    output += "   For detailed documentation including parameters, "
                    output += "return values, and limitations,\n"
                    output += "   please refer to the "
                    output += ":doc:`corresponding C API documentation "  # noqa: E231
                    output += f"<{c_group_file}>`.\n"  # noqa: W604, E231
                    output += "\n"

                output += f".. doxygengroup:: {group_label}\n"  # noqa: E231
                output += "   :project: cvcuda\n"  # noqa: E231

                # For C++ API, don't show the namespace members to avoid recursive namespace cvcuda
                if is_cpp:
                    output += "   :content-only:\n"

                outfile.write_text(output)
