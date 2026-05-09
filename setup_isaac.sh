#!/bin/bash
# 设置 Isaac ROS GXF 扩展库路径
export LD_LIBRARY_PATH=$HOME/ISAAC/install/gxf_isaac_optimizer/share/gxf_isaac_optimizer/gxf/lib:$LD_LIBRARY_PATH

for sys_dir in /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu; do
    [ -d "$sys_dir" ] && export LD_LIBRARY_PATH="$sys_dir:$LD_LIBRARY_PATH"
done

# 设置 GXF 扩展路径（GXF 运行时必需）
export GXF_EXTENSION_PATH=$HOME/ISAAC/install/gxf_isaac_optimizer/share/gxf_isaac_optimizer/gxf/extensions:$GXF_EXTENSION_PATH

# 设置其他可能需要的 GXF 包路径（批量添加）
for pkg in $(find $HOME/ISAAC/install -type d -name "gxf" 2>/dev/null | grep -E "/share/[^/]+/gxf$"); do
    lib_dir="$pkg/lib"
    ext_dir="$pkg/extensions"
    if [ -d "$lib_dir" ]; then
        bad_nvbuf="$lib_dir/libnvbufsurface.so.1.0.0"
        if [ -f "$bad_nvbuf" ]; then
            magic=$(head -c 4 "$bad_nvbuf" 2>/dev/null || true)
            if [ "$magic" != $'\x7fELF' ]; then
                continue
            fi
        fi
        export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
    fi
    [ -d "$ext_dir" ] && export GXF_EXTENSION_PATH="$ext_dir:$GXF_EXTENSION_PATH"
done

echo "✓ Isaac ROS GXF environment configured"
echo "  LD_LIBRARY_PATH includes: $(echo $LD_LIBRARY_PATH | cut -d: -f1)"
echo "  GXF_EXTENSION_PATH includes: $(echo $GXF_EXTENSION_PATH | cut -d: -f1)"

