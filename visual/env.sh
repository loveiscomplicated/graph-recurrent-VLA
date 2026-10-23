# Source this file from bash: source /path/to/visual/env.sh
_visual_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$_visual_dir/.venv/bin/activate" ]]; then
    echo "Missing visual/.venv; run: cd \"$_visual_dir\" && bash setup.sh" >&2
    unset _visual_dir
    return 1
fi
# Prevent inherited interpreter settings from leaking into the isolated environment.
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export VISION_ROOT="${VISION_ROOT:-$_visual_dir}"
export VISION_CACHE="${VISION_CACHE:-$VISION_ROOT/cache}"
export HF_HOME="$VISION_CACHE/hf"
export PIP_CACHE_DIR="$VISION_CACHE/pip"
export TORCH_HOME="$VISION_CACHE/torch"
export UV_CACHE_DIR="$VISION_CACHE/uv"
source "$_visual_dir/.venv/bin/activate"
# Preserve an explicitly selected ICD; default to the historical EGL workaround.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$VISION_ROOT/vulkan/nvidia_icd_egl.json}"
unset _visual_dir
