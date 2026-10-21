# 사용: visual/ 에서 source env.sh
# VISION_ROOT(코드·데이터·체크포인트), VISION_CACHE(HF/pip/torch 캐시), VISION_ENVS(venv 위치)는 필요하면 미리 export 해서 바꾼다.
export VISION_ROOT=${VISION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export VISION_CACHE=${VISION_CACHE:-$VISION_ROOT/cache}
export VISION_ENVS=${VISION_ENVS:-$HOME/envs}
export HF_HOME=$VISION_CACHE/hf
export PIP_CACHE_DIR=$VISION_CACHE/pip
export TORCH_HOME=$VISION_CACHE/torch
source $VISION_ENVS/vision/bin/activate
# 컨테이너 기본 Vulkan ICD(libGLX_nvidia)로는 vkCreateInstance 가 실패해서 EGL ICD 를 쓴다 (docs/known_issues.md)
export VK_ICD_FILENAMES=$VISION_ROOT/vulkan/nvidia_icd_egl.json
