#!/bin/bash
#
# ==============================================================================
#      项目 'ANT' 环境安装脚本 (为中国大陆用户优化)
# ==============================================================================
#
# 功能:
#   1. 创建一个名为 'ant' 的 Conda 环境 (可配置)，并指定 Python 3.10。
#   2. 在 'ant' 环境中安装特定 CUDA 版本的 PyTorch。
#   3. 根据 'requirements.txt' 文件安装所有其他 Python 依赖。
#   4. 安装 ffmpeg 和 x264 编解码器。
#   5. 显示一条重要提示，指导用户手动修改 CLIP 库文件。
#
# 注意:
#   此脚本不会修改您的全局 Conda 或 Pip 配置。
#   所有镜像源的指定都是临时的，仅在本次脚本执行期间有效。
#
# 使用方法:
#   1. 将此脚本与 'requirements.txt' 文件放在同一目录下。
#   2. 在终端中运行:  bash ./setup_env_cn.sh
#
# ==============================================================================

# 如果任何命令执行失败，则立即退出脚本
set -e

# --- 可配置变量 ---
# 您可以在这里修改希望创建的 Conda 环境名称
ENV_NAME="openANT_test"

# --- 步骤 1: 创建 Conda 环境 ---
echo ">>> 步骤 1/5: 创建名为 '${ENV_NAME}' 的 Conda 环境 (Python 3.10)..."
# 使用 -c 参数临时指定清华源
conda create -n ${ENV_NAME} python=3.10 -y \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch \

echo "环境 '${ENV_NAME}' 创建成功。"
echo ""

# --- 步骤 2: 安装 PyTorch ---
echo ">>> 步骤 2/5: 在 '${ENV_NAME}' 环境中安装 PyTorch (cu126)..."
# 此命令从 PyTorch 官方源下载，以确保 CUDA 版本正确
conda run --no-capture-output -n ${ENV_NAME} pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

echo "PyTorch 安装成功。"
echo ""

# --- 步骤 3: 安装 'requirements.txt' 中的依赖 ---
echo ">>> 步骤 3/5: 安装 'requirements.txt' 中的所有依赖包..."
# 使用 --index-url 参数临时指定 pip 的清华源
conda run --no-capture-output -n ${ENV_NAME} pip install -r requirements.txt 
echo "依赖包安装成功。"
echo ""

# --- 步骤 4: 安装 ffmpeg 和 x264 ---
echo ">>> 步骤 4/5: 安装 ffmpeg 和 x264..."
# 再次临时指定 conda-forge 的清华源
conda install -n ${ENV_NAME} ffmpeg x264=20131218 -y -c conda-forge 

echo "ffmpeg 和 x264 安装成功。"
echo ""

# --- 步骤 5: 显示 CLIP 手动修改提示 ---
echo ">>> 步骤 5/5: 显示重要提示信息..."

# 首先，自动执行命令来获取 model.py 的确切路径
echo "--> 正在自动查找 'clip/model.py' 的文件路径..."
CLIP_MODEL_PATH=$(conda run -n ${ENV_NAME} python -c "import clip; print(clip.__file__.replace('__init__.py', 'model.py'))")
echo "--> 文件路径查找成功!"
echo ""


# 然后，在下面的提示信息中打印这个路径
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}"
echo "========================================================================================"
echo "  重要提示：需要为 CLIP 包进行一项手动代码修改"
echo "========================================================================================"
echo -e "${NC}"
echo ""
echo "为了确保在使用半精度 (FP16) 推理时获得稳定性能，您需要手动修改 'clip' 库中的一个文件。"
echo ""
echo "---"
echo "  1. 激活环境: conda activate ${ENV_NAME}"
echo ""
echo "  2. 找到并打开以下文件 (在 VS Code 等编辑器中通常可直接点击跳转):"
echo -e "${RED}"
echo "     ${CLIP_MODEL_PATH}"
echo -e "${NC}"
echo ""
echo "  3. 将文件中的 'class LayerNorm(nn.LayerNorm):' 代码块替换为以下内容:"
echo ""
echo -e "${RED}vvvvvvvvvvv 【请复制并粘贴下面的全部代码】 vvvvvvvvvvv${NC}"
echo "class LayerNorm(nn.LayerNorm):"
echo "    \"\"\"Subclass torch's LayerNorm to handle fp16.\"\"\""
echo ""
echo "    def forward(self, x: torch.Tensor):"
echo "        if self.weight.dtype == torch.float32:"
echo "            orig_type = x.dtype"
echo "            ret = super().forward(x.type(torch.float32))"
echo "            return ret.type(orig_type)"
echo "        else:"
echo "            return super().forward(x)"
echo -e "${RED}^^^^^^^^^^^^^ 【以上是要粘贴的全部代码】 ^^^^^^^^^^^^^${NC}"
echo ""
echo "修改完成后请保存文件。"
echo ""

echo "========================================================================================"
echo "  安装脚本执行完毕！"
echo "  请不要忘记根据上面的提示手动修改 CLIP 文件。"
echo "  使用 'conda activate ${ENV_NAME}' 来激活并开始使用您的环境。"
echo "========================================================================================"