#!/bin/bash
#
# ==============================================================================
#      Installation Script for the 'ANT' Project
# ==============================================================================
#
# This script automates the setup of the required Conda environment.
#
# Features:
#   1. Creates a new, isolated Conda environment with a configurable name.
#   2. Installs a specific CUDA-enabled version of PyTorch.
#   3. Installs all other Python dependencies from the 'requirements.txt' file.
#   4. Installs the ffmpeg and x264 codecs from the official conda-forge channel.
#   5. Displays an important notice guiding the user to manually patch the
#      CLIP library for FP16 stability.
#
# Prerequisites:
#   - A working Conda (Miniconda or Anaconda) installation.
#   - This script must be in the same directory as 'requirements.txt'.
#
# Usage:
#   From your terminal, run:
#   bash ./setup_env.sh
#
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configurable Variables ---
# You can change the desired name for your Conda environment here.
ENV_NAME="openANT_test"

# --- Step 1: Create Conda Environment ---
echo ">>> Step 1/5: Creating Conda environment '${ENV_NAME}' with Python 3.10..."
# Creates the environment using the default Conda channels.
conda create -n ${ENV_NAME} python=3.10 -y 
echo "Environment '${ENV_NAME}' created successfully."
echo ""

# --- Step 2: Install PyTorch ---
echo ">>> Step 2/5: Installing PyTorch for cu126 in '${ENV_NAME}'..."
# This command downloads from the official PyTorch repository to ensure the correct CUDA version.
# The --no-capture-output flag provides a real-time installation log.
conda run --no-capture-output -n ${ENV_NAME} pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
echo "PyTorch installed successfully."
echo ""

# --- Step 3: Install Dependencies from requirements.txt ---
echo ">>> Step 3/5: Installing all other dependencies from 'requirements.txt'..."
# Installs packages from the default PyPI repository.
conda run --no-capture-output -n ${ENV_NAME} pip install -r requirements.txt
echo "Dependencies installed successfully."
echo ""

# --- Step 4: Install ffmpeg and x264 ---
echo ">>> Step 4/5: Installing ffmpeg and x264..."
# Installs from the official conda-forge channel.
conda install -n ${ENV_NAME} ffmpeg x264=20131218 -y -c conda-forge
echo "ffmpeg and x264 installed successfully."
echo ""

# --- Step 5: Display Manual Patch Instructions for CLIP ---
echo ">>> Step 5/5: Displaying important notice..."

# First, automatically find the exact path to the model.py file.
echo "--> Automatically locating 'clip/model.py' file path..."
CLIP_MODEL_PATH=$(conda run -n ${ENV_NAME} python -c "import clip; print(clip.__file__.replace('__init__.py', 'model.py'))")
echo "--> File path found!"
echo ""

# Define colors for the notice.
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display the detailed instructions.
echo -e "${RED}"
echo "========================================================================================"
echo "  IMPORTANT: A manual code modification for the CLIP library is required."
echo "========================================================================================"
echo -e "${NC}"
echo ""
echo "To ensure stable performance with half-precision (FP16) inference, you need to"
echo "manually modify one of the library's source files."
echo ""
echo "---"
echo "  1. Activate the environment: conda activate ${ENV_NAME}"
echo ""
echo "  2. Open the following file (modern terminals like VS Code's often allow direct clicking):"
echo -e "${RED}"
echo "     ${CLIP_MODEL_PATH}"
echo -e "${NC}"
echo ""
echo "  3. In that file, replace the entire 'class LayerNorm(nn.LayerNorm):' block"
echo "     with the following code:"
echo ""
echo -e "${RED}vvvvvvvvvvv [Copy and paste the entire block below] vvvvvvvvvvv${NC}"
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
echo -e "${RED}^^^^^^^^^^^^^ [End of the block to copy] ^^^^^^^^^^^^^${NC}"
echo ""
echo "Please save the file after making the change."
echo ""

echo "========================================================================================"
echo "  Installation script finished!"
echo "  Please do not forget to manually patch the CLIP library as instructed above."
echo "  To begin, activate your environment with: conda activate ${ENV_NAME}"
echo "========================================================================================"