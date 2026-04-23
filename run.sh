#!/bin/bash
echo "=== 1. Exporting Weights & Data ==="
python3 scripts/train_minst.py

echo "=== 2. Building C++ Engine ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo "=== 3. Running Inference ==="
./tiny_infer

