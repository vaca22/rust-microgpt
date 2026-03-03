#!/bin/bash
# 快速运行推理脚本

set -e

cd "/Users/macbook/.openclaw/workspaces/executor/rust-microgpt"

echo "🤖 小说专家模型 - 推理模式"
echo "=========================================="
echo ""

# 检查 input.txt 是否存在
if [ ! -f "input.txt" ]; then
    echo "❌ 错误：input.txt 不存在"
    echo "   请先运行训练数据准备"
    exit 1
fi

echo "📊 数据文件信息:"
echo "   大小：$(du -h input.txt | cut -f1)"
echo "   行数：$(wc -l < input.txt)"
echo ""

echo "🔥 开始训练并推理..."
echo ""

# 运行训练 + 推理
RUSTFLAGS="-C target-cpu=native" cargo run --release

echo ""
echo "✅ 完成!"
