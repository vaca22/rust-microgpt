#!/bin/bash
# 对话模式启动脚本

set -e

cd "/Users/macbook/.openclaw/workspaces/executor/rust-microgpt"

echo "🤖 小说专家 - 对话模式"
echo "=========================================="
echo ""

# 检查 input.txt 是否存在
if [ ! -f "input.txt" ]; then
    echo "❌ 错误：input.txt 不存在"
    echo "   请先运行：cp ../novels_training.txt ./input.txt"
    exit 1
fi

echo "📚 训练数据：$(du -h input.txt | cut -f1)"
echo ""
echo "🔥 开始训练（首次需要等待）..."
echo ""

# 运行对话模式
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --chat
