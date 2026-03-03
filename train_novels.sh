#!/bin/bash
# 小说专家模型训练脚本

set -e

PROJECT_DIR="/Users/macbook/.openclaw/workspaces/executor/rust-microgpt"
DATA_FILE="/Users/macbook/.openclaw/workspaces/executor/novels_training.txt"

cd "$PROJECT_DIR"

echo "📚 准备训练数据..."

# 复制训练数据到项目目录
cp "$DATA_FILE" ./input.txt

echo "✓ 数据已复制到 $PROJECT_DIR/input.txt"
echo "  文件大小：$(du -h input.txt | cut -f1)"
echo "  行数：$(wc -l < input.txt)"

echo ""
echo "🔥 开始训练小说专家模型..."
echo "  使用优化编译：RUSTFLAGS=\"-C target-cpu=native\""
echo "  发布模式：--release"
echo ""

# 运行训练（带优化标志）
RUSTFLAGS="-C target-cpu=native" cargo run --release

echo ""
echo "✅ 训练完成!"
echo "  模型已生成并输出了 20 段小说风格的文本"
