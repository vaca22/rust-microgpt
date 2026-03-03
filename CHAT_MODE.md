# 💬 对话模式使用指南

## 快速启动

### 方式 1: 使用脚本
```bash
./chat.sh
```

### 方式 2: 命令行参数
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release -- --chat
```

或简写：
```bash
cargo run --release -- -c
```

## 对话示例

```
🤖 小说专家 - 对话模式
==================================================
输入文字，我会继续生成小说内容
输入 'quit' 或 'exit' 退出
输入 'clear' 清空对话历史
==================================================

📝 你 > Once upon a time
🤖 AI > , there was a brave knight who journeyed...

📝 你 > Tell me more about the knight
🤖 AI > . He was known throughout the land for his...

📝 你 > quit
👋 再见！
```

## 命令

| 命令 | 说明 |
|------|------|
| 任意文字 | 继续生成小说内容 |
| `quit` / `exit` / `q` | 退出对话 |
| `clear` | 清空对话历史 |

## 工作原理

1. **首次运行**：模型会先训练 1000 步（约 10-20 秒）
2. **输入文字**：你的输入作为 prompt
3. **模型生成**：基于训练的小说风格继续生成
4. **多轮对话**：历史会被保留（最多 BLOCK_SIZE 字符）

## 限制

- **上下文长度**：最多 16 字符（BLOCK_SIZE）
- **生成长度**：每次最多 100 字符
- **语言**：基于英文小说训练，英文效果更好
- **模型大小**：10K 参数，生成内容较简单

## 提升效果

### 增加上下文长度
编辑 `src/model.rs`:
```rust
pub const BLOCK_SIZE: usize = 64;  // 从 16 增加到 64
```

### 增加生成长度
编辑 `src/chat.rs`:
```rust
const INFERENCE_LENGTH: usize = 200;  // 从 100 增加到 200
```

### 调整温度（随机性）
编辑 `src/chat.rs`:
```rust
let temperature = 0.8;  // 0.3-1.0, 越高越随机
```

## 退出对话

- 输入 `quit`、`exit` 或 `q`
- 或按 `Ctrl+C`

## 提示

- 输入小说开头，让模型继续
- 输入角色名，生成相关描述
- 输入场景，生成情节发展
- 多轮对话可以构建完整故事
