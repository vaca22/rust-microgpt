# 🤖 小说专家模型

基于 Andrej Karpathy MicroGPT 的 Rust 实现，训练于 288 本经典小说。

## 🚀 快速开始

### 普通推理（生成 20 个片段）
```bash
cd /Users/macbook/.openclaw/workspaces/executor/rust-microgpt
./run_infer.sh
```

### 💬 对话模式
```bash
./chat.sh
```

或：
```bash
cargo run --release -- --chat
```

## 📊 模型信息

| 项目 | 数值 |
|------|------|
| 训练数据 | 288 本经典小说 |
| 总字符数 | 1.06 亿 |
| 词汇表 | 215 字符 |
| 参数量 | 10,208 |
| 上下文长度 | 16 字符 |
| 训练步数 | 1000 |

## 📚 训练数据包含

- 《双城记》《大卫·科波菲尔》《傲慢与偏见》
- 《战争与和平》《安娜·卡列尼娜》
- 《基督山伯爵》《三个火枪手》
- 《福尔摩斯探案集》《爱丽丝梦游仙境》
- 等 288 部经典作品

## 📁 文件结构

```
rust-microgpt/
├── src/
│   ├── main.rs        # 主程序
│   ├── chat.rs        # 对话模式
│   ├── model.rs       # 模型定义
│   ├── tape.rs        # 自动微分引擎
│   └── mt19937_rng.rs # 随机数生成器
├── input.txt          # 训练数据
├── run_infer.sh       # 推理脚本
├── chat.sh            # 对话脚本
├── CHAT_MODE.md       # 对话模式文档
└── INFERENCE.md       # 推理文档
```

## 💡 对话示例

```
🤖 小说专家 - 对话模式
==================================================

📝 你 > Once upon a time
🤖 AI > , there was a brave knight...

📝 你 > Tell me more
🤖 AI > . He journeyed to the castle...

📝 你 > quit
👋 再见！
```

## ⚙️ 调整参数

### 增加上下文长度
编辑 `src/model.rs`:
```rust
pub const BLOCK_SIZE: usize = 64;  // 默认 16
```

### 增加训练步数
编辑 `src/main.rs`:
```rust
const NUM_STEPS: usize = 10000;  // 默认 1000
```

### 调整温度（随机性）
编辑 `src/chat.rs`:
```rust
let temperature = 0.8;  // 0.3-1.0
```

## 📖 文档

- [对话模式指南](CHAT_MODE.md)
- [推理指南](INFERENCE.md)

## 🎯 下一步

1. **更大模型** - 增加 `N_EMBD`, `N_LAYER`
2. **更长上下文** - 增加 `BLOCK_SIZE`
3. **更多训练** - 增加 `NUM_STEPS`
4. **保存权重** - 添加权重序列化功能

## 📄 许可证

原始 MicroGPT by Andrej Karpathy (MIT)
Rust 翻译及小说训练数据
