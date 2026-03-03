# 🤖 小说专家模型 - 推理指南

## 快速运行

### 方式 1: 直接运行（训练 + 推理）
```bash
cd /Users/macbook/.openclaw/workspaces/executor/rust-microgpt
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

### 方式 2: 使用脚本
```bash
./run_infer.sh
```

## 当前配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 词汇表 | 215 字符 | 英文 + 标点 |
| 训练步数 | 1000 | `NUM_STEPS` |
| 生成长度 | 16 字符 | `BLOCK_SIZE` |
| 样本数量 | 20 个 | 每次推理生成 |
| 温度参数 | 0.5 | 控制随机性 |

## 修改推理参数

### 调整生成长度
编辑 `src/main.rs`:
```rust
const BLOCK_SIZE: usize = 32;  // 增加生成长度 (当前 16)
```

### 调整生成样本数
编辑 `src/main.rs` 推理部分:
```rust
for sample_idx in 0..50 {  // 改为 50 个样本
```

### 调整温度（随机性）
```rust
let temperature = 0.8;  // 越高越随机 (0.3-1.0)
```

## 使用自己的提示词 (Prompt)

当前模型是纯生成模式。要用提示词开头，需要修改推理部分：

```rust
// 在推理前添加提示词
let prompt = "Once upon a time";
let mut samples = prompt.to_string();
let mut token_ids: Vec<usize> = prompt.chars()
    .map(|c| *char_to_idx.get(&c).unwrap_or(&bos_idx))
    .collect();

// 然后从 token_ids 的最后一个继续生成
```

## 输出示例

```
-------- inference --------

0: "Ther desy agear
1: "She aith s t th
2: "I theso to core
3: "Mas the thelore
...
```

## 保存生成结果

```bash
# 运行并保存输出
RUSTFLAGS="-C target-cpu=native" cargo run --release > generated_novels.txt 2>&1
```

## 下一步提升

1. **增加训练步数**: 修改 `NUM_STEPS = 10000`
2. **增大模型**: 修改 `src/model.rs` 中的 `N_EMBD`, `N_LAYER`
3. **更长的上下文**: 增加 `BLOCK_SIZE`
4. **保存/加载权重**: 需要添加权重序列化功能

## 文件位置

- 训练数据：`/Users/macbook/.openclaw/workspaces/executor/novels_training.txt`
- 模型代码：`/Users/macbook/.openclaw/workspaces/executor/rust-microgpt/`
- 生成输出：终端直接显示
