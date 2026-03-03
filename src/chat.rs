// 对话模式 - 支持用户输入 prompt 并继续生成
use crate::mt19937_rng::PythonRandom;
use crate::tape::*;
use crate::model::*;

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

const INFERENCE_LENGTH: usize = 200;  // 从 100 增加到 200，生成更长的回复

pub fn run_chat_mode(
    tape: &mut Tape,
    rng: &mut PythonRandom,
    vocab_size: usize,
    idx_to_char: &[char],
    char_to_idx: &HashMap<char, usize>,
    state_dict: &Model,
    weights_end: usize,
    bos_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 小说专家 - 对话模式 (增强版)");
    println!("{}", "=".repeat(50));
    println!("上下文长度：64 字符 | 生成长度：200 字符");
    println!("输入文字，我会继续生成小说内容");
    println!("输入 'quit' 或 'exit' 退出");
    println!("输入 'clear' 清空对话历史");
    println!("{}", "=".repeat(50));
    
    let mut conversation_history = String::new();
    let temperature = 0.7;
    
    loop {
        print!("\n📝 你 > ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("q") {
            println!("\n👋 再见！");
            break;
        }
        
        if input.eq_ignore_ascii_case("clear") {
            conversation_history.clear();
            println!("✓ 对话历史已清空");
            continue;
        }
        
        // 添加用户输入到历史
        if !conversation_history.is_empty() {
            conversation_history.push('\n');
        }
        conversation_history.push_str(input);
        
        // 生成回复
        print!("🤖 AI > ");
        io::stdout().flush()?;
        
        let response = generate_text(
            tape,
            rng,
            vocab_size,
            idx_to_char,
            char_to_idx,
            state_dict,
            weights_end,
            bos_idx,
            &conversation_history,
            temperature,
        )?;
        
        println!("{}", response);
        
        // 将生成的内容也加入历史（用于多轮对话）
        conversation_history.push_str(&response);
        
        // 限制历史长度，避免超出 BLOCK_SIZE（现在 64 字符）
        if conversation_history.len() > BLOCK_SIZE - 20 {
            // 保留最近的对话
            conversation_history = conversation_history.chars().skip(
                conversation_history.chars().count() - (BLOCK_SIZE - 20)
            ).collect();
        }
    }
    
    Ok(())
}

fn generate_text(
    tape: &mut Tape,
    rng: &mut PythonRandom,
    vocab_size: usize,
    idx_to_char: &[char],
    char_to_idx: &HashMap<char, usize>,
    state_dict: &Model,
    weights_end: usize,
    bos_idx: usize,
    prompt: &str,
    temperature: f32,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut keys: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
    let mut values: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
    
    // 将 prompt 转换为 token
    let prompt_chars: Vec<char> = prompt.chars().collect();
    let mut generated = String::new();
    
    // 先处理 prompt 中的字符（作为上下文）
    let mut last_token_id = bos_idx;
    for (pos, &c) in prompt_chars.iter().enumerate().take(BLOCK_SIZE - 1) {
        let mut logits_indices = [0usize; MAX_VOCAB_SIZE];
        let token_id = *char_to_idx.get(&c).unwrap_or(&bos_idx);
        
        gpt(tape, &mut logits_indices, token_id, pos, &mut keys, &mut values, state_dict);
        last_token_id = token_id;
    }
    
    // 从最后一个位置开始生成新内容
    let start_pos = prompt_chars.len().min(BLOCK_SIZE - 1);
    
    for pos_id in start_pos..INFERENCE_LENGTH.min(BLOCK_SIZE) {
        let mut logits_indices = [0usize; MAX_VOCAB_SIZE];
        
        gpt(tape, &mut logits_indices, last_token_id, pos_id, &mut keys, &mut values, state_dict);
        
        // 应用温度
        let mut temp_logits = [0usize; MAX_VOCAB_SIZE];
        for i in 0..vocab_size {
            temp_logits[i] = tape.mul_const(logits_indices[i], 1.0 / temperature);
        }
        
        let mut probs = [0usize; MAX_VOCAB_SIZE];
        tape.softmax(&mut probs, &temp_logits[..vocab_size]);
        
        let mut weights = [0.0f32; MAX_VOCAB_SIZE];
        for i in 0..vocab_size {
            weights[i] = tape.data[probs[i]];
        }
        
        let token_id = rng.choices(&weights[..vocab_size], 1)[0];
        if token_id == bos_idx {
            break;
        }
        
        let c = idx_to_char[token_id];
        generated.push(c);
        last_token_id = token_id;
        
        // 遇到换行符停止
        if c == '\n' && generated.len() > 20 {
            break;
        }
    }
    
    tape.truncate(weights_end);
    Ok(generated)
}
