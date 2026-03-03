// -----------------------------
// Rust translation of Andrej Karpathy's microgpt
// https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95#file-microgpt-py
// 
// -----------------------------
pub mod mt19937_rng;
pub mod tape;
pub mod model;
pub mod chat;

use crate::mt19937_rng::PythonRandom;
use crate::tape::*;
use crate::model::*;

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::env;

const NUM_STEPS: usize = 5000;  // 从 1000 增加到 5000，更充分的训练

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Tape and Random Number Generator
    let mut rng = PythonRandom::new(42);
    let mut tape = Tape::new(
        MAX_VOCAB_SIZE * N_EMBD * 3
            + N_LAYER * (4 * N_EMBD * N_EMBD + 4 * N_EMBD * N_EMBD),
    );

    // 2. Load and Shuffle Documents
    let file = File::open("input.txt")?;
    let reader = BufReader::new(file);
    let mut docs: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty())
        .collect();

    rng.shuffle(&mut docs);
    println!("Num docs: {}", docs.len());

    // 3. Build Vocabulary
    let mut uchars = BTreeSet::new();
    for doc in &docs {
        for c in doc.chars() {
            uchars.insert(c);
        }
    }

    let bos_idx = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("Vocab size: {}", vocab_size);

    if vocab_size > MAX_VOCAB_SIZE {
        panic!("vocab_size ({}) exceeds MAX_VOCAB_SIZE", vocab_size);
    }

    let idx_to_char: Vec<char> = uchars.iter().cloned().collect();

    // Map char to index (for training/tokenization)
    // Using a HashMap or BTreeMap handles Unicode range (0 to 0x10FFFF)
    let mut char_to_idx = std::collections::HashMap::new();
    for (idx, &c) in uchars.iter().enumerate() {
        char_to_idx.insert(c, idx);
    }

    // 4. Initialize Model and Optimizer State
    let state_dict = Model::new(&mut tape, &mut rng, vocab_size, N_EMBD, BLOCK_SIZE, N_LAYER);
    let weights_end = tape.len();
    println!("Num params: {}", weights_end);

    let learning_rate = 0.01;
    let beta1: f32 = 0.85;
    let beta2: f32 = 0.99;
    let eps_adam = 1e-8;

    let mut m = vec![0.0; weights_end];
    let mut v = vec![0.0; weights_end];

    // 5. Training Loop
    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        
        // Tokenizer: BOS + doc characters + BOS
        let mut tokens = Vec::with_capacity(BLOCK_SIZE + 2);
        tokens.push(bos_idx);
        for ch in doc.chars() {
            // .get() returns an Option, protecting against chars not in training set
            if let Some(&idx) = char_to_idx.get(&ch) {
                tokens.push(idx);
            }
        }
        tokens.push(bos_idx);

        let n = (tokens.len() - 1).min(BLOCK_SIZE);

        // Forward Pass
        let mut keys: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut values: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
        let mut losses = [0usize; BLOCK_SIZE];

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            
            let mut logits = [0usize; MAX_VOCAB_SIZE];
            gpt(&mut tape, &mut logits, token_id, pos_id, &mut keys, &mut values, &state_dict);
            
            let mut probs = [0usize; MAX_VOCAB_SIZE];
            tape.softmax(&mut probs, &logits[..vocab_size]);
            
            losses[pos_id] = tape.inv_log(probs[target_id]);
        }

        let mut total_losses = losses[0];
        for i in 1..n {
            total_losses = tape.add(total_losses, losses[i]);
        }
        let loss_idx = tape.mul_const(total_losses, 1.0 / (n as f32));

        // Backward Pass
        tape.backward(loss_idx);

        // Adam Optimizer
        let lr_t = learning_rate * (1.0 - (step as f32 / NUM_STEPS as f32));
        let beta1_pow = beta1.powi((step + 1) as i32);
        let beta2_pow = beta2.powi((step + 1) as i32);

        for i in 0..weights_end {
            let p_grad = tape.grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * p_grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p_grad * p_grad;
            
            let m_hat = m[i] / (1.0 - beta1_pow);
            let v_hat = v[i] / (1.0 - beta2_pow);
            
            tape.data[i] -= lr_t * m_hat / (v_hat.sqrt() + eps_adam);
        }

        if (step + 1) % 10 == 0 {
            print!(
                "Step {} / {} | loss {:.4} | beta1_pow {:.6} | beta2_pow {:.6}\r",
                step + 1, NUM_STEPS, tape.data[loss_idx], beta1_pow, beta1_pow
            );
            use std::io::Write;
            std::io::stdout().flush()?;
        }

        // Reset tape for next step, keeping weights
        tape.truncate(weights_end);
    }

    // 6. Check for chat mode
    let args: Vec<String> = env::args().collect();
    let chat_mode = args.iter().any(|arg| arg == "--chat" || arg == "-c");
    
    if chat_mode {
        // Run interactive chat mode
        chat::run_chat_mode(
            &mut tape,
            &mut rng,
            vocab_size,
            &idx_to_char,
            &char_to_idx,
            &state_dict,
            weights_end,
            bos_idx,
        )?;
    } else {
        // Run standard inference (generate 20 samples)
        println!("\n\n-------- inference --------\n");
        let temperature = 0.5;

        for sample_idx in 0..20 {
            let mut keys: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
            let mut values: KVCache = [[[0; N_EMBD]; BLOCK_SIZE]; N_LAYER];
            let mut token_id = bos_idx;
            let mut samples = String::new();

            for pos_id in 0..BLOCK_SIZE {
                let mut logits_indices = [0usize; MAX_VOCAB_SIZE];
                gpt(&mut tape, &mut logits_indices, token_id, pos_id, &mut keys, &mut values, &state_dict);
                
                // Apply temperature
                let mut temp_logits = [0usize; MAX_VOCAB_SIZE];
                for i in 0..vocab_size {
                    temp_logits[i] = tape.mul_const(logits_indices[i], 1.0 / temperature);
                }

                let mut probs = [0usize; MAX_VOCAB_SIZE];
                tape.softmax(&mut probs, &temp_logits[..vocab_size]);

                // Convert tape indices to actual weight values for choices
                let mut weights = [0.0f32; MAX_VOCAB_SIZE];
                for i in 0..vocab_size {
                    weights[i] = tape.data[probs[i]];
                }

                token_id = rng.choices(&weights[..vocab_size], 1)[0];
                if token_id == bos_idx {
                    break;
                }
                samples.push(idx_to_char[token_id]);
            }
            
            println!("{}: {}", sample_idx, samples);
            tape.truncate(weights_end);
        }
    }

    Ok(())
}
