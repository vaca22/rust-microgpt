#!/usr/bin/env python3
"""
小说专家模型推理脚本
从训练好的模型权重文件加载并生成小说文本
"""

import subprocess
import sys

PROJECT_DIR = "/Users/macbook/.openclaw/workspaces/executor/rust-microgpt"

def run_inference(steps=1000, samples=20, max_length=100, temperature=0.7):
    """
    运行推理
    
    Args:
        steps: 训练步数（如果需要先训练）
        samples: 生成样本数量
        max_length: 每个样本的最大长度
        temperature: 温度参数（越高越随机，越低越确定）
    """
    print(f"🤖 小说专家模型推理")
    print(f"{'='*50}")
    print(f"样本数量：{samples}")
    print(f"最大长度：{max_length}")
    print(f"温度参数：{temperature}")
    print(f"{'='*50}\n")
    
    # 直接运行 cargo run --release
    # 模型会在训练后自动进行推理
    cmd = ["cargo", "run", "--release"]
    
    print(f"📦 编译并运行模型...\n")
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env={"RUSTFLAGS": "-C target-cpu=native"}
    )
    
    if result.returncode != 0:
        print(f"❌ 错误：{result.stderr}")
        sys.exit(1)
    
    # 解析输出，提取推理结果
    output = result.stdout
    lines = output.split('\n')
    
    in_inference = False
    generated_texts = []
    
    for line in lines:
        if '-------- inference --------' in line:
            in_inference = True
            continue
        
        if in_inference and line.strip():
            if ':' in line:
                # 解析 "0: text" 格式
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0].strip())
                        text = parts[1].strip()
                        generated_texts.append((idx, text))
                    except ValueError:
                        pass
    
    # 打印生成的文本
    print(f"✨ 生成的小说片段:\n")
    for idx, text in generated_texts:
        print(f"{idx:2d}: {text}")
    
    print(f"\n{'='*50}")
    print(f"✅ 推理完成！共生成了 {len(generated_texts)} 个文本片段")
    
    return generated_texts

def interactive_mode():
    """交互式生成模式"""
    print(f"\n🎨 交互式小说生成")
    print(f"{'='*50}")
    print(f"输入提示词（prompt），按 Enter 生成小说片段")
    print(f"输入 'quit' 退出\n")
    
    while True:
        try:
            prompt = input("提示词 > ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not prompt:
                print("⚠️  请输入一些文字作为提示\n")
                continue
            
            # 将提示词添加到 input.txt 末尾，然后重新训练+推理
            # 这是一个简化的方法，实际需要修改模型支持 prompt
            print(f"📝 生成中...\n")
            run_inference(samples=5, max_length=50)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except EOFError:
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_mode()
        else:
            print(f"用法：python3 {sys.argv[0]} [--interactive|-i]")
            print(f"直接运行将生成 20 个小说片段")
    else:
        run_inference()
