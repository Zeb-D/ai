//! 用 Rust 加载并推理 [`export_onnx_from_hf.py`](../export_onnx_from_hf.py) 导出的同一份 ONNX 模型。
//!
//! 目的：证明"ONNX 是跨语言交付物"——**同一组 `.onnx` 文件，不经过 Python 也能跑出相同译文**。
//!
//! 本程序只负责"图推理 + 贪心解码"；分词仍由 Python 侧(`run_rust_parity.py`)预处理后
//! 以纯文本 id 列表传入，从而把变量隔离到"ONNX 图本身能否在 Rust 中复现"这一点上。
//!
//! 用法：
//! ```text
//! ORT_DYLIB_PATH=/path/libonnxruntime.1.30.0.dylib \
//!   ./onnx_rust_demo <model_dir> <ids.txt> <mask.txt> <out_ids.txt> <start_id> <eos_id> <max_new>
//! ```

use std::env;
use std::error::Error;
use std::fs;

use ort::session::Session;
use ort::value::Tensor;

struct Args {
    model_dir: String,
    ids_file: String,
    mask_file: String,
    out_file: String,
    start_id: i64,
    eos_id: i64,
    max_new: usize,
}

fn parse_args() -> Result<Args, String> {
    let a: Vec<String> = env::args().collect();
    if a.len() != 8 {
        return Err(format!(
            "用法: {} <model_dir> <ids.txt> <mask.txt> <out_ids.txt> <start_id> <eos_id> <max_new>",
            a.first().map(String::as_str).unwrap_or("onnx_rust_demo")
        ));
    }
    Ok(Args {
        model_dir: a[1].clone(),
        ids_file: a[2].clone(),
        mask_file: a[3].clone(),
        out_file: a[4].clone(),
        start_id: a[5].parse().map_err(|e| format!("start_id: {e}"))?,
        eos_id: a[6].parse().map_err(|e| format!("eos_id: {e}"))?,
        max_new: a[7].parse().map_err(|e| format!("max_new: {e}"))?,
    })
}

/// 读取空格/换行分隔的整数序列
fn read_ints(path: &str) -> Result<Vec<i64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        out.push(tok.parse::<i64>()?);
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn Error> { e.into() })?;

    // load-dynamic：从 ORT_DYLIB_PATH 指定（或系统默认路径）加载 libonnxruntime
    // 注意 ort 2.0.0-rc.10 的 init_from()/init() 直接返回 EnvironmentBuilder（非 Result）。
    if let Ok(dylib) = env::var("ORT_DYLIB_PATH") {
        ort::init_from(&dylib).commit()?;
    } else {
        ort::init().commit()?;
    }

    let enc_path = format!("{}/encoder_model.onnx", args.model_dir);
    let dec_path = format!("{}/decoder_model.onnx", args.model_dir);

    let mut enc = Session::builder()?.commit_from_file(&enc_path)?;
    let mut dec = Session::builder()?.commit_from_file(&dec_path)?;

    let input_ids = read_ints(&args.ids_file)?;
    let mask = read_ints(&args.mask_file)?;
    let s = input_ids.len() as i64;

    // ---------- 1) encoder 只跑一次 ----------
    let hidden_vec: Vec<f32>;
    let hidden_shape: Vec<i64>;
    {
        let ids_t = Tensor::from_array((vec![1i64, s], input_ids.clone()))?;
        let mask_t = Tensor::from_array((vec![1i64, s], mask.clone()))?;
        let out = enc.run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t
        ])?;
        let (shape, data) = out["last_hidden_state"].try_extract_tensor::<f32>()?;
        hidden_shape = shape.iter().map(|&x| x as i64).collect();
        hidden_vec = data.to_vec(); // 拷贝：outputs 借用了 session
    }
    eprintln!(
        "  [rust] encoder ok: hidden = {:?} ({} floats)",
        hidden_shape,
        hidden_vec.len()
    );

    // ---------- 2) 贪心解码（无 KV Cache，每步喂完整序列） ----------
    let mut dec_ids: Vec<i64> = vec![args.start_id];
    for _step in 0..args.max_new {
        let t = dec_ids.len() as i64;
        let h_t = Tensor::from_array((hidden_shape.clone(), hidden_vec.clone()))?;
        let m_t = Tensor::from_array((vec![1i64, s], mask.clone()))?;
        let d_t = Tensor::from_array((vec![1i64, t], dec_ids.clone()))?;

        let out = dec.run(ort::inputs![
            "encoder_hidden_states" => h_t,
            "encoder_attention_mask" => m_t,
            "decoder_input_ids" => d_t
        ])?;
        let (shape, logits) = out["logits"].try_extract_tensor::<f32>()?;
        let vocab = *shape.last().ok_or("logits 形状为空")? as usize;
        let base = logits.len() - vocab; // 只看最后一个时间步

        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for v in 0..vocab {
            if logits[base + v] > best_v {
                best_v = logits[base + v];
                best = v;
            }
        }
        let tok = best as i64;
        dec_ids.push(tok);
        if tok == args.eos_id {
            break;
        }
    }

    let generated = &dec_ids[1..]; // 去掉 decoder_start_token
    let joined = generated
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&args.out_file, joined)?;
    println!("  [rust] generated {} tokens -> {}", generated.len(), args.out_file);
    Ok(())
}
