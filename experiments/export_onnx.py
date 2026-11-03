# -*- coding: utf-8 -*-
"""Export the fine-tuned BERT to ONNX (fp32 + int8 dynamic quantization)."""

import os

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"E:\python\sentiment-analysis"
MODEL_NAME = "bert-base-chinese"
DEVICE = torch.device("cpu")
MAX_LEN = 128

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
cfg._attn_implementation = "eager"
model = AutoModelForSequenceClassification.from_config(cfg).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(ROOT, "models", "best_model_bert.pth"),
                                 map_location=DEVICE))
model.eval()

enc = tok("样例文本", max_length=MAX_LEN, padding="max_length", truncation=True,
          return_tensors="pt")

fp32_path = os.path.join(ROOT, "models", "best_model_bert.onnx")
torch.onnx.export(
    model,
    (enc["input_ids"], enc["attention_mask"]),
    fp32_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                  "logits": {0: "batch"}},
    opset_version=14,
    dynamo=False,
)
print("fp32 onnx:", fp32_path, os.path.getsize(fp32_path), flush=True)

from onnxruntime.quantization import QuantType, quantize_dynamic

quant_path = os.path.join(ROOT, "models", "best_model_bert_quant.onnx")
quantize_dynamic(fp32_path, quant_path, weight_type=QuantType.QInt8)
print("quant onnx:", quant_path, os.path.getsize(quant_path), flush=True)
