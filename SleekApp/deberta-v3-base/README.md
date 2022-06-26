---
language: en
tags: 
  - deberta
  - deberta-v3
thumbnail: https://huggingface.co/front/thumbnails/microsoft.png
license: mit
---

## DeBERTa: Decoding-enhanced BERT with Disentangled Attention

[DeBERTa](https://arxiv.org/abs/2006.03654) improves the BERT and RoBERTa models using disentangled attention and enhanced mask decoder. With those two improvements, DeBERTa out perform RoBERTa on a majority of NLU tasks with 80GB training data. 

Please check the [official repository](https://github.com/microsoft/DeBERTa) for more details and updates.

In [DeBERTa V3](https://arxiv.org/abs/2111.09543), we replaced the MLM objective with the RTD(Replaced Token Detection) objective introduced by ELECTRA for pre-training, as well as some innovations to be introduced in our upcoming paper. Compared to DeBERTa-V2,  our V3 version significantly improves the model performance in downstream tasks.  You can find a simple introduction about the model from the appendix A11 in our original [paper](https://arxiv.org/abs/2006.03654),  but we will provide more details in a separate write-up.

The DeBERTa V3 base model comes with 12 layers and a hidden size of 768. Its total parameter number is 183M since we use a vocabulary containing 128K tokens which introduce 98M parameters in the Embedding layer.  This model was trained using the 160GB data as DeBERTa V2.


#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and MNLI tasks.

| Model             | SQuAD 1.1 | SQuAD 2.0 | MNLI-m |
|-------------------|-----------|-----------|--------|
| RoBERTa-base      | 91.5/84.6 | 83.7/80.5 | 87.6   |
| XLNet-base        | -/-       | -/80.2    | 86.8   |
| DeBERTa-base  | 93.1/87.2 | 86.2/83.1 | 88.8   |
| **DeBERTa-v3-base**  | 93.9/88.4 | 88.4/85.4 | 90.6   |
| DeBERTa-v3-base+SiFT  | -/- | -/- | **91.0**   |

#### Fine-tuning with HF transformers

```bash
#!/bin/bash

cd transformers/examples/pytorch/text-classification/

pip install datasets
export TASK_NAME=mnli

output_dir="ds_results"

num_gpus=8

batch_size=8

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --max_seq_length 256 \
  --warmup_steps 500 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $output_dir \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir $output_dir

```

### Citation

If you find DeBERTa useful for your work, please cite the following paper:

``` latex
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

``` latex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```
