# Natural Language Inference




## Experiments


### Results

Models

- ESIM: Enhanced Sequential Inference Model ([Paper](https://arxiv.org/pdf/1609.06038.pdf))

#### SNLI

| Model | BRIEF | Accu_Dev | Accu_Test |
| - | - | - | - |
| ESIM | LSTM: Single layer, 600D | 88.09 | 87.28 |

```
Default
```

#### MedNLI



| Model | BRIEF | Accu_Dev | Accu_Test |
| - | - | - | - |
| ESIM | LSTM: Single layer, 600D | 80.07 | 79.54 |


```
python3 run.py --data_dir ../mednli_data --batch_size 20 --checkpoint 200 --file_pretrain_vocab vocab_w2v_bioasq_mimic_300d --file_pretrain_vec w2v_bioasq_mimic_300d.npy
```
