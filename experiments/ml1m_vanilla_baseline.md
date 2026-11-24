# Vanilla DiffRec Baseline — MovieLens-1M

## Dataset
- Dataset: ML-1M (`ml-1m_clean`)
- Users: 5,949
- Items: 2,810
- Training interactions: 403,277
- Validation interactions: 110,722
- Test interactions: 57,532

## Environment
- GPU: NVIDIA Tesla T4
- PyTorch: 2.14.1+cu130
- CUDA: 13.0

## Training configuration
- Learning rate: 0.001
- Weight decay: 0
- Batch size: 400
- Dimensions: [200, 600]
- Embedding size: 10
- Mean type: x0
- Diffusion steps: 40
- Noise scale: 0.005
- Noise minimum: 0.005
- Noise maximum: 0.01
- Sampling steps: 0
- Reweight: True
- Best epoch: 10
- Early stopping epoch: 29

## Best validation results
| Metric | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|
| Precision | 0.0699 | 0.0638 | 0.0542 | 0.0457 |
| Recall | 0.0647 | 0.1151 | 0.2238 | 0.3444 |
| NDCG | 0.0859 | 0.0997 | 0.1360 | 0.1787 |
| MRR | 0.1677 | 0.1800 | 0.1871 | 0.1886 |

## Best test results
| Metric | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|
| Precision | 0.0569 | 0.0508 | 0.0421 | 0.0338 |
| Recall | 0.1006 | 0.1725 | 0.3206 | 0.4664 |
| NDCG | 0.0872 | 0.1112 | 0.1607 | 0.2055 |
| MRR | 0.1449 | 0.1570 | 0.1647 | 0.1663 |

## Training command
```bash
python main.py --cuda --dataset=ml-1m_clean --data_path=./datasets/ml-1m_clean/ --lr=0.001 --weight_decay=0 --batch_size=400 --dims='[200,600]' --emb_size=10 --mean_type=x0 --steps=40 --noise_scale=0.005 --noise_min=0.005 --noise_max=0.01 --sampling_steps=0 --reweight=True --log_name=log --round=1 --gpu=0
```

This is the vanilla DiffRec baseline for the current ML-1M preprocessing and configuration.
