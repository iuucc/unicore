"""Fixed 64-sample router/tokenizer overfit diagnostic (GPU bf16 only)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig
from unicore_eeg.batching import collate_variable_channels
from unicore_eeg.synthetic import SyntheticEEGDataset

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--channels',type=int,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--steps',type=int,default=300); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA is required; refusing CPU diagnostic')
    torch.manual_seed(a.seed); np.random.seed(a.seed); device=torch.device('cuda:0')
    ds=SyntheticEEGDataset(samples=64,channels=a.channels,seed=a.seed,mode='first_experiment')
    samples=[ds[i] for i in range(64)]
    batches=[{k:v.to(device) for k,v in collate_variable_channels(samples[i:i+8]).items()} for i in range(0,64,8)]
    model=UniCOREEG(UniCOREEGConfig(in_channels=a.channels)).to(device)
    for name,param in model.named_parameters(): param.requires_grad = name.startswith('tokenizer.')
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=5e-3)
    final_grad={}
    model.train()
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True); loss=0.0
        for batch in batches:
          with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            out=model(batch['noisy'],metadata=batch['metadata'],disabled_experts=batch['disabled_experts'],route_mode='learned',enable_residual=False)
            mask=batch['label_mask']
            step_loss=torch.nn.functional.binary_cross_entropy_with_logits(out['probability_logits'].float(),batch['labels'],weight=mask,reduction='sum')/mask.sum().clamp_min(1)
          loss = loss + step_loss
          step_loss.backward()
        for name,param in model.named_parameters():
            if name.startswith('tokenizer.') and param.grad is not None: final_grad[name]=float(param.grad.detach().norm())
        opt.step()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        outs=[model(batch['noisy'],metadata=batch['metadata'],disabled_experts=batch['disabled_experts'],route_mode='learned',enable_residual=False) for batch in batches]
    labels=torch.cat([b['labels'] for b in batches]).float().cpu().numpy(); probs=torch.cat([o['probabilities'] for o in outs]).float().cpu().numpy(); logits=torch.cat([o['probability_logits'] for o in outs]).float().cpu().numpy()
    from sklearn.metrics import f1_score, roc_auc_score
    rows=[]
    for i,name in enumerate(ARTIFACT_NAMES):
        y=labels[:,i]; rows.append({'class':name,'positive':int(y.sum()),'negative':int((1-y).sum()),'f1':float(f1_score(y,probs[:,i]>=.5,zero_division=0)),'auroc':float(roc_auc_score(y,probs[:,i])) if len(np.unique(y))==2 else None,'positive_logit_mean':float(logits[y==1,i].mean()) if y.sum() else None,'negative_logit_mean':float(logits[y==0,i].mean()) if (1-y).sum() else None,'logit_margin':float(logits[y==1,i].mean()-logits[y==0,i].mean()) if y.sum() and (1-y).sum() else None})
    payload={'channels':a.channels,'samples':64,'steps':a.steps,'seed':a.seed,'loss':float(loss),'metrics':rows,'trainable_gradient_norms':final_grad,'all_head_gradient_finite_nonzero':bool(final_grad and all(np.isfinite(v) and v>0 for v in final_grad.values()))}
    a.out.mkdir(parents=True,exist_ok=True); (a.out/f'router_overfit_c{a.channels}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
