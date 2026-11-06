"""Paired AB/BA whole-layer graph comparison on the same fresh inputs."""
import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from measure import ROOT, load, make_data, copy_data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('control');p.add_argument('candidate')
    p.add_argument('--cases',default='')
    p.add_argument('--fresh',type=int,default=3)
    p.add_argument('--reps',type=int,default=20)
    p.add_argument('--pad-seed',type=int,default=11)
    p.add_argument('--prepare-order',choices=['control_first','candidate_first'],default='control_first',
                   help='Reverse allocation/preparation order without changing control/candidate timing labels')
    p.add_argument('--weights-seed',type=int,default=5353)
    p.add_argument('--router-seed',type=int)
    p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--attention-donor',help='Development-only: replace candidate attention with this separate module')
    p.add_argument('--stage',choices=['layer','attention'],default='layer',
                   help='Attention-only timing uses identical BF16 QKV in both arms; full output checks run outside timing')
    p.add_argument('--json',required=True)
    args=p.parse_args()
    assert args.reps >= 2 and args.reps % 2 == 0
    import torch
    import reference as R
    torch.set_grad_enabled(False)
    weights=R.generate_weights(args.weights_seed,router_seed=args.router_seed)
    out={'args':vars(args),'status':'running','sources':{},'cases':[]}
    out['harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.attention_donor or args.stage != 'layer':
        out['development_only']=True
    def save():
        Path(args.json).parent.mkdir(parents=True,exist_ok=True)
        Path(args.json).write_text(json.dumps(out,indent=2))
    arms={}; pads=[]; rng=random.Random(args.pad_seed)
    preparation=[('control',args.control),('candidate',args.candidate)]
    if args.prepare_order == 'candidate_first':
        preparation.reverse()
    for label,path in preparation:
        pad_bytes=512*rng.randrange(65536)
        pads.append(torch.empty(pad_bytes,dtype=torch.uint8,device='cuda'))
        mod=load(Path(path).resolve(),'paired_'+label)
        start=time.monotonic()
        donor_source=None
        if label=='candidate' and args.attention_donor:
            donor_path=Path(args.attention_donor).resolve()
            donor=load(donor_path,'development_attention_donor')
            def attention(data,state,qkv):
                supplied=dict(data,qkv=qkv)
                if 'positions' not in supplied:
                    supplied['positions']=(data['seq_lens'].long()[:,None]+torch.arange(
                        data['config']['q_len'],device=qkv.device)).reshape(-1)
                return donor.kernel_fn(supplied).view(-1,mod.Q_DIM)
            mod._decode_attention7=attention
            donor_source={'path':str(donor_path),'sha256':hashlib.sha256(donor_path.read_bytes()).hexdigest()}
        state=mod.prepare(weights);torch.cuda.synchronize()
        arms[label]=(mod,state)
        out['sources'][label]={'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'pad_bytes':pad_bytes,'prepare_s':time.monotonic()-start}
        if donor_source: out['sources'][label]['attention_donor']=donor_source
        print('PREPARE',label,round(time.monotonic()-start,2),flush=True)
    cases=json.loads((ROOT/'runs/problem_start.json').read_text())['benchmarks']
    chosen=[int(x) for x in args.cases.split(',') if x] or list(range(len(cases)))
    flush=torch.zeros(64*1024*1024,dtype=torch.int32,device='cuda');sink=torch.empty((),dtype=torch.int64,device='cuda')
    save()
    for ci in chosen:
        params=cases[ci];row={'index':ci,'params':params,'draws':[]};out['cases'].append(row)
        for fresh in range(args.fresh):
            data=make_data(R,params,params['seed']+args.seed_offset+fresh*104729,weights)
            sequence_lengths=data['seq_lens'].cpu().tolist()
            input_metadata=dict(sequence_lengths=sequence_lengths,
                                total_context=sum(sequence_lengths),
                                min_context=min(sequence_lengths),max_context=max(sequence_lengths))
            if args.stage == 'attention':
                data['qkv_probe']=torch.mm(data['attn_input'],weights['qkv_gate'].t())
            graphs={};check={};routing={}
            for label,(mod,state) in arms.items():
                d={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
                def run_stage():
                    if args.stage == 'attention':
                        return mod._decode_attention7(d,state,d['qkv_probe'])
                    return mod.kernel_fn(d,state)
                for _ in range(3): copy_data(d,data);run_stage()
                torch.cuda.synchronize();copy_data(d,data)
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): result=run_stage()
                g.replay();torch.cuda.synchronize()
                if args.stage == 'attention':
                    # Older backends kept an unused scratch argument; the lean
                    # integration removed it. This changes only untimed checks.
                    if len(inspect.signature(mod._after_attention).parameters)==4:
                        result=mod._after_attention(d,state,result,d['config']['tokens'])
                    else:
                        result=mod._after_attention(d,state,result,None,d['config']['tokens'])
                    torch.cuda.synchronize()
                ok,msg=R.check_implementation(data,R.collect(d,result),weights)
                check[label]={'passed':bool(ok),'message':msg}
                local_start=int(data['local_start'].item())
                local=(result[2].long()-local_start+256)%256
                counts=torch.bincount(local[local<32],minlength=32).cpu().tolist()
                routing[label]={'local_start':local_start,'routes':sum(counts),
                                'distinct':sum(v>0 for v in counts),'counts':counts}
                if not ok:
                    out['status']='failed_correctness';row['failure']={label:check[label]};save();raise RuntimeError(msg)
                graphs[label]=(d,g,result)
            times={k:[] for k in arms}
            for rep in range(args.reps):
                for label in (('control','candidate') if rep%2==0 else ('candidate','control')):
                    d,g,_=graphs[label];copy_data(d,data);torch.sum(flush,dim=0,out=sink)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();g.replay();b.record();b.synchronize();times[label].append(a.elapsed_time(b)*1000)
            deltas=[(times['control'][i]+times['control'][i+1]-times['candidate'][i]-times['candidate'][i+1])/2 for i in range(0,args.reps,2)]
            draw={'fresh':fresh,'checks':check,'routing':routing,'raw_us':times,'saving_us':statistics.median(deltas),'positive_pairs':sum(x>0 for x in deltas),'pairs':len(deltas),'paired_deltas_us':deltas}
            draw['input_metadata']=input_metadata
            row['draws'].append(draw)
            print(f"CASE {ci} fresh={fresh} A={statistics.median(times['control']):.3f} B={statistics.median(times['candidate']):.3f} saving={draw['saving_us']:+.3f} us positive={draw['positive_pairs']}/{draw['pairs']}",flush=True)
            save();del graphs,d,g,result,data
            gc.collect();torch.cuda.empty_cache()
        row['median_saving_us']=statistics.median(x['saving_us'] for x in row['draws'])
        save()
    out['status']='ok';save();print('DONE',flush=True)


if __name__=='__main__': main()
