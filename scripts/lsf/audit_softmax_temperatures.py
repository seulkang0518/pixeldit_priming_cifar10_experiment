#!/usr/bin/env python3
"""Measure temperature fields and gradients on a frozen softmax training checkpoint.

This diagnostic never steps an optimizer or writes to the training run. It uses
online weights, training dropout, 128 generated / 256 real images, and the
run's frozen source. Sampled batches are probes, not a replay of training.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--parameter-gradients', action='store_true')
    parser.add_argument('--checkpoint', type=Path,
                        help='Audit a historical checkpoint instead of the latest recovery checkpoint')
    parser.add_argument('--capture-field-outliers', action='store_true',
                        help='Preserve views whose raw field exceeds its unit-direction magnitude bound')
    args = parser.parse_args()
    os.environ['DRIFT_COMPILE'] = '0'
    source = args.run.parent / 'source'
    sys.path.insert(0, str(source.resolve()))
    import torch
    from torch.nn import functional as F
    from backbone import build_backbone
    from data import load_dataset
    from loss import _riesz_topk_softmax_loss_impl as loss_fn

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    preserved = args.output / 'checkpoint.pt'
    if not preserved.exists():
        os.link(args.checkpoint or args.run / 'checkpoint_latest.pt', preserved)
    saved = torch.load(preserved, map_location='cpu', weights_only=False, mmap=True)
    config = saved['arguments']
    taus = config['tau_list']
    assert taus == [.2, .05, .02]
    assert config['noise_levels'] == [0.] and config['field_step'] == 1.
    assert config['train_particles'] == 128 and config['target_batch_size'] == 256
    assert config['multiscale_loss'] and config['random_flip']
    assert config['coarse_sizes'] == [8, 16] and config['patch_sizes'] == [7, 14]
    assert config['crops_per_size'] == 4
    assert [config[k] for k in ('global_loss_weight', 'coarse_loss_weight', 'patch_loss_weight')] == [2, 1, 1]
    model = build_backbone(config['dataset'], config['architecture'], config['latent_dim'],
                           config['base_channels'], config['hidden_dims'], config['num_res_blocks']).to(args.device)
    model.load_state_dict(saved['model'])
    model.train()
    result = dict(run=str(args.run), checkpoint=str(preserved), update=saved['update_count'],
                  source=str(source), loss_sha256=hashlib.sha256((source/'loss.py').read_bytes()).hexdigest(),
                  normalization=config['normalize_per_temperature'],
                  no_rms=config.get('no_field_rms_normalization', False),
                  weights='online', mode='train (dropout enabled)', generated_count=128, real_count=256,
                  device=args.device, batches=[])
    del saved
    _, (train, _), _ = load_dataset('cifar10', args.data_root)
    prefix = 'riesz_topk_softmax_'
    raw_supported = 'no_field_rms_normalization' in inspect.signature(loss_fn).parameters

    def kwargs(temperatures):
        options = dict(tau_list=temperatures, normalize_per_temperature=config['normalize_per_temperature'],
                       epsilon=1e-6, return_field=True)
        if raw_supported:
            options['no_field_rms_normalization'] = config.get('no_field_rms_normalization', False)
        return options

    def rms(v):
        return float(v.square().mean().sqrt())

    def describe(vectors):
        norms = torch.stack([v.norm() for v in vectors])
        combined = sum(vectors)
        return dict(norms=norms.tolist(), norm_shares=(norms/norms.sum()).tolist(),
                    combined_norm=float(combined.norm()),
                    coherence=float(combined.norm()/norms.sum()),
                    projection_shares=[float((v*combined).sum()/combined.square().sum()) for v in vectors],
                    cosine=[[float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0, eps=1e-20))
                             for b in vectors] for a in vectors],
                    removal_relative_change=[float(v.norm()/combined.norm()) for v in vectors])

    started=time.monotonic()
    for batch in range(args.batches):
        seed=91000+batch
        torch.manual_seed(seed)
        rng=torch.Generator().manual_seed(seed)
        z=torch.randn(128, config['latent_dim'], generator=rng).to(args.device)
        target=train[torch.randint(len(train),(256,),generator=rng)].to(args.device,dtype=torch.float32).div(255)
        target=target.reshape(-1,3,32,32)
        flip=torch.rand(256,generator=rng).to(args.device)<.5
        target[flip]=target[flip].flip(-1)
        outputs,states=[],[]
        with torch.no_grad():
            for chunk in z.split(args.chunk):
                states.append((torch.get_rng_state(), torch.cuda.get_rng_state() if args.device=='cuda' else None))
                outputs.append(model(chunk))
        x=torch.cat(outputs).detach().requires_grad_(True)
        views=[('global', x.reshape(-1,3,32,32),target,.5)]
        for size in (8,16):
            views.append((f'coarse{size}',F.adaptive_avg_pool2d(x.reshape(-1,3,32,32),size),
                          F.adaptive_avg_pool2d(target,size),.125))
        for size in (7,14):
            for crop in range(4):
                top=int(torch.randint(33-size,(),generator=rng)); left=int(torch.randint(33-size,(),generator=rng))
                views.append((f'patch{size}_{crop}',x.reshape(-1,3,32,32)[:,:,top:top+size,left:left+size],
                              target[:,:,top:top+size,left:left+size],.03125))
        tau_losses=[0.,0.,0.]
        full_loss=0.
        view_rows=[]
        for name,a,b,weight in views:
            vectors=[]; row=dict(view=name,weight=weight,dimension=a[0].numel(),temperatures=[])
            for i,tau in enumerate(taus):
                loss,info=loss_fn(a.flatten(1).unsqueeze(0),b.flatten(1).unsqueeze(0),**kwargs([tau]))
                tau_losses[i]=tau_losses[i]+weight*loss.mean()
                velocity=info['frozen_velocity']
                vectors.append(velocity)
                row['temperatures'].append(dict(tau=tau,raw_rms=float(info[prefix+f'force_rms_tau_{tau:g}']),
                    applied_rms=rms(velocity),positive_ess=float(info[prefix+'positive_ess']),
                    generated_ess=float(info[prefix+'generated_ess']),
                    target_displacement_rms=rms(velocity*info['scale_inputs'])))
                bound = 2 / (tau * a[0].numel()**.5)
                row['temperatures'][-1]['unit_direction_rms_bound'] = bound
                if args.capture_field_outliers and float(info[prefix+f'force_rms_tau_{tau:g}']) > bound * 1.01:
                    destination = args.output / f'outlier_batch{batch}_{name}_tau{tau:g}.pt'
                    torch.save(dict(features=a.detach().cpu(), target=b.detach().cpu(), tau=tau,
                                    scale_inputs=info['scale_inputs'].cpu(),
                                    measured_raw_rms=info[prefix+f'force_rms_tau_{tau:g}'].cpu()), destination)
                    row['temperatures'][-1]['outlier_file'] = str(destination)
            loss,info=loss_fn(a.flatten(1).unsqueeze(0),b.flatten(1).unsqueeze(0),**kwargs(taus))
            full_loss=full_loss+weight*loss.mean()
            row.update(describe(vectors)); row['combined_field_rms']=rms(info['frozen_velocity'])
            view_rows.append(row)
        pixel_grads=[torch.autograd.grad(loss,x,retain_graph=True)[0].detach() for loss in tau_losses]
        full_grad=torch.autograd.grad(full_loss,x)[0].detach()
        relative_error=float((full_grad-sum(pixel_grads)).norm()/full_grad.norm())
        assert relative_error<1e-4, relative_error
        probe=dict(seed=seed,views=view_rows,pixel_gradient=describe(pixel_grads),
                   gradient_additivity_relative_error=relative_error,
                   combined_pixel_gradient_rms=rms(full_grad),
                   saturation_fraction=float(((x<.01)|(x>.99)).float().mean()),
                   sigmoid_derivative_mean=float((x*(1-x)).mean()))
        if args.parameter_gradients:
            parameter_grads=[]
            for pixel_grad in pixel_grads:
                model.zero_grad(set_to_none=True)
                for index,chunk in enumerate(z.split(args.chunk)):
                    torch.set_rng_state(states[index][0])
                    if args.device=='cuda': torch.cuda.set_rng_state(states[index][1])
                    y=model(chunk)
                    assert torch.allclose(y.detach(),x[index*args.chunk:(index+1)*args.chunk],atol=1e-6)
                    (y*pixel_grad[index*args.chunk:(index+1)*args.chunk]).sum().backward()
                parameter_grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach())
            probe['parameter_gradient']=describe(parameter_grads)
            norm=probe['parameter_gradient']['combined_norm']
            probe['clip_multiplier']=min(1.,config['max_grad_norm']/(norm+1e-6)) if config['max_grad_norm'] else 1.
            del parameter_grads
            model.zero_grad(set_to_none=True)
        result['batches'].append(probe)
        result['elapsed_seconds']=time.monotonic()-started
        (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(run=args.run.parent.name,update=result['update'],batch=batch,
                             parameter_gradient=probe.get('parameter_gradient'),
                             pixel_gradient=probe['pixel_gradient'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    print('Saved '+str(args.output/'audit.json'),flush=True)


if __name__=='__main__':
    main()
