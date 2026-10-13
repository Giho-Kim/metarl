"""Re-score saved compose checkpoints under one matched protocol.

The training runs logged evaluations before two fixes landed (a held-out split
in the latent probe, and a single shared seed across the four switch
conditions), so their numbers are not comparable across conditions.  This
re-measures everything from the checkpoints instead.

Run from sanity/: python final_eval.py compose2 results/compose2_seed0
"""
import argparse
import importlib
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', choices=('compose', 'compose2'))
    parser.add_argument('results')
    parser.add_argument('--batch', type=int, default=2048)
    args = parser.parse_args()
    env = importlib.import_module(args.module)
    root = Path(args.results)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'=== {args.module}  ({root})')
    print(f'    chance: m={1/len(env.PERMUTATIONS):.3f}  g={1/len(env.GOALS):.3f}'
          f'   horizon={env.META_STEPS} steps ({env.ROUNDS} rounds)')
    table = {}
    for path in sorted(root.glob('*.pt')):
        kind = path.stem
        ck = torch.load(path, map_location=device)
        agent = env.Agent(ck['kind'], ck['deter'], ck['latent_size']).to(device)
        agent.load_state_dict(ck['state_dict'])
        rows = env.evaluate(agent, kind, 'final', batch=args.batch)
        table[kind] = {(r.get('split', 'all'), r['condition']): r for r in rows}
        print(f'\n-- {kind}')
        for r in rows:
            label = f"{r.get('split', 'all')}/{r['condition']}"
            print(f"   {label:20s} total={r['total']:6.2f} "
                  f"round1={r['round1']:5.2f} round2={r['round2']:5.2f}")
        redraw_g = table[kind].get(('heldout', 'redraw_g'))
        redraw_m = table[kind].get(('heldout', 'redraw_m'))
        if redraw_g and redraw_m:
            reuse = redraw_g['round2'] - redraw_m['round2']
            print(f"   reuse gain (redraw_g - redraw_m, round2) = {reuse:+.2f}")
        scores = env.disentanglement(agent, batch=args.batch)
        print('   probe ' + '  '.join(f'{k}={v:.3f}' for k, v in scores.items()))
        if ck['kind'] == 'conditional':
            sep = ((scores['m|z_dyn'] - scores['m|z_task'])
                   + (scores['g|z_task'] - scores['g|z_dyn'])) / 2
            print(f'   separation score = {sep:+.3f}  '
                  '(mean of m favouring z_dyn and g favouring z_task; 0 = no split)')
    if len(table) == 2:
        print('\n-- conditional minus joint')
        for key in table['joint']:
            delta = table['conditional'][key]['total'] - table['joint'][key]['total']
            print(f'   {key[0]}/{key[1]:12s} {delta:+6.2f}')


if __name__ == '__main__':
    main()
