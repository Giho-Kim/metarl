"""Compositional meta-RL probe, v2: distant goals force full dynamics knowledge.

v1 let the agent reach any goal in one button press, so it only ever needed to
know which single button pointed at the goal - a partial function of m worth
about two bits.  Here goals sit 4-6 cells away on a 13-cell line, so arriving
exactly requires composing several displacements and therefore knowing most of
the permutation.  That makes m expensive to acquire and fully reusable, while g
stays thin, mirroring humanoid-dir's large shared locomotion skill under a thin
directional layer.

Reward is dense and directional (progress toward the goal plus an arrival
bonus), as in humanoid-dir, rather than the sparse hit-the-cell signal of v1.

Run: python sanity/compose2.py --models joint conditional --updates 8000
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DISPLACEMENTS = (-2, -1, 1, 2)
PERMUTATIONS = np.array(list(itertools.permutations(DISPLACEMENTS)), dtype=np.int64)
GOALS = np.array((-6, -5, -4, 4, 5, 6), dtype=np.int64)
LIMIT = 6
POSITIONS = 2 * LIMIT + 1
N_ACTIONS = len(DISPLACEMENTS)
ROUND_STEPS = 14
ROUNDS = 2
META_STEPS = ROUND_STEPS * ROUNDS
# Switch conditions are parked for now: the task is held fixed across both
# rounds so the only question is whether memory carries anything forward.
# Restore with ('keep', 'redraw_m', 'redraw_g', 'redraw_both').
CONDITIONS = ('keep',)
PROGRESS_SCALE = 1.

# Balanced 1:1 compositional split: every dynamics is paired with three train
# goals and three held-out goals; every goal has 12/12 train/held-out dynamics.
# Held-out (dynamics, goal) combinations are never trained.
ALL_TASKS = np.array(list(itertools.product(
    range(len(PERMUTATIONS)), range(len(GOALS)))), dtype=np.int64)
HELDOUT_MASK = (ALL_TASKS[:, 0] + ALL_TASKS[:, 1]) % 2 == 0
TRAIN_TASKS = ALL_TASKS[~HELDOUT_MASK]
HELDOUT_TASKS = ALL_TASKS[HELDOUT_MASK]


def sample_tasks(pool, batch, seed):
    """Sample a fixed task pool approximately uniformly and reproducibly."""
    rng = np.random.default_rng(seed)
    indices = np.arange(batch) % len(pool)
    rng.shuffle(indices)
    # Randomise which pool item gets the possible one-sample surplus.
    offset = rng.integers(0, len(pool))
    return pool[(indices + offset) % len(pool)]


class Environment:
    """Batched two-round world; memory is expected to persist across rounds."""

    def __init__(self, batch, seed, condition=None, tasks=None):
        self.rng = np.random.default_rng(seed)
        self.batch = batch
        if tasks is None:
            self.m = self.rng.integers(0, len(PERMUTATIONS), batch)
            self.g = self.rng.integers(0, len(GOALS), batch)
        else:
            pairs = np.asarray(tasks, dtype=np.int64)
            if pairs.shape != (batch, 2):
                raise ValueError('tasks must hold batch (m_index, g_index) pairs')
            self.m, self.g = pairs[:, 0].copy(), pairs[:, 1].copy()
        self.condition = (self.rng.integers(0, len(CONDITIONS), batch) if condition is None
                          else np.full(batch, condition, dtype=np.int64))
        self.x = np.zeros(batch, dtype=np.int64)

    def _resample(self, current, size, change):
        """Draw a value different from the current one wherever change is set."""
        offset = self.rng.integers(1, size, self.batch)
        return np.where(change, (current + offset) % size, current)

    def next_round(self):
        change_m = np.isin(self.condition, (1, 3))
        change_g = np.isin(self.condition, (2, 3))
        self.m = self._resample(self.m, len(PERMUTATIONS), change_m)
        self.g = self._resample(self.g, len(GOALS), change_g)
        self.x.fill(0)

    def step(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        if a.shape != (self.batch,) or not ((0 <= a) & (a < N_ACTIONS)).all():
            raise ValueError(f'actions must be a batch in [0, {N_ACTIONS})')
        goal = GOALS[self.g]
        before = np.abs(self.x - goal)
        self.x = np.clip(self.x + PERMUTATIONS[self.m, a], -LIMIT, LIMIT)
        # Dense directional term plus an arrival bonus, as in humanoid-dir.
        progress = (before - np.abs(self.x - goal)) / (2. * LIMIT)
        reward = ((self.x == goal) + PROGRESS_SCALE * progress).astype(np.float32)
        return torch.from_numpy(self.x.copy()), torch.from_numpy(reward)


def one_hot(index, size):
    return F.one_hot(index, size).float()


class Agent(nn.Module):
    """Joint and conditional RSSMs sharing every head size and the same policy."""

    def __init__(self, kind, deter=64, latent_size=16):
        super().__init__()
        if kind not in ('joint', 'conditional', 'split'):
            raise ValueError(kind)
        if kind in ('conditional', 'split') and (latent_size < 2 or latent_size % 2):
            raise ValueError('conditional needs an even latent size >= 2')
        self.kind, self.deter, self.latent_size = kind, deter, latent_size
        half = latent_size // 2
        self.half = half
        # Every kind carries one recurrent state fed by both latents and the
        # action, exactly as networks_creme.RSSM.img_step does.
        self.cell = nn.GRUCell(latent_size + N_ACTIONS, deter)
        self.initial_h = nn.Parameter(torch.zeros(deter))
        # 'split' is 'conditional' with the z_dyn -> z_task link removed, so the
        # pair isolates what that link buys and nothing else.
        self.link = half if kind == 'conditional' else 0
        if kind == 'joint':
            self.prior = nn.Linear(deter, 2 * latent_size)
            self.posterior = nn.Linear(deter + POSITIONS + 1, 2 * latent_size)
            self.state_decoder = nn.Linear(deter + latent_size, POSITIONS)
            self.reward_decoder = nn.Linear(deter + latent_size, 1)
        else:
            self.dyn_prior = nn.Linear(deter, 2 * half)
            self.dyn_posterior = nn.Linear(deter + POSITIONS, 2 * half)
            # Conditioned on the z_dyn SAMPLE, as in networks_creme.RSSM.
            self.task_prior = nn.Linear(deter + self.link, 2 * half)
            self.task_posterior = nn.Linear(deter + self.link + 1, 2 * half)
            self.state_decoder = nn.Linear(deter + half, POSITIONS)
            self.reward_decoder = nn.Linear(deter + half, 1)
        self.body = nn.Sequential(
            nn.Linear(deter + latent_size + POSITIONS + 1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    @staticmethod
    def distribution(parameters):
        mean, raw_std = parameters.chunk(2, -1)
        return mean, F.softplus(raw_std) + .1

    @staticmethod
    def sample_from(stats, sample):
        mean, std = stats
        return mean + std * torch.randn_like(mean) if sample else mean

    @staticmethod
    def gaussian_kl(q_mean, q_std, p_mean, p_std):
        return (torch.log(p_std / q_std) +
                (q_std.square() + (q_mean - p_mean).square()) /
                (2 * p_std.square()) - .5).sum(-1)

    def task_input(self, h, dyn_sample):
        """Task-stream input: the recurrent state, plus z_dyn only when linked."""
        if not self.link:
            return h
        return torch.cat([h, dyn_sample.detach()], -1)

    def initial_memory(self, batch, device):
        h = self.initial_h.tanh().unsqueeze(0).expand(batch, -1)
        if self.kind == 'joint':
            mean, std = self.distribution(self.prior(h))
            return {'h': h, 'z': mean, 'mu': mean, 'sigma': std}
        dyn = self.distribution(self.dyn_prior(h))
        task = self.distribution(self.task_prior(self.task_input(h, dyn[0])))
        return {'h': h, 'z': torch.cat([dyn[0], task[0]], -1),
                'mu': torch.cat([dyn[0], task[0]], -1),
                'sigma': torch.cat([dyn[1], task[1]], -1)}

    def heads(self, memory, x_index, step):
        # creme's get_feat: [z_dyn, z_task, deter]; actor never shapes the model.
        feature = torch.cat([memory['z'], memory['h']], -1).detach()
        clock = torch.full((len(x_index), 1), float(step) / (META_STEPS - 1),
                           device=feature.device)
        hidden = self.body(torch.cat([feature, one_hot(x_index, POSITIONS), clock], -1))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def observe(self, memory, action, next_x, reward, sample=True):
        next_obs = one_hot(next_x, POSITIONS)
        act = one_hot(action, N_ACTIONS)
        h = self.cell(torch.cat([memory['z'], act], -1), memory['h'])
        if self.kind == 'joint':
            prior = self.distribution(self.prior(h))
            post = self.distribution(self.posterior(
                torch.cat([h, next_obs, reward[:, None]], -1)))
            z = self.sample_from(post, sample)
            feature = torch.cat([h, z], -1)
            return {'h': h, 'z': z, 'mu': post[0], 'sigma': post[1],
                    'state_logits': self.state_decoder(feature),
                    'reward_logits': self.reward_decoder(feature).squeeze(-1),
                    'prior_kl': self.gaussian_kl(*[t.detach() for t in post], *prior),
                    'post_kl': self.gaussian_kl(*post, *[t.detach() for t in prior])}
        dyn_prior = self.distribution(self.dyn_prior(h))
        dyn_post = self.distribution(self.dyn_posterior(torch.cat([h, next_obs], -1)))
        d_prior = self.sample_from(dyn_prior, sample)
        d_post = self.sample_from(dyn_post, sample)
        task_prior = self.distribution(self.task_prior(self.task_input(h, d_prior)))
        task_post = self.distribution(self.task_posterior(
            torch.cat([self.task_input(h, d_post), reward[:, None]], -1)))
        q_post = self.sample_from(task_post, sample)
        pairs = ((dyn_post, dyn_prior), (task_post, task_prior))
        return {'h': h, 'z': torch.cat([d_post, q_post], -1),
                'mu': torch.cat([dyn_post[0], task_post[0]], -1),
                'sigma': torch.cat([dyn_post[1], task_post[1]], -1),
                'd': d_post, 'q': q_post,
                'state_logits': self.state_decoder(torch.cat([h, d_post], -1)),
                'reward_logits': self.reward_decoder(torch.cat([h, q_post], -1)).squeeze(-1),
                'prior_kl': sum(self.gaussian_kl(*[t.detach() for t in post], *prior)
                                for post, prior in pairs),
                'post_kl': sum(self.gaussian_kl(*post, *[t.detach() for t in prior])
                               for post, prior in pairs)}


def rollout(agent, batch, seed, condition=None, greedy=False, tasks=None, collect_aux=False):
    env = Environment(batch, seed, condition, tasks=tasks)
    device = next(agent.parameters()).device
    memory = agent.initial_memory(batch, device)
    x = torch.zeros(batch, dtype=torch.long, device=device)
    logps, values, entropies, rewards, auxiliary = [], [], [], [], []
    latents = []
    for t in range(META_STEPS):
        if t and t % ROUND_STEPS == 0:
            env.next_round()
            x = torch.zeros(batch, dtype=torch.long, device=device)
            latents.append({k: memory[k].detach() for k in ('z', 'h')})
        logits, value = agent.heads(memory, x, t)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        next_x, reward = env.step(action.detach().cpu().numpy())
        next_x = (next_x + LIMIT).to(device)
        reward = reward.to(device)
        memory = agent.observe(memory, action, next_x, reward, sample=not greedy)
        if collect_aux:
            auxiliary.append({
                'state_loss': F.cross_entropy(memory['state_logits'], next_x),
                'reward_loss': F.mse_loss(memory['reward_logits'], reward),
                'prior_kl': memory['prior_kl'], 'post_kl': memory['post_kl']})
        logps.append(dist.log_prob(action))
        values.append(value)
        entropies.append(dist.entropy())
        rewards.append(reward)
        x = next_x
    latents.append({k: memory[k].detach() for k in ('z', 'h')})
    out = {'logp': torch.stack(logps), 'value': torch.stack(values),
           'entropy': torch.stack(entropies), 'reward': torch.stack(rewards),
           'latents': latents, 'tasks': (env.m, env.g)}
    if collect_aux:
        out['aux'] = {key: torch.stack([row[key] for row in auxiliary]) for key in auxiliary[0]}
    return out


def probe_accuracy(features, labels, classes, steps=300):
    """Held-out linear read-out accuracy of a discrete factor from a latent block.

    The split matters: blocks differ in width, and a probe scored on its own
    training points rewards the wider block for memorising rather than for
    carrying the factor.
    """
    cut = len(features) // 2
    head = nn.Linear(features.shape[1], classes).to(features.device)
    opt = torch.optim.Adam(head.parameters(), lr=.05)
    for _ in range(steps):
        loss = F.cross_entropy(head(features[:cut]), labels[:cut])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (head(features[cut:]).argmax(-1) == labels[cut:]).float().mean().item()


def disentanglement(agent, seed=4242, batch=1024):
    """Decode m and g from each latent block at the end of round one."""
    agent.eval()
    with torch.no_grad():
        out = rollout(agent, batch, seed, condition=0, greedy=True)
    agent.train()
    state = out['latents'][0]
    m, g = (torch.as_tensor(v, device=state['z'].device).long() for v in out['tasks'])
    half = agent.half
    blocks = {'z_dyn': state['z'][:, :half], 'z_task': state['z'][:, half:],
              'deter': state['h']}
    scores = {}
    for name, block in blocks.items():
        scores[f'm|{name}'] = probe_accuracy(block, m, len(PERMUTATIONS))
        scores[f'g|{name}'] = probe_accuracy(block, g, len(GOALS))
    return scores


def evaluate(agent, kind, update, batch=512):
    agent.eval()
    rows = []
    for split, pool in (('train', TRAIN_TASKS), ('heldout', HELDOUT_TASKS)):
        tasks = sample_tasks(pool, batch, 777000)
        for flag, name in enumerate(CONDITIONS):
            with torch.no_grad():
                out = rollout(agent, batch, 777000, condition=flag,
                              greedy=True, tasks=tasks)
            reward = out['reward']
            rows.append(dict(model=kind, update=update, split=split, condition=name,
                             round1=reward[:ROUND_STEPS].sum(0).mean().item(),
                             round2=reward[ROUND_STEPS:].sum(0).mean().item(),
                             total=reward.sum(0).mean().item()))
    agent.train()
    return rows


def train(args):
    torch.set_num_threads(1)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    out = Path(args.out or f'results/compose2_seed{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        {**vars(args), 'device_resolved': str(device), 'permutations': len(PERMUTATIONS),
         'goals': GOALS.tolist(), 'round_steps': ROUND_STEPS, 'rounds': ROUNDS,
         'train_tasks': len(TRAIN_TASKS), 'heldout_tasks': len(HELDOUT_TASKS),
         'heldout_rule': '(m_index + g_index) % 2 == 0'}, indent=2))
    print('Device:', device, flush=True)
    eval_rows, probe_rows = [], []
    for kind in args.models:
        model_eval_rows, model_probe_rows = [], []
        torch.manual_seed(args.seed)
        agent = Agent(kind, args.deter, args.latent_size).to(device)
        opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
        curve = []
        for update in range(1, args.updates + 1):
            train_seed = args.seed * 100000 + update
            tasks = sample_tasks(TRAIN_TASKS, args.batch, train_seed)
            res = rollout(agent, args.batch, train_seed, tasks=tasks, collect_aux=True)
            reward, aux = res['reward'], res['aux']
            future = torch.flip(torch.cumsum(torch.flip(reward, (0,)), 0), (0,))
            advantage = future - res['value'].detach()
            if args.adv_norm:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            loss = (-(res['logp'] * advantage).mean()
                    + args.value_coef * F.smooth_l1_loss(res['value'], future)
                    - args.entropy_coef * res['entropy'].mean())
            state_loss, reward_loss = aux['state_loss'].mean(), aux['reward_loss'].mean()
            kl_loss = (args.dyn_scale * aux['prior_kl'].clamp_min(args.kl_free).mean()
                       + args.rep_scale * aux['post_kl'].clamp_min(args.kl_free).mean())
            loss = loss + args.state_coef * state_loss + args.reward_coef * reward_loss + kl_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 5.)
            opt.step()
            if update == 1 or update % args.log_every == 0 or update == args.updates:
                row = dict(model=kind, update=update,
                           train_total=reward.sum(0).mean().item(),
                           train_round1=reward[:ROUND_STEPS].sum(0).mean().item(),
                           train_round2=reward[ROUND_STEPS:].sum(0).mean().item(),
                           state_loss=state_loss.item(), reward_loss=reward_loss.item(),
                           kl_loss=kl_loss.item(), raw_kl=aux['prior_kl'].mean().item())
                curve.append(row)
                print(row, flush=True)
                if update % args.eval_every == 0 or update == args.updates:
                    rows = evaluate(agent, kind, update)
                    eval_rows.extend(rows)
                    model_eval_rows.extend(rows)
                    print('  EVAL', {r['split']: round(r['total'], 2) for r in rows},
                          flush=True)
                    scores = disentanglement(agent)
                    probe_row = dict(model=kind, update=update,
                                     **{k: round(v, 3) for k, v in scores.items()})
                    probe_rows.append(probe_row)
                    model_probe_rows.append(probe_row)
                    print('  PROBE', probe_row, flush=True)
        torch.save({'kind': kind, 'state_dict': agent.state_dict(),
                    'deter': args.deter, 'latent_size': args.latent_size},
                   out / f'{kind}.pt')
        for name, rows in (('curve', curve),):
            with (out / f'{kind}_{name}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
        for name, rows in (('evaluation', model_eval_rows), ('probe', model_probe_rows)):
            if rows:
                with (out / f'{kind}_{name}.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
    # Keep legacy combined files only when both models belong to this one run.
    # Separate/concurrent model runs must not overwrite each other's logs.
    if len(args.models) > 1:
        for name, rows in (('evaluation', eval_rows), ('probe', probe_rows)):
            if rows:
                with (out / f'{name}.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
    print('Saved:', out.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--models', nargs='+',
                        choices=('joint', 'conditional', 'split'),
                        default=['joint', 'conditional'])
    parser.add_argument('--deter', type=int, default=64)
    parser.add_argument('--latent-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--value-coef', type=float, default=.5)
    parser.add_argument('--entropy-coef', type=float, default=.05)
    parser.add_argument('--adv-norm', action='store_true',
                        help='standardise advantages; cuts REINFORCE variance')
    parser.add_argument('--state-coef', type=float, default=1.)
    parser.add_argument('--reward-coef', type=float, default=1.)
    parser.add_argument('--kl-free', type=float, default=1.)
    parser.add_argument('--dyn-scale', type=float, default=.5)
    parser.add_argument('--rep-scale', type=float, default=.1)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--out', default=None)
    train(parser.parse_args())
