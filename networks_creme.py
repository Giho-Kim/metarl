import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    def __init__(
            self,
            stoch=30,
            task_stoch=30,  # [New] Task latent dimension
            deter=200,
            hidden=200,
            layers_input=1,
            layers_output=1,
            rec_depth=1,
            shared=False,
            discrete=False,
            act="SiLU",
            norm="LayerNorm",
            mean_act="none",
            std_act="softplus",
            temp_post=True,
            min_std=0.1,
            cell="gru",
            unimix_ratio=0.01,
            initial="learned",
            num_actions=None,
            embed=None,
            device=None,
            detach_every=None
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._task_stoch = task_stoch

        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device
        self._detach_every = detach_every

        # --- Recurrent Stream Inputs ---
        # Recurrent state h_t sees action, dynamics latent (z_dyn) AND task latent (z_task).
        # Feeding z_task back into the GRU lets reward history accumulate in h_t.
        # Current reward enters only the task posterior below.
        if self._discrete:
            inp_dim = (self._stoch + self._task_stoch) * self._discrete + num_actions
        else:
            inp_dim = self._stoch + self._task_stoch + num_actions

        inp_layers = []
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        # --- Dynamics Prior Network (h_t -> z_dyn) ---
        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        # --- Dynamics Posterior Network (h_t + embed -> z_dyn) ---
        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # --- Task Prior Network (h_t + z_dyn -> z_task) ---
        # z_task history is carried by the recurrent state (z_task is fed into the GRU),
        # so the prior conditions only on h_t and the current dynamics latent.
        task_prior_layers = []
        if self._discrete:
            inp_dim = self._deter + self._stoch * self._discrete
        else:
            inp_dim = self._deter + self._stoch

        for i in range(self._layers_output):
            task_prior_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            task_prior_layers.append(norm(self._hidden, eps=1e-03))
            task_prior_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._task_prior_layers = nn.Sequential(*task_prior_layers)
        self._task_prior_layers.apply(tools.weight_init)

        # --- Task Posterior Network (h_t + sg(z_dyn) + r_t -> z_task) ---
        task_post_layers = []
        if self._discrete:
            inp_dim = self._deter + self._stoch * self._discrete + 1
        else:
            inp_dim = self._deter + self._stoch + 1

        for i in range(self._layers_output):
            task_post_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            task_post_layers.append(norm(self._hidden, eps=1e-03))
            task_post_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._task_post_layers = nn.Sequential(*task_post_layers)
        self._task_post_layers.apply(tools.weight_init)

        # --- Stat Layers ---
        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._task_prior_stat_layer = nn.Linear(self._hidden, self._task_stoch * self._discrete)
            self._task_post_stat_layer = nn.Linear(self._hidden, self._task_stoch * self._discrete)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._task_prior_stat_layer = nn.Linear(self._hidden, 2 * self._task_stoch)
            self._task_post_stat_layer = nn.Linear(self._hidden, 2 * self._task_stoch)

        self._ims_stat_layer.apply(tools.weight_init)
        self._obs_stat_layer.apply(tools.weight_init)
        self._task_prior_stat_layer.apply(tools.weight_init)
        self._task_post_stat_layer.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                task_logit=torch.zeros([batch_size, self._task_stoch, self._discrete]).to(self._device),
                task_stoch=torch.zeros([batch_size, self._task_stoch, self._discrete]).to(self._device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                task_mean=torch.zeros([batch_size, self._task_stoch]).to(self._device),
                task_std=torch.zeros([batch_size, self._task_stoch]).to(self._device),
                task_stoch=torch.zeros([batch_size, self._task_stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            # Initialize task latent using the task prior network. The task prior expects
            # the concatenation of the deterministic state and the dynamics latent.

            # Prepare the dynamics latent (z_dyn) in flattened form.
            z_dyn_input = state["stoch"]
            if self._discrete:
                # (batch, stoch, discrete) -> (batch, stoch * discrete)
                shape = list(z_dyn_input.shape[:-2]) + [self._stoch * self._discrete]
                z_dyn_input = z_dyn_input.reshape(shape)

            # Construct the input for the task prior: [deter, z_dyn_input].
            task_input = torch.cat([state["deter"], z_dyn_input], -1)

            # Pass through the task prior network to obtain the task latent statistics.
            x_task = self._task_prior_layers(task_input)
            task_stats = self._suff_stats_layer("task_prior", x_task)

            # Sample (or take the mode of) the task latent distribution to initialize
            # task_stoch. We use the mode here for determinism.
            dist = self.get_dist(task_stats, prefix="task_")
            state["task_stoch"] = dist.mode()

            # Update state with the statistics of the task prior (mean/std or logits).
            state.update(task_stats)

            return state
            #
            # if self._discrete:
            #     state["task_stoch"] = torch.zeros([batch_size, self._task_stoch, self._discrete]).to(self._device)
            #     state["task_logit"] = torch.zeros([batch_size, self._task_stoch, self._discrete]).to(self._device)
            # else:
            #     state["task_stoch"] = torch.zeros([batch_size, self._task_stoch]).to(self._device)
            #     state["task_mean"] = torch.zeros([batch_size, self._task_stoch]).to(self._device)
            #     state["task_std"] = torch.zeros([batch_size, self._task_stoch]).to(self._device)
            # return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, *, reward):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        reward = swap(reward)

        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        should_detach = lambda i: (i + 1) % self._detach_every == 0 if self._detach_every != -1 else False
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first, reward, i: self.obs_step(
                prev_state[0], prev_act, embed, is_first,
                should_detach=should_detach(i), reward=reward
            ),
            (action, embed, is_first, reward, range(len(is_first))),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        # Used for video prediction
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        # Combined features for Reward Head, Policy, Value
        stoch = state["stoch"]
        task_stoch = state["task_stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
            shape_t = list(task_stoch.shape[:-2]) + [self._task_stoch * self._discrete]
            task_stoch = task_stoch.reshape(shape_t)
        return torch.cat([stoch, task_stoch, state["deter"]], -1)

    def get_dyn_feat(self, state):
        # Dynamics-only features for Image Decoder: [z_dyn, h]
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_task_feat(self, state):
        # Task-only features for the Reward Head: [z_task, h]
        task_stoch = state["task_stoch"]
        if self._discrete:
            shape_t = list(task_stoch.shape[:-2]) + [self._task_stoch * self._discrete]
            task_stoch = task_stoch.reshape(shape_t)
        return torch.cat([task_stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None, prefix=""):
        if self._discrete:
            logit = state[f"{prefix}logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state[f"{prefix}mean"], state[f"{prefix}std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, should_detach=False, *, reward):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()

        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first  # zero out prev_action if is_first is True
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = val * (1.0 - is_first_r) + init_state[
                    key] * is_first_r  # zero out prev_state if is_first is True
        if should_detach:
            prev_state = {k: v.detach() for k, v in prev_state.items()}

        # 1. Get Priors (Dynamics & Task)
        prior = self.img_step(prev_state, prev_action, None, sample)

        # 2. Dynamics Posterior: q(z_dyn | h, o) - Does NOT see reward
        if self._shared:
            # Not typically used with this disentangled setup, but kept for compatibility
            post_dyn = self.img_step(prev_state, prev_action, embed, sample)
            dyn_stoch = post_dyn["stoch"]
            dyn_stats = {k: v for k, v in post_dyn.items() if k in ["mean", "std", "logit"]}
        else:
            if self._temp_post:
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self._obs_out_layers(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            dyn_stats = self._suff_stats_layer("obs", x)
            if sample:
                dyn_stoch = self.get_dist(dyn_stats).sample()
            else:
                dyn_stoch = self.get_dist(dyn_stats).mode()

        # 3. Task Posterior: q(z_task | h_t, sg(z_dyn), r_t).
        # Task history is carried by h_t; current observation enters via z_dyn.
        z_dyn_input = dyn_stoch.detach()  # STOP GRADIENT HERE
        if self._discrete:
            shape_dyn = list(z_dyn_input.shape[:-2]) + [self._stoch * self._discrete]
            z_dyn_input = z_dyn_input.reshape(shape_dyn)

        task_input = torch.cat([prior["deter"], z_dyn_input, reward], -1)

        x_task = self._task_post_layers(task_input)
        task_stats = self._suff_stats_layer("task_post", x_task)

        if sample:
            task_stoch = self.get_dist(task_stats, prefix="task_").sample()
        else:
            task_stoch = self.get_dist(task_stats, prefix="task_").mode()

        post = {"stoch": dyn_stoch, "deter": prior["deter"], **dyn_stats,
                "task_stoch": task_stoch, **task_stats}
        return post, prior

    # this is used for making future image
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # 1. Recurrent Step (h_t): Depends on z_dyn_{t-1}, z_task_{t-1} and a_{t-1}
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        prev_task = prev_state["task_stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
            shape_task = list(prev_task.shape[:-2]) + [self._task_stoch * self._discrete]
            prev_task = prev_task.reshape(shape_task)

        if self._shared and embed is not None:
            # Legacy shared mode support (simplified)
            x = torch.cat([prev_stoch, prev_task, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_task, prev_action], -1)

        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._inp_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.

        # 2. Dynamics Prior: p(z_dyn | h_t)
        x_dyn = self._img_out_layers(x)
        dyn_stats = self._suff_stats_layer("ims", x_dyn)
        if sample:
            dyn_stoch = self.get_dist(dyn_stats).sample()
        else:
            dyn_stoch = self.get_dist(dyn_stats).mode()

        # 3. Task Prior: p(z_task | h_t, sg(z_dyn)).
        # z_task history is carried by h_t (z_task is fed into the GRU), so the prior
        # conditions only on the recurrent state and the current dynamics latent.
        z_dyn_input = dyn_stoch.detach()
        if self._discrete:
            shape_dyn = list(z_dyn_input.shape[:-2]) + [self._stoch * self._discrete]
            z_dyn_input = z_dyn_input.reshape(shape_dyn)

        task_input = torch.cat([deter, z_dyn_input], -1)
        x_task = self._task_prior_layers(task_input)
        task_stats = self._suff_stats_layer("task_prior", x_task)

        if sample:
            task_stoch = self.get_dist(task_stats, prefix="task_").sample()
        else:
            task_stoch = self.get_dist(task_stats, prefix="task_").mode()

        prior = {"stoch": dyn_stoch, "deter": deter, **dyn_stats,
                 "task_stoch": task_stoch, **task_stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if name == "ims":
            x = self._ims_stat_layer(x)
            dim_mul = self._stoch
            prefix = ""
        elif name == "obs":
            x = self._obs_stat_layer(x)
            dim_mul = self._stoch
            prefix = ""
        elif name == "task_prior":
            x = self._task_prior_stat_layer(x)
            dim_mul = self._task_stoch
            prefix = "task_"
        elif name == "task_post":
            x = self._task_post_stat_layer(x)
            dim_mul = self._task_stoch
            prefix = "task_"
        else:
            raise NotImplementedError

        if self._discrete:
            logit = x.reshape(list(x.shape[:-1]) + [dim_mul, self._discrete])
            return {f"{prefix}logit": logit}
        else:
            mean, std = torch.split(x, [dim_mul] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {f"{prefix}mean": mean, f"{prefix}std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x, p="": self.get_dist(x, prefix=p)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # Preserve MAMBA's KL balance: rep_scale trains the posterior,
        # while dyn_scale trains the prior, for both latent streams.
        # 1. Dynamics-latent KL
        rep_loss = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )

        # 2. Task KL
        task_rep_loss = kld(
            dist(post, "task_") if self._discrete else dist(post, "task_")._dist,
            dist(sg(prior), "task_") if self._discrete else dist(sg(prior), "task_")._dist,
        )
        task_dyn_loss = kld(
            dist(sg(post), "task_") if self._discrete else dist(sg(post), "task_")._dist,
            dist(prior, "task_") if self._discrete else dist(prior, "task_")._dist,
        )

        # Sum KLs
        total_dyn_loss = dyn_loss + task_dyn_loss
        total_rep_loss = rep_loss + task_rep_loss

        rep_loss = torch.mean(torch.clip(total_rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(total_dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, total_rep_loss, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
            self,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            symlog_inputs,
            input_reward
    ):
        super(MultiEncoder, self).__init__()
        if input_reward is True:
            excluded = ("is_first", "is_last", "is_terminal")
        else:
            excluded = ("is_first", "is_last", "is_terminal", "reward")

        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shapes,
            mlp_keys,
            cnn_keys,
            act,
            norm,
            cnn_depth,
            kernel_size,
            minres,
            mlp_layers,
            mlp_units,
            cnn_sigmoid,
            image_dist,
            vector_dist,
            input_reward
    ):
        super(MultiDecoder, self).__init__()
        if input_reward is True:
            excluded = ("is_first", "is_last", "is_terminal")
        else:
            excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
            self,
            input_shape,
            depth=32,
            act="SiLU",
            norm="LayerNorm",
            kernel_size=4,
            minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2 ** i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
            self,
            feat_size,
            shape=(3, 64, 64),
            depth=32,
            act=nn.ELU,
            norm=nn.LayerNorm,
            kernel_size=4,
            minres=4,
            outscale=1.0,
            cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres ** 2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres ** 2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres ** 2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres ** 2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean


class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,
            shape,
            layers,
            units,
            act="SiLU",
            norm="LayerNorm",
            dist="normal",
            std=1.0,
            outscale=1.0,
            symlog_inputs=False,
            device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    def __init__(
            self,
            inp_dim,
            size,
            layers,
            units,
            act=nn.ELU,
            norm=nn.LayerNorm,
            dist="trunc_normal",
            init_std=0.0,
            min_std=0.1,
            max_std=1.0,
            temp=0.1,
            outscale=1.0,
            unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
