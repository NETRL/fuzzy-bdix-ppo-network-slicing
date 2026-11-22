from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_gain=0.01, act="tanh"):
        self.sizes = sizes
        self.act = act
        self.params = []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W = rng.standard_normal((fan_in, fan_out))

            u, _, vt = np.linalg.svd(W, full_matrices=False)
            W = (u if u.shape == (fan_in, fan_out) else vt)
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
            W = W * gain
            self.params.append(W)
            self.params.append(np.zeros(fan_out))

    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float64)
        cache = [x]
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            x = x @ W + b
            if i < n_layers - 1:
                x = np.tanh(x) if self.act == "tanh" else np.maximum(x, 0.0)
            cache.append(x)
        return x, cache

    def backward(self, cache, dout):

        grads = [None] * len(self.params)
        n_layers = len(self.params) // 2
        d = dout
        for i in reversed(range(n_layers)):
            W = self.params[2 * i]
            a_prev = cache[i]
            grads[2 * i] = a_prev.T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = d @ W.T
                a = cache[i]
                d = d * (1.0 - a ** 2) if self.act == "tanh" else d * (a > 0)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5, max_norm=0.5):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.max_norm = lr, betas[0], betas[1], eps, max_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        if self.max_norm is not None:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if total > self.max_norm:
                grads = [g * (self.max_norm / (total + 1e-12)) for g in grads]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class RMSProp:
    def __init__(self, params, lr=2.5e-4, alpha=0.95, eps=0.01, max_norm=None):
        self.params, self.lr, self.alpha, self.eps, self.max_norm = params, lr, alpha, eps, max_norm
        self.sq = [np.zeros_like(p) for p in params]
    def step(self, grads):
        if self.max_norm is not None:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
            if total > self.max_norm: grads = [g * (self.max_norm / (total + 1e-12)) for g in grads]
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.sq[i] = self.alpha * self.sq[i] + (1 - self.alpha) * g * g
            p -= self.lr * g / (np.sqrt(self.sq[i]) + self.eps)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class PPOAgent:
    def __init__(self, obs_dim, n_actions=4, n_heads=1, seed=0, lr=3e-4, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, epochs=10, minibatch=128, critic_obs_dim=None, hidden=64, lr_v=None, max_norm=0.5,
                 huber_delta=None, value_norm=False, kl_target=None):
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.n_heads = n_actions, n_heads
        self.actor = MLP([obs_dim, hidden, hidden, n_actions * n_heads], self.rng, out_gain=0.01)
        self.critic = MLP([critic_obs_dim or obs_dim, hidden, hidden, 1], self.rng, out_gain=1.0)
        self.opt_a = Adam(self.actor.params, lr=lr, max_norm=max_norm)
        self.opt_c = Adam(self.critic.params, lr=lr_v or lr, max_norm=max_norm)
        self.huber_delta, self.value_norm, self.kl_target = huber_delta, value_norm, kl_target
        self.ret_mean, self.ret_std = 0.0, 1.0
        self.clip, self.ent_coef, self.vf_coef, self.epochs, self.minibatch = clip, ent_coef, vf_coef, epochs, minibatch


    def policy(self, obs):
        logits, _ = self.actor.forward(obs)
        return softmax(logits.reshape(-1, self.n_heads, self.n_actions))

    def act(self, obs, deterministic=False):
        p = self.policy(obs)[0]
        if deterministic:
            a = p.argmax(axis=-1)
        else:
            a = np.array([self.rng.choice(self.n_actions, p=p[h]) for h in range(self.n_heads)])
        logp = float(np.log(p[np.arange(self.n_heads), a] + 1e-12).sum())
        return (int(a[0]) if self.n_heads == 1 else a), logp

    def value(self, cobs):
        v, _ = self.critic.forward(cobs)
        return float(v[0, 0]) * self.ret_std + self.ret_mean


    @staticmethod
    def gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
        T = len(rewards)
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            nv = last_value if t == T - 1 else values[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + gamma * nv * nonterm - values[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        return adv, adv + values

    def update(self, obs, cobs, actions, old_logp, adv, returns):
        n = len(obs)
        actions = np.asarray(actions).reshape(n, self.n_heads)
        stats = dict(pl=0.0, vl=0.0, ent=0.0, kl=0.0, n=0)
        if self.value_norm:
            self.ret_mean = 0.99 * self.ret_mean + 0.01 * float(returns.mean()); self.ret_std = 0.99 * self.ret_std + 0.01 * float(returns.std() + 1e-6)
        tgt = (returns - self.ret_mean) / self.ret_std
        for _ in range(self.epochs):
            kl_epoch = []
            perm = self.rng.permutation(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                b = len(idx)
                A = adv[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)

                logits, cache = self.actor.forward(obs[idx])
                p = softmax(logits.reshape(b, self.n_heads, self.n_actions))
                onehot = np.zeros_like(p)
                onehot[np.arange(b)[:, None], np.arange(self.n_heads)[None, :], actions[idx]] = 1.0
                logp = np.log((p * onehot).sum(-1) + 1e-12).sum(-1)
                ratio = np.exp(logp - old_logp[idx])
                clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip)
                use_unclipped = ((A > 0) & (ratio < 1 + self.clip)) | ((A < 0) & (ratio > 1 - self.clip))
                coef = np.where(use_unclipped, ratio * A, 0.0)
                ent = -(p * np.log(p + 1e-12)).sum(-1)

                dlogits = -(coef[:, None, None] * (onehot - p)) / b
                dent = -(p * (np.log(p + 1e-12) + ent[:, :, None]))
                dlogits += -self.ent_coef * dent / b
                grads = self.actor.backward(cache, dlogits.reshape(b, -1))
                self.opt_a.step(grads)

                v, vcache = self.critic.forward(cobs[idx])
                err = v - tgt[idx, None]
                dv = self.vf_coef * (np.clip(err, -self.huber_delta, self.huber_delta) if self.huber_delta else err) / b
                self.opt_c.step(self.critic.backward(vcache, dv))
                stats["pl"] += float(-np.minimum(ratio * A, clipped * A).mean())
                stats["vl"] += float(0.5 * ((v[:, 0] - tgt[idx]) ** 2).mean())
                kl_epoch.append(float((old_logp[idx] - logp).mean()))
                stats["ent"] += float(ent.sum(-1).mean())
                stats["kl"] += float((old_logp[idx] - logp).mean())
                stats["n"] += 1
            if self.kl_target is not None and np.mean(kl_epoch) > 1.5 * self.kl_target:
                break
        return {k: v / max(stats["n"], 1) for k, v in stats.items() if k != "n"}

    def n_params(self):
        return self.actor.n_params() + self.critic.n_params()


class DQNAgent:
    def __init__(self, obs_dim, n_actions, seed=0, lr=2.5e-4, gamma=0.99, buffer=100_000, batch=32,
                 target_update=8000, eps_start=1.0, eps_end=0.1, eps_frac=0.1, train_freq=4, learning_starts=5000, eval_eps=0.05):
        self.rng = np.random.default_rng(seed)
        self.q = MLP([obs_dim, 64, 64, n_actions], self.rng, out_gain=1.0)
        self.q_target = MLP([obs_dim, 64, 64, n_actions], self.rng, out_gain=1.0)
        self.q_target.params = [p.copy() for p in self.q.params]
        self.opt = RMSProp(self.q.params, lr=lr, alpha=0.95, eps=0.01)
        self.eval_eps = eval_eps
        self.n_actions, self.gamma, self.batch, self.target_update = n_actions, gamma, batch, target_update
        self.eps_start, self.eps_end, self.eps_frac, self.train_freq, self.learning_starts = eps_start, eps_end, eps_frac, train_freq, learning_starts
        self.buf_size = buffer
        self.obs_b = np.zeros((buffer, obs_dim)); self.nobs_b = np.zeros((buffer, obs_dim))
        self.a_b = np.zeros(buffer, dtype=int); self.r_b = np.zeros(buffer); self.d_b = np.zeros(buffer)
        self.ptr, self.size, self.steps = 0, 0, 0

    def epsilon(self, total):
        f = min(1.0, self.steps / max(1, self.eps_frac * total))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, obs, total=1, deterministic=False):
        eps = self.eval_eps if deterministic else self.epsilon(total)
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q, _ = self.q.forward(obs)
        return int(q[0].argmax())

    def store(self, o, a, r, no, d):
        i = self.ptr
        self.obs_b[i], self.a_b[i], self.r_b[i], self.nobs_b[i], self.d_b[i] = o, a, r, no, d
        self.ptr = (self.ptr + 1) % self.buf_size
        self.size = min(self.size + 1, self.buf_size)
        self.steps += 1
        if self.steps >= self.learning_starts and self.steps % self.train_freq == 0:
            self.train_step()
        if self.steps % self.target_update == 0:
            self.q_target.params = [p.copy() for p in self.q.params]

    def train_step(self):
        idx = self.rng.integers(0, self.size, self.batch)
        q, cache = self.q.forward(self.obs_b[idx])
        qn, _ = self.q_target.forward(self.nobs_b[idx])
        target = self.r_b[idx] + self.gamma * (1 - self.d_b[idx]) * qn.max(axis=1)
        qa = q[np.arange(self.batch), self.a_b[idx]]
        err = qa - target
        dq = np.zeros_like(q)

        dq[np.arange(self.batch), self.a_b[idx]] = np.clip(err, -1.0, 1.0) / self.batch
        self.opt.step(self.q.backward(cache, dq))

    def n_params(self):
        return self.q.n_params()
