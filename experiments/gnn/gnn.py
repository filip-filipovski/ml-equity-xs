"""P13 graph models in plain torch (no PyTorch Geometric): MLP, relational GraphSAGE-mean, relational GAT, temporal R-SAGE.

Edge convention (all models): ``edges`` is a list with one entry per relation, each ``(src, dst, w)`` of 1-D tensors holding
POSITIONS into the present node set (``src``/``dst`` int64 in [0, N), ``w`` float). A message flows src -> dst; "incoming
neighbours of i" = {src : dst == i}. A relation with no edges (or a node without incoming edges in a relation) contributes 0.
Every model maps ``x [N, in_dim]`` -> ``[N]`` (one score per node).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------------------------------------------- helpers
def _mean_aggregate(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """mean over incoming edges of (w_ij * h_j); degree = number of incoming edges (unweighted count); 0 for isolated nodes."""
    n = h.shape[0]
    out = torch.zeros(n, h.shape[1], dtype=h.dtype, device=h.device)
    if src.numel() == 0:
        return out
    msg = h[src]
    if w is not None:
        msg = msg * w.to(h.dtype).unsqueeze(1)
    out.index_add_(0, dst, msg)
    deg = torch.zeros(n, dtype=h.dtype, device=h.device).index_add_(0, dst, torch.ones(dst.shape[0], dtype=h.dtype, device=h.device))
    return out / deg.clamp(min=1.0).unsqueeze(1)


def _scatter_max(e: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """per-destination maximum of e [E, H] -> [n, H] (-inf-free: nodes without edges get 0)."""
    out = torch.full((n, e.shape[1]), -1e30, dtype=e.dtype, device=e.device)
    out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce="amax", include_self=True)
    return torch.where(out <= -1e29, torch.zeros_like(out), out)


class MLP(nn.Module):
    """`layers` hidden Linear+ReLU+Dropout blocks and a linear head; ignores the graph (`edges` accepted and unused)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edges=None) -> torch.Tensor:
        h = x
        for lin in self.lins:
            h = self.drop(torch.relu(lin(h)))
        return self.head(h).squeeze(-1)


class RSAGE(nn.Module):
    """Relational GraphSAGE-mean: h' = ReLU(W_self h + sum_r W_r mean_{j in N_r(i)} (w_ij h_j)); dropout; linear head."""

    def __init__(self, in_dim: int, hidden: int, layers: int, n_relations: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.n_relations = n_relations
        self.self_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.rel_lin = nn.ModuleList([nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for _ in range(n_relations)])
                                      for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor, edges) -> torch.Tensor:
        assert len(edges) == self.n_relations, (len(edges), self.n_relations)
        h = x
        for l, self_lin in enumerate(self.self_lin):
            z = self_lin(h)
            for r, (src, dst, w) in enumerate(edges):
                if src.numel():
                    z = z + self.rel_lin[l][r](_mean_aggregate(h, src, dst, w))
            h = self.drop(torch.relu(z))
        return h

    def forward(self, x: torch.Tensor, edges) -> torch.Tensor:
        return self.head(self.encode(x, edges)).squeeze(-1)


class RGAT(nn.Module):
    """Relational GAT: per relation r, per head a GAT attention over incoming neighbours (softmax per destination node via
    scatter, per-node max subtracted), heads concatenated (hidden = heads * d_head), summed over relations plus a self term;
    ReLU; dropout; linear head. Edge weights ``w`` are NOT used by the attention (documented choice)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, n_relations: int, dropout: float, heads: int = 4):
        super().__init__()
        assert hidden % heads == 0
        self.n_relations, self.heads, self.dh = n_relations, heads, hidden // heads
        dims = [in_dim] + [hidden] * layers
        self.self_lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.w_lin = nn.ModuleList([nn.ModuleList([nn.Linear(dims[i], hidden, bias=False) for _ in range(n_relations)])
                                    for i in range(layers)])
        self.a_src = nn.ParameterList([nn.Parameter(torch.empty(n_relations, heads, self.dh)) for _ in range(layers)])
        self.a_dst = nn.ParameterList([nn.Parameter(torch.empty(n_relations, heads, self.dh)) for _ in range(layers)])
        for p in list(self.a_src) + list(self.a_dst):
            nn.init.xavier_uniform_(p)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor, edges) -> torch.Tensor:
        assert len(edges) == self.n_relations
        n = x.shape[0]
        h = x
        for l, self_lin in enumerate(self.self_lin):
            z = self_lin(h)
            for r, (src, dst, _w) in enumerate(edges):
                if not src.numel():
                    continue
                g = self.w_lin[l][r](h).view(n, self.heads, self.dh)
                es = (g * self.a_src[l][r]).sum(-1)        # [N, H]
                ed = (g * self.a_dst[l][r]).sum(-1)
                e = torch.nn.functional.leaky_relu(es[src] + ed[dst], 0.2)             # [E, H]
                e = e - _scatter_max(e.detach(), dst, n)[dst]
                a = torch.exp(e)
                den = torch.zeros(n, self.heads, dtype=a.dtype, device=a.device).index_add_(0, dst, a)
                a = a / den[dst].clamp(min=1e-12)
                out = torch.zeros(n, self.heads, self.dh, dtype=g.dtype, device=g.device)
                out.index_add_(0, dst, g[src] * a.unsqueeze(-1))
                z = z + out.reshape(n, self.heads * self.dh)
            h = self.drop(torch.relu(z))
        return h

    def forward(self, x: torch.Tensor, edges) -> torch.Tensor:
        return self.head(self.encode(x, edges)).squeeze(-1)


class RSAGETemporal(nn.Module):
    """C5: a GRU over each firm's own feature sequence (last seq_len snapshots, earlier-missing snapshots zero-padded with a
    mask) gives the node input (dim hidden); then RSAGE. forward(x, edges, mask=None): x is [N, T, in_dim] (or [N, in_dim] =
    sequence of length 1) and mask [N, T] (1 = snapshot present; defaults to all ones). The last time step is the current
    snapshot; padded steps are zero-masked inputs, and the GRU runs over all T steps (zeros carry the 'absent' signal)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, n_relations: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(in_dim + 1, hidden, batch_first=True)    # +1 = presence mask channel
        self.sage = RSAGE(hidden, hidden, layers, n_relations, dropout)

    def forward(self, x: torch.Tensor, edges, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        x = x * mask.unsqueeze(-1)
        _, hN = self.gru(torch.cat([x, mask.unsqueeze(-1)], dim=-1))
        return self.sage(hN[-1], edges)


class MLPAgg(nn.Module):
    """C0g control (A5): the same MLP on [X, mean_r(X) over incoming neighbours for each relation (weighted mean, explicit zero for
    isolated nodes: exactly what the first R-SAGE layer aggregates), log1p(in-degree_r) per relation]. forward(x, edges) builds the
    augmented input itself, so the training loop is identical to the other cells. Degree inputs exist ONLY in this model."""

    def __init__(self, in_dim: int, hidden: int, layers: int, n_relations: int, dropout: float):
        super().__init__()
        self.n_relations = n_relations; self.aug_dim = in_dim * (1 + n_relations) + n_relations
        self.mlp = MLP(self.aug_dim, hidden, layers, dropout)

    def augment(self, x: torch.Tensor, edges) -> torch.Tensor:
        assert len(edges) == self.n_relations
        parts, degs = [x], []
        for src, dst, w in edges:
            parts.append(_mean_aggregate(x, src, dst, w))
            deg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            if src.numel():
                deg.index_add_(0, dst, torch.ones(dst.shape[0], dtype=x.dtype, device=x.device))
            degs.append(torch.log1p(deg).unsqueeze(1))
        return torch.cat(parts + degs, dim=1)

    def forward(self, x: torch.Tensor, edges) -> torch.Tensor:
        return self.mlp(self.augment(x, edges))


def build_model(kind: str, in_dim: int, n_relations: int, hidden: int, layers: int, dropout: float, heads: int = 4) -> nn.Module:
    if kind == "mlp":
        return MLP(in_dim, hidden, layers, dropout)
    if kind == "mlp_agg":
        return MLPAgg(in_dim, hidden, layers, n_relations, dropout)
    if kind == "rsage":
        return RSAGE(in_dim, hidden, layers, n_relations, dropout)
    if kind == "rgat":
        return RGAT(in_dim, hidden, layers, n_relations, dropout, heads)
    if kind == "rsage_temporal":
        return RSAGETemporal(in_dim, hidden, layers, n_relations, dropout)
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------------------- rewiring
# TEST / DIAGNOSTIC HELPERS ONLY. The trainer no longer rewires (A6): C4 loads the persistent stratified rewired tables
# common.load_graph(h, r, rewire_seed=seed). Nothing below is called from train.py.
def rewire_permute_dst(src, dst, w, n_nodes: int, rng: np.random.Generator):
    """Option 1: permute dst uniformly over the edge list, keep src and w. Preserves the edge count and each source's
    out-degree; in-degree is NOT preserved (it is randomised). May create self-loops / duplicates."""
    src, dst, w = np.asarray(src), np.asarray(dst), np.asarray(w)
    return src.copy(), dst[rng.permutation(len(dst))].copy(), w.copy()


def rewire_double_edge_swap(src, dst, w, n_nodes: int, rng: np.random.Generator, swaps_per_edge: int = 100):
    """Option 2 (default): degree-preserving directed double-edge swaps. Pick two edges (a->b),(c->d), replace them with
    (a->d),(c->b); reject a swap that creates a self-loop or an edge already present (or created in this round). Preserves
    BOTH every node's out-degree and in-degree and the edge count; w stays attached to its source slot. Executed in
    vectorised rounds of E/2 disjoint random pairs (dense boolean adjacency for the existence test) until
    swaps_per_edge * E swap attempts have been made."""
    src = np.asarray(src, np.int64).copy(); dst = np.asarray(dst, np.int64).copy(); w = np.asarray(w).copy()
    E = len(src)
    if E < 2:
        return src, dst, w
    n = int(max(n_nodes, src.max() + 1, dst.max() + 1))
    assert n * n < 4 * 10**8, "snapshot too large for the dense adjacency"
    adj = np.zeros(n * n, bool); adj[src * n + dst] = True
    half = E // 2; attempts, target = 0, swaps_per_edge * E
    while attempts < target:
        perm = rng.permutation(E); a, b = perm[:half], perm[half:2 * half]
        sa, sb, da, db = src[a], src[b], dst[a], dst[b]
        nk1, nk2 = sa * n + db, sb * n + da
        ok = (sa != db) & (sb != da) & (nk1 != nk2) & ~adj[nk1] & ~adj[nk2]
        idx = np.flatnonzero(ok)
        if len(idx):
            newk = np.concatenate([nk1[idx], nk2[idx]])
            _, first = np.unique(newk, return_index=True)         # two accepted swaps must not create the same new edge
            dup = np.ones(len(newk), bool); dup[first] = False
            bad = dup[:len(idx)] | dup[len(idx):]
            idx = idx[~bad]
        if len(idx):
            ia, ib = a[idx], b[idx]
            adj[src[ia] * n + dst[ia]] = False; adj[src[ib] * n + dst[ib]] = False
            dst[ia], dst[ib] = db[idx], da[idx]
            adj[src[ia] * n + dst[ia]] = True; adj[src[ib] * n + dst[ib]] = True
        attempts += max(half, 1)
    return src, dst, w


def rewire(src, dst, w, n_nodes: int, rng: np.random.Generator, method: str = "double_edge_swap", swaps_per_edge: int = 100):
    """Degree-preserving (default) rewiring used by cell C4. method: 'double_edge_swap' | 'permute_dst'."""
    if method == "double_edge_swap":
        return rewire_double_edge_swap(src, dst, w, n_nodes, rng, swaps_per_edge)
    if method == "permute_dst":
        return rewire_permute_dst(src, dst, w, n_nodes, rng)
    raise ValueError(method)
