"""Riemannion: Muon on the fixed-rank matrix manifold, for LoRA factor pairs.

Clean-room implementation of arXiv:2507.12142 "LoRA meets Riemannion: Muon
Optimizer for Parametrization-independent Low-Rank Adapters" (equation and
algorithm numbers below refer to the v2 HTML). Written from the paper only —
the authors' reference repository is unlicensed and was not consulted.

Why not plain Muon on LoRA: per-factor Muon orthogonalizes ``lora_A`` and
``lora_B`` separately, so the induced step on the adapter product depends on
the (arbitrary) factor gauge — the paper shows it underperforming AdamW.
Riemannion instead treats the adapter increment as a point on the manifold
``M_r`` of rank-r matrices and performs the whole Muon update (momentum,
orthogonalization, step) in the tangent space of ``M_r``, which makes the
update on ``dW`` independent of how ``dW`` is factored.

PEFT convention mapping
-----------------------
PEFT's LoRA computes ``dW = scaling * lora_B.weight @ lora_A.weight`` with
``lora_A.weight`` of shape ``(r, n)`` (n = in_features) and ``lora_B.weight``
of shape ``(m, r)`` (m = out_features). The paper works on ``X = A B^T`` with
``A in R^{m x r}``, ``B in R^{n x r}``; we therefore identify

    P := lora_B.weight            (m x r)   -- the paper's left factor A
    Q := lora_A.weight^T          (n x r)   -- the paper's right factor B
    X := P Q^T = lora_B @ lora_A  (m x n)

The constant PEFT ``scaling = alpha/r`` multiplies X everywhere; because the
loss sees ``scaling * X``, autograd gives exactly ``lora_B.grad = G_X @ Q``
and ``lora_A.grad^T = G_X^T @ P`` for ``G_X := dL/dX`` (the scaling is folded
into ``G_X``), so no scaling factor appears in this module. Orthogonalization
normalizes the update magnitude anyway, so ``alpha`` only enters through the
loss gradient as usual.

Each optimizer param group holds exactly one LoRA layer's factor pair in the
order ``params=[lora_A.weight, lora_B.weight]`` plus a ``names`` entry for
error messages.

Algorithm (paper Alg. 4, one step; sub-procedures Alg. 1 "OrthoLR" and
Alg. 2 "ProjectLR"):

1. re-factor ``X = A_L G B_R^T`` (Eq. 4) via QR of P and Q;
2. express the Euclidean gradient in tangent coordinates (Eq. 5/6);
3. vector-transport the stored momentum by projecting it onto the new
   tangent space (Eq. 9-11) and accumulate heavy-ball momentum;
4. orthogonalize the momentum with QR + an SVD of a small 2r x 2r core
   (Alg. 1 — never Newton-Schulz on full matrices), project back onto the
   tangent space (Alg. 2);
5. retract to rank r via a truncated SVD of the rank <= 2r update (Eq. 8),
   applying decoupled weight decay inside the retraction (Alg. 4 step 5).

Extensions beyond the paper, both flagged: ``scale_lr`` (default on)
rescales the step by ``0.2 * sqrt(m*n/r)`` so a shared AdamW-scale learning
rate transfers (Muon/Moonlight RMS-matching: an orthogonalized rank-r
update has Frobenius norm ~ sqrt(r), hence entry RMS ~ sqrt(r/(m*n)); the
rescale makes the update RMS ~ 0.2*lr, AdamW-like). ``nesterov`` (default
off) uses ``grad + momentum*buf`` as the direction to orthogonalize, torch
SGD-style; the paper only specifies heavy-ball.

This module imports torch at the top: it is pod-side only and must never be
imported by ``scimt`` package ``__init__``s (``import scimt`` stays
torch-free — the axolotl plugin imports it lazily inside the hook).
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["Riemannion", "CombinedOptimizer"]


def _reject_dtensor(tensor: torch.Tensor, name: str) -> None:
    """Fail loud on FSDP2-sharded (DTensor) LoRA params.

    Riemannion needs whole factor matrices for the QR/SVD core; DTensor
    support (sharded QR or gather-compute-scatter) is future work.
    """
    if any(klass.__name__ == "DTensor" for klass in type(tensor).__mro__):
        raise RuntimeError(
            f"Riemannion got a DTensor parameter ({name}): LoRA adapter "
            "params must be excluded from FSDP sharding to use Riemannion "
            "(keep FSDP for the frozen base only). DTensor support is "
            "future work."
        )


def _project_lr(
    a: torch.Tensor, b: torch.Tensor, a_l: torch.Tensor, b_r: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alg. 2 (ProjectLR): project ``Z = a @ b^T`` onto ``T_X M_r``.

    Returns the tangent pair ``(adot, bdot)`` of Eq. 5, i.e.
    ``P_{T_X}(Z) = adot @ b_r^T + a_l @ bdot^T`` with ``adot^T a_l = 0``,
    where (Eq. 6) ``P_{T_X}(Z) = a_l a_l^T Z + (I - a_l a_l^T) Z b_r b_r^T``.
    Cost O((m+n) r r') — Z is never materialized.
    """
    bdot = b @ (a.T @ a_l)  # = Z^T a_l
    adot = (a - a_l @ (a_l.T @ a)) @ (b.T @ b_r)  # = (I - a_l a_l^T) Z b_r
    return adot, bdot


def _ortho_lr(
    left: torch.Tensor, right: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alg. 1 (OrthoLR): matrix sign of ``left @ right^T`` in factored form.

    QR both tall blocks (O((m+n) r^2)), then a thin SVD of the small
    ``T_L T_R^T`` core (2r x 2r for tangent vectors; O(r^3)) — this replaces
    Muon's Newton-Schulz iteration on the full matrix. Returns ``(oa, ob)``
    with ``Ortho(left @ right^T) = oa @ ob^T``. Singular directions with
    sigma <= eps * sigma_max are zeroed rather than blown up to unit size.
    """
    q_l, t_l = torch.linalg.qr(left)
    q_r, t_r = torch.linalg.qr(right)
    u_c, s_c, vh_c = torch.linalg.svd(t_l @ t_r.T)
    keep = s_c > eps * s_c.max().clamp(min=torch.finfo(s_c.dtype).tiny)
    oa = q_l @ (u_c * keep.to(u_c.dtype))
    ob = q_r @ vh_c.T
    return oa, ob


def _solve_r(mat: torch.Tensor, r_tri: torch.Tensor) -> torch.Tensor:
    """``mat @ r_tri^{-1}`` via pseudo-inverse of the small r x r factor.

    ``r_tri`` comes from a QR and is singular at standard LoRA init
    (lora_B = 0 puts dW at the rank-deficient boundary of M_r, where the
    paper's Eq. 6 projector is not defined); the pinv makes the first steps
    a plain descent direction on the recoverable subspace instead of NaNs.
    """
    return mat @ torch.linalg.pinv(r_tri, rtol=1e-6)


class Riemannion(torch.optim.Optimizer):
    """Muon on the fixed-rank manifold for LoRA (A, B) pairs (arXiv:2507.12142).

    Each param group is one LoRA layer: ``params=[lora_A.weight (r x n),
    lora_B.weight (m x r)]`` (PEFT convention, ``dW ∝ lora_B @ lora_A``),
    optionally ``names=(a_name, b_name)``.

    Args:
        lr: step size; with ``scale_lr`` (default) this is AdamW-scale.
        momentum: heavy-ball coefficient beta (Eq. 10-11), default 0.9.
        weight_decay: decoupled decay gamma applied inside the retraction
            (Alg. 4 step 5): retract ``(1 - lr*gamma) X - lr * direction``.
        nesterov: use ``grad + beta * momentum`` as the update direction
            (extension; the paper specifies heavy-ball only).
        scale_lr: multiply the step by ``0.2 * sqrt(m*n/r)`` (RMS-matching,
            see module docstring), default True.
        eps: relative cutoff for singular directions in OrthoLR.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        scale_lr: bool = True,
        eps: float = 1e-7,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Riemannion: lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Riemannion: momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(
                f"Riemannion: weight_decay must be >= 0, got {weight_decay}"
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            scale_lr=scale_lr,
            eps=eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._check_group(group)

    @staticmethod
    def _check_group(group: dict[str, Any]) -> None:
        names = group.get("names", ("lora_A", "lora_B"))
        if len(group["params"]) != 2:
            raise ValueError(
                "Riemannion: each param group must hold exactly one LoRA "
                f"factor pair [lora_A.weight, lora_B.weight], got "
                f"{len(group['params'])} params for {names}"
            )
        a, b = group["params"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"Riemannion: LoRA factors must be 2-D matrices, got shapes "
                f"{tuple(a.shape)} / {tuple(b.shape)} for {names} (stacked "
                "3-D expert tensors are not supported)"
            )
        if a.shape[0] != b.shape[1]:
            raise ValueError(
                "Riemannion: expected params=[lora_A.weight (r x n), "
                f"lora_B.weight (m x r)] sharing rank r, got {tuple(a.shape)}"
                f" / {tuple(b.shape)} for {names}"
            )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: D102 - torch API
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            a_param, b_param = group["params"]  # peft lora_A (r x n), lora_B (m x r)
            names = group.get("names", ("lora_A", "lora_B"))
            for tensor, name in ((a_param, names[0]), (b_param, names[1])):
                _reject_dtensor(tensor, str(name))
                if tensor.grad is not None:
                    _reject_dtensor(tensor.grad, f"grad of {name}")
            if a_param.grad is None and b_param.grad is None:
                continue
            if a_param.grad is None or b_param.grad is None:
                raise ValueError(
                    f"Riemannion: factor pair {names} has a gradient on only "
                    "one factor — both LoRA factors must be trainable"
                )
            self._step_pair(group, a_param, b_param)
        return loss

    def _step_pair(
        self, group: dict[str, Any], a_param: torch.Tensor, b_param: torch.Tensor
    ) -> None:
        beta = group["momentum"]
        gamma = group["weight_decay"]
        eps = group["eps"]

        # fp32 compute (QR/SVD are unstable-or-unsupported in bf16); keep
        # fp64 if the params are fp64 (numerical tests).
        dtype = torch.float64 if a_param.dtype == torch.float64 else torch.float32
        p = b_param.to(dtype)  # paper's A: m x r
        q = a_param.to(dtype).T  # paper's B: n x r
        g_p = b_param.grad.to(dtype)  # = G_X @ q, m x r
        g_q = a_param.grad.to(dtype).T  # = G_X^T @ p, n x r
        m, r = p.shape
        n = q.shape[0]

        # (1) re-factor X = A_L G B_R^T (Eq. 4): QR gives orthonormal bases
        # of the factor column spaces; G = R_P R_Q^T is the r x r core.
        a_l, r_p = torch.linalg.qr(p)
        b_r, r_q = torch.linalg.qr(q)
        core_g = r_p @ r_q.T

        # (2) Euclidean gradient in tangent coordinates (Eq. 5 with the
        # gauge Ȧ^T A_L = 0). G_X is never materialized: from the factor
        # grads, G_X b_r = g_p r_q^{-1} and G_X^T a_l = g_q r_p^{-1}
        # (paper's Eq. 18 backward-pass trick, adapted to PEFT autograd).
        grad_br = _solve_r(g_p, r_q)  # m x r
        bdot_g = _solve_r(g_q, r_p)  # n x r  (= G_X^T a_l)
        adot_g = grad_br - a_l @ (a_l.T @ grad_br)  # (I - a_l a_l^T) G_X b_r

        # (3) vector transport of momentum = projection onto the new tangent
        # space (Eq. 9), then heavy-ball accumulation (Eq. 10-11 / Alg. 4
        # step 3). Momentum is stored in factored ambient form
        # ξ = a_hb @ b_hb^T (m x 2r, n x 2r) — Alg. 4 step 6.
        state = self.state[a_param]
        if "a_hb" in state:
            adot_p, bdot_p = _project_lr(
                state["a_hb"].to(dtype), state["b_hb"].to(dtype), a_l, b_r
            )
            adot_m = beta * adot_p + adot_g
            bdot_m = beta * bdot_p + bdot_g
        else:
            adot_m, bdot_m = adot_g, bdot_g
        if group["nesterov"]:
            # extension (torch-SGD convention); the paper is heavy-ball only
            adot_d = adot_g + beta * adot_m
            bdot_d = bdot_g + beta * bdot_m
        else:
            adot_d, bdot_d = adot_m, bdot_m

        # (4) manifold-aware orthogonalization of the momentum: OrthoLR
        # (Alg. 1) on ξ = adot_d b_r^T + a_l bdot_d^T = [adot_d, a_l]
        # [b_r, bdot_d]^T, then ProjectLR back to the tangent space
        # (Alg. 4 step 4). The SVD core is 2r x 2r — no full-matrix work.
        o_a, o_b = _ortho_lr(
            torch.cat([adot_d, a_l], dim=1), torch.cat([b_r, bdot_d], dim=1), eps
        )
        adot, bdot = _project_lr(o_a, o_b, a_l, b_r)

        # store momentum for the next step's transport (Alg. 4 step 6:
        # A_HB, B_HB := [Ȧ, A_L], [B_R, Ḃ] — the post-orthogonalization
        # tangent vector, in ambient factored form)
        state["a_hb"] = torch.cat([adot, a_l], dim=1)
        state["b_hb"] = torch.cat([b_r, bdot], dim=1)

        # per-matrix RMS-matching LR rescale (flagged extension, default on)
        eta = group["lr"]
        if group["scale_lr"]:
            eta = eta * 0.2 * (m * n / r) ** 0.5

        # (5) retraction (Eq. 8 / Alg. 4 step 5): rank-r truncated SVD of
        #   (1 - eta*gamma) X - eta ξ
        # = [a_l, -eta*adot] @ [(1-eta*gamma) b_r G^T - eta*bdot, b_r]^T,
        # a rank <= 2r product — QR both blocks, SVD the 2r x 2r core,
        # truncate to r. Decoupled weight decay shrinks X inside the
        # retraction, matching the paper's -eta*(Ḃ + gamma*B) term.
        left = torch.cat([a_l, -eta * adot], dim=1)
        right = torch.cat(
            [(1.0 - eta * gamma) * (b_r @ core_g.T) - eta * bdot, b_r], dim=1
        )
        q_l, t_l = torch.linalg.qr(left)
        q_r, t_r = torch.linalg.qr(right)
        u_c, s_c, vh_c = torch.linalg.svd(t_l @ t_r.T)
        u_r = q_l @ u_c[:, :r]
        v_r = q_r @ vh_c[:r].T
        s_r = s_c[:r]

        # write back through a balanced split (gauge choice — the paper's
        # Alg. 4 step 7 uses A_L := U, B := Σ V^T; sqrt(Σ) on both factors
        # keeps their norms comparable, which is kinder to low precision).
        s_half = s_r.clamp(min=0.0).sqrt()
        b_param.copy_((u_r * s_half).to(b_param.dtype))  # new P, m x r
        a_param.copy_((v_r * s_half).T.to(a_param.dtype))  # new Q^T, r x n


class CombinedOptimizer(torch.optim.Optimizer):
    """Delegating wrapper over several optimizers (Riemannion + AdamW).

    Exposes the child optimizers' *live* param-group dicts as its own, so
    LR schedulers that mutate ``optimizer.param_groups[i]["lr"]`` reach the
    groups the children actually read. Subclasses ``torch.optim.Optimizer``
    because torch LR schedulers type-check their optimizer argument.
    """

    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        self.optimizers = list(optimizers)
        super().__init__(
            [p for o in self.optimizers for g in o.param_groups for p in g["params"]],
            defaults={},
        )
        # replace the single catch-all group built by Optimizer.__init__
        # with the children's own group dicts (shared, not copied)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def step(self, closure: Any = None) -> Any:  # noqa: D102 - torch API
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: D102
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:  # noqa: D102 - torch API
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102
        children = state_dict["optimizers"]
        if len(children) != len(self.optimizers):
            raise ValueError(
                f"CombinedOptimizer: state_dict has {len(children)} child "
                f"optimizers, expected {len(self.optimizers)}"
            )
        for optimizer, child in zip(self.optimizers, children):
            optimizer.load_state_dict(child)
