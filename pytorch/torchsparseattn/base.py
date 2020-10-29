import torch

from torch import autograd as ta


class _BaseBatchProjection(ta.Function):
    """Applies a sample-wise normalizing projection over a batch."""

    @classmethod
    def forward(cls, ctx, x, alpha=None, beta=None, lengths=None):
        requires_squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = x.size()
        y_star = torch.zeros_like(x)

        has_lengths = True
        if lengths is None:
            has_lengths = False
            lengths = [max_dim] * n_samples

        for i in range(n_samples):
            if alpha is None and beta is None:
                y_star[i, :lengths[i]] = cls.project(x[i, :lengths[i]])
            elif alpha is not None and beta is None:
                y_star[i, :lengths[i]] = cls.project(x[i, :lengths[i]], alpha)
            elif alpha is not None and beta is not None:
                y_star[i, :lengths[i]] = cls.project(x[i, :lengths[i]], alpha, beta)

        if requires_squeeze:
            y_star = y_star.squeeze(0)

        # ctx.mark_non_differentiable(y_star)
        if has_lengths:
            ctx.mark_non_differentiable(lengths)
            ctx.save_for_backward(y_star, lengths)
        else:
            ctx.save_for_backward(y_star)

        return y_star

    @classmethod
    def backward(cls, ctx, dout):
        if not ctx.needs_input_grad[0]:
            return None

        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            raise ValueError("Cannot differentiate {} w.r.t. the "
                             "sequence lengths".format(ctx.__name__))

        saved = ctx.saved_tensors
        if len(saved) == 2:
            y_star, lengths = saved
        else:
            y_star, = saved
            lengths = None

        requires_squeeze = False
        if y_star.dim() == 1:
            y_star = y_star.unsqueeze(0)
            dout = dout.unsqueeze(0)
            requires_squeeze = True

        n_samples, max_dim = y_star.size()
        din = torch.zeros_like(y_star)
        if lengths is None:
            lengths = [max_dim] * n_samples

        for i in range(n_samples):
            din[i, :lengths[i]] = cls.project_jv(dout[i, :lengths[i]], y_star[i, :lengths[i]])

        if requires_squeeze:
            din = din.squeeze(0)

        return din, None, None, None
