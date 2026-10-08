"""Función de pérdida con datos ponderados por temporada + física."""

import torch
import torch.nn.functional as F


def physics_residual(model, X_norm, mean, std, alpha=1e-3, beta=1e-4):
    """Residuo de la ecuación de difusión + advección topográfica.

        ∂T/∂t ≈ α ∇²T + β ∇z · ∇T

    Índices de X_norm: 0=x, 1=y, 2=hour, 3=season, 4=elev, 5=slope,
    6=aspect, 7=NDVI, 8=FVC, 9=ALBEDO, 10=EMIS, 11=LST, 12=T_air_C,
    13=RH, 14=wind_speed, 15=SSRD, 16=MRT
    """
    X = X_norm.clone().requires_grad_(True)
    y = model(X)

    grads = torch.autograd.grad(y, X, torch.ones_like(y), create_graph=True)[0]
    dT_dx = grads[:, 0:1]
    dT_dy = grads[:, 1:2]
    dT_dt = grads[:, 2:3]

    d2T_dx2 = torch.autograd.grad(dT_dx, X, torch.ones_like(dT_dx),
                                   create_graph=True)[0][:, 0:1]
    d2T_dy2 = torch.autograd.grad(dT_dy, X, torch.ones_like(dT_dy),
                                   create_graph=True)[0][:, 1:2]

    slope = X[:, 5:6]
    aspect = X[:, 6:7]
    dz_dx = slope * torch.cos(aspect)
    dz_dy = slope * torch.sin(aspect)
    advection = dz_dx * dT_dx + dz_dy * dT_dy

    residual = dT_dt - (alpha * (d2T_dx2 + d2T_dy2) + beta * advection)
    return (residual ** 2).mean()


def temporal_smoothness(model, X_norm, hour_idx=2, delta_h=0.5):
    """Penaliza cambios bruscos entre horas consecutivas."""
    X_next = X_norm.clone()
    X_next[:, hour_idx] = X_next[:, hour_idx] + delta_h
    y_here = model(X_norm)
    y_next = model(X_next)
    return ((y_next - y_here) ** 2).mean()


def total_loss(model, X_batch, y_batch, weights, mean, std,
               lambda_data=1.0, lambda_phys=0.1, lambda_smooth=0.05):
    """Pérdida total ponderada por temporada.

    Parámetros
    ----------
    weights : torch.Tensor (B,) — peso por muestra (calculado del ESS por temporada)
    """
    y_pred = model(X_batch)

    # Pérdida de datos ponderada
    err2 = (y_pred - y_batch) ** 2                     # (B, 1)
    L_data = (err2.squeeze(-1) * weights).mean()       # ponderada

    L_phys = physics_residual(model, X_batch, mean, std)
    L_smooth = temporal_smoothness(model, X_batch)

    L = (lambda_data * L_data
         + lambda_phys * L_phys
         + lambda_smooth * L_smooth)

    return L, {"data": L_data.item(),
               "phys": L_phys.item(),
               "smooth": L_smooth.item()}