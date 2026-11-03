"""Physics-informed neural network for leak detection, localisation and quantification.

What the paper specifies (section III-C/D): a feed-forward MLP with tanh activations;
inputs are sensor-derived features (location x, measured P, Q); outputs are leak
probability (classification), leak location and leak rate (regression); loss
L = L_data + lambda_phys L_physics with L_data = BCE + MSE(location) + MSE(rate) and
L_physics the squared mass and momentum residuals at randomly sampled collocation points
(x_i, t_i), derivatives by automatic differentiation; Adam; inputs and outputs scaled.

What the paper does not specify, and the interpretation chosen here (RESULTS.md 1.4):

* The residuals need alpha_k, rho_k, u_k and p as differentiable functions of (x, t). The
  network therefore has two parts sharing one scenario encoding h = E(features):
    - heads  g(h) -> (leak probability, x_L / L, Q_leak / Q_scale);
    - fields f(x/L, t/T, h) -> (p, alpha_G, u_G, u_L), a tanh MLP.
  The fields are tied to the measurements by an MSE term on p and Q at the two sensor
  positions inside the physics window (this is how "measured P, Q" enter the physics).
  The physics residual is evaluated on the fields with the leak sink placed at the
  predicted location with the predicted rate (times the predicted probability), so the
  physics term back-propagates into the heads. The window is t in [200, 300] s, after
  every leak has fully opened, so no onset time has to be predicted.
* The black-box baseline is the identical network trained with lambda_phys = 0 (the
  sensor-fit term is kept, so the only difference is the physics residual).
* MSE(location) is applied to leak scenarios only; MSE(rate) uses rate 0 for no-leak.

Reproducibility: every random draw (scenario batch, sensor sample indices, collocation
points) is made with one NumPy generator outside the TensorFlow graph and passed in as
tensors. Stateful TF random ops inside one graph can execute in varying order and consume
the generator state differently from run to run, which made earlier runs with the
physics term non-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from .data.config import REGIMES, PipelineConfig
from .features import PHYSICS_WINDOW_S, RATE_SCALE
from .physics import P_SCALE, U_SCALE, residuals_from_fields

T_TOTAL_S = 300.0


@dataclass
class PINNConfig:
    hidden: int = 64
    depth: int = 3
    latent: int = 32
    lambda_phys: float = 0.1
    lambda_sensor: float = 1.0
    lr: float = 2e-3
    batch_scenarios: int = 32
    n_collocation: int = 256  # per batch
    n_sensor_points: int = 64  # sensor samples per scenario per batch
    steps: int = 1500
    seed: int = 0
    pipe: PipelineConfig = field(default_factory=PipelineConfig)


def _mlp(n_in: int, n_out: int, hidden: int, depth: int, seed: int, out_act=None, name=""):
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    layers = [tf.keras.layers.Input(shape=(n_in,))]
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(hidden, activation="tanh", kernel_initializer=init))
    layers.append(tf.keras.layers.Dense(n_out, activation=out_act, kernel_initializer=init))
    return tf.keras.Sequential(layers, name=name)


class LeakPINN(tf.keras.Model):
    """Encoder + heads + field network. ``call`` returns the head outputs."""

    def __init__(self, n_features: int, cfg: PINNConfig):
        super().__init__(name="leak_pinn")
        self.cfg = cfg
        self.encoder = _mlp(
            n_features, cfg.latent, cfg.hidden, cfg.depth, cfg.seed, "tanh", "encoder"
        )
        self.heads = _mlp(cfg.latent, 3, cfg.hidden, 1, cfg.seed + 1, None, "heads")
        self.fields = _mlp(cfg.latent + 2, 4, cfg.hidden, cfg.depth, cfg.seed + 2, None, "fields")

    def encode(self, features: tf.Tensor) -> tf.Tensor:
        return self.encoder(features)

    def call(self, features: tf.Tensor, training=False):
        h = self.encode(features)
        raw = self.heads(h)
        return {
            "logit": raw[:, 0],
            "prob": tf.sigmoid(raw[:, 0]),
            "loc": tf.sigmoid(raw[:, 1]),  # x_L / L
            "rate": tf.sigmoid(raw[:, 2]),  # leak fraction / RATE_SCALE
        }

    def field_values(self, h: tf.Tensor, x_hat: tf.Tensor, t_hat: tf.Tensor):
        """Physical fields at scaled coordinates. ``h`` (n, latent); x_hat, t_hat (n, 1)."""
        raw = self.fields(tf.concat([h, x_hat, t_hat], axis=1))
        p = self.cfg.pipe.p_outlet + P_SCALE * raw[:, 0:1]
        alpha_g = tf.sigmoid(raw[:, 1:2])
        u_g = U_SCALE * raw[:, 2:3]
        u_l = U_SCALE * raw[:, 3:4]
        return p, alpha_g, u_g, u_l


class Batch(NamedTuple):
    """Tensors for one training step (a NamedTuple so tf.function can trace it)."""

    features: tf.Tensor  # (n, F)
    is_leak: tf.Tensor  # (n,)
    loc: tf.Tensor  # (n,)
    rate: tf.Tensor  # (n,)
    t_sensor: tf.Tensor  # (T_w,) seconds
    sensors: tf.Tensor  # (n, T_w, 4) noisy p_in, p_out, q_in, q_out
    nominal_mass_flow: tf.Tensor  # (n,) kg/s
    drag_coeff: tf.Tensor  # (n,)
    sensor_ti: tf.Tensor  # (n, n_pts) int32 sample indices into t_sensor
    col_x: tf.Tensor  # (n * per, 1) collocation x / L
    col_t: tf.Tensor  # (n * per, 1) collocation t / T


def _mass_flow_nominal(d: dict, idx: np.ndarray, pipe: PipelineConfig) -> np.ndarray:
    ag = np.array([REGIMES[r].inlet_gas_fraction for r in d["regime"][idx]])
    u = d["nominal_velocity"][idx]
    return pipe.area_m2 * u * (ag * pipe.rho_gas_ref + (1 - ag) * pipe.rho_liquid)


def make_batch(
    d: dict,
    idx: np.ndarray,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    t_w: np.ndarray,
    sensors_w: np.ndarray,
    cfg: PINNConfig,
    rng: np.random.Generator,
) -> Batch:
    """Assemble tensors for the scenarios ``idx`` (positions into ``features``/``targets``)."""
    n = len(idx)
    per = max(cfg.n_collocation // max(n, 1), 1)
    drag = np.array([REGIMES[r].drag_coeff for r in d["regime"][idx]])
    ti = rng.integers(0, len(t_w), size=(n, cfg.n_sensor_points))
    col_x = rng.uniform(0.0, 1.0, size=(n * per, 1))
    col_t = rng.uniform(
        PHYSICS_WINDOW_S[0] / T_TOTAL_S, PHYSICS_WINDOW_S[1] / T_TOTAL_S, size=(n * per, 1)
    )
    return Batch(
        features=tf.constant(features, tf.float32),
        is_leak=tf.constant(targets["is_leak"], tf.float32),
        loc=tf.constant(targets["loc"], tf.float32),
        rate=tf.constant(targets["rate"], tf.float32),
        t_sensor=tf.constant(t_w, tf.float32),
        sensors=tf.constant(sensors_w, tf.float32),
        nominal_mass_flow=tf.constant(_mass_flow_nominal(d, idx, cfg.pipe), tf.float32),
        drag_coeff=tf.constant(drag, tf.float32),
        sensor_ti=tf.constant(ti, tf.int32),
        col_x=tf.constant(col_x, tf.float32),
        col_t=tf.constant(col_t, tf.float32),
    )


def data_loss(out: dict, b: Batch) -> dict[str, tf.Tensor]:
    bce = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=b.is_leak, logits=out["logit"])
    )
    n_leak = tf.maximum(tf.reduce_sum(b.is_leak), 1.0)
    mse_loc = tf.reduce_sum(b.is_leak * (out["loc"] - b.loc) ** 2) / n_leak
    mse_rate = tf.reduce_mean((out["rate"] - b.rate) ** 2)
    return {"bce": bce, "mse_loc": mse_loc, "mse_rate": mse_rate}


def sensor_loss(model: LeakPINN, h: tf.Tensor, b: Batch):
    """MSE between field-network p, Q at x = 0 and x = L and the measured sensors."""
    pipe = model.cfg.pipe
    n_pts = tf.shape(b.sensor_ti)[1]
    t_sel = tf.gather(b.t_sensor, b.sensor_ti)  # (n, n_pts)
    meas = tf.gather(b.sensors, b.sensor_ti, batch_dims=1)  # (n, n_pts, 4)
    h_rep = tf.repeat(h, n_pts, axis=0)
    t_hat = tf.reshape(t_sel, (-1, 1)) / T_TOTAL_S
    loss = 0.0
    for side, x_hat_val in (("in", 0.0), ("out", 1.0)):
        x_hat = tf.fill(tf.shape(t_hat), x_hat_val)
        p, ag, ug, ul = model.field_values(h_rep, x_hat, t_hat)
        q = pipe.area_m2 * (ag * ug + (1 - ag) * ul)
        c = 0 if side == "in" else 1
        p_meas = tf.reshape(meas[..., c], (-1, 1))
        q_meas = tf.reshape(meas[..., 2 + c], (-1, 1))
        loss += tf.reduce_mean(((p - p_meas) / P_SCALE) ** 2)
        loss += tf.reduce_mean(((q - q_meas) / (0.1 * pipe.area_m2 * U_SCALE)) ** 2)
    return loss


def physics_loss(model: LeakPINN, h: tf.Tensor, out: dict, b: Batch):
    """Mean squared scaled residuals at the batch's collocation points."""
    pipe = model.cfg.pipe
    n = tf.shape(h)[0]
    per = tf.shape(b.col_x)[0] // n
    h_rep = tf.repeat(h, per, axis=0)
    x_hat, t_hat = b.col_x, b.col_t
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_hat, t_hat])
        p, ag, ug, ul = model.field_values(h_rep, x_hat, t_hat)
    grads = {}
    for name, fld in (("p", p), ("ag", ag), ("ug", ug), ("ul", ul)):
        gx, gt = tape.gradient(fld, [x_hat, t_hat])
        grads[name] = (gt / T_TOTAL_S, gx / pipe.length_m)
    del tape
    x = x_hat * pipe.length_m
    leak_rate = tf.repeat(out["prob"] * out["rate"] * RATE_SCALE * b.nominal_mass_flow, per)[
        :, None
    ]
    leak_x = tf.repeat(out["loc"] * pipe.length_m, per)[:, None]
    drag = tf.repeat(b.drag_coeff, per)[:, None]
    res = residuals_from_fields(
        p,
        ag,
        ug,
        ul,
        grads["p"][0],
        grads["p"][1],
        grads["ag"][0],
        grads["ag"][1],
        grads["ug"][0],
        grads["ug"][1],
        grads["ul"][0],
        grads["ul"][1],
        x,
        leak_rate,
        leak_x,
        drag,
        pipe,
    )
    return sum(tf.reduce_mean(r**2) for r in res.values())


def train_pinn(
    d: dict,
    train_idx: np.ndarray,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    t_w: np.ndarray,
    sensors_w: np.ndarray,
    cfg: PINNConfig,
    verbose: bool = False,
) -> tuple[LeakPINN, list[dict[str, float]]]:
    """Train with Adam on random scenario batches. Returns the model and the loss log."""
    tf.keras.utils.set_random_seed(cfg.seed)
    tf.config.experimental.enable_op_determinism()
    np_rng = np.random.default_rng(cfg.seed)
    model = LeakPINN(features.shape[1], cfg)
    opt = tf.keras.optimizers.Adam(cfg.lr)
    log: list[dict[str, float]] = []
    n = len(train_idx)
    use_physics = cfg.lambda_phys > 0

    @tf.function
    def step(b: Batch):
        with tf.GradientTape() as tape:
            h = model.encode(b.features)
            out = model(b.features)
            dl = data_loss(out, b)
            sl = sensor_loss(model, h, b)
            pl = physics_loss(model, h, out, b) if use_physics else tf.constant(0.0)
            total = (
                dl["bce"]
                + dl["mse_loc"]
                + dl["mse_rate"]
                + cfg.lambda_sensor * sl
                + cfg.lambda_phys * pl
            )
        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables, strict=True))
        return total, dl["bce"], dl["mse_loc"], dl["mse_rate"], sl, pl

    for it in range(cfg.steps):
        pos = np_rng.choice(n, size=min(cfg.batch_scenarios, n), replace=False)
        b = make_batch(
            d,
            train_idx[pos],
            features[pos],
            {k: v[pos] for k, v in targets.items()},
            t_w,
            sensors_w[pos],
            cfg,
            np_rng,
        )
        total, bce, ml, mr, sl, pl = (float(v) for v in step(b))
        if it % 50 == 0 or it == cfg.steps - 1:
            rec = {
                "step": it,
                "total": total,
                "bce": bce,
                "mse_loc": ml,
                "mse_rate": mr,
                "sensor": sl,
                "physics": pl,
            }
            log.append(rec)
            if verbose:
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    return model, log


def predict(model: LeakPINN, features: np.ndarray) -> dict[str, np.ndarray]:
    out = model(tf.constant(features, tf.float32))
    return {k: v.numpy() for k, v in out.items()}


def save_model(model: LeakPINN, path) -> None:
    """Save weights (the architecture is rebuilt from ``PINNConfig`` on load)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))


def load_model(path, n_features: int, cfg: PINNConfig) -> LeakPINN:
    model = LeakPINN(n_features, cfg)
    model(tf.zeros((1, n_features), tf.float32))
    model.field_values(tf.zeros((1, cfg.latent)), tf.zeros((1, 1)), tf.zeros((1, 1)))
    model.load_weights(str(path))
    return model
