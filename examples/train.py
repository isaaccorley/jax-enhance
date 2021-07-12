import argparse

import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch_enhance
import jax_enhance


def main(cfg):
    train_dataset = torch_enhance.datasets.BSDS500(
        scale_factor=cfg.scale_factor,
        image_size=cfg.image_size,
        set_type="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    writer = SummaryWriter()

    # init rng
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # init model
    model = getattr(jax_enhance.models, cfg.model)(cfg.scale_factor, cfg.channels)
    params = model.init(init_rng, jnp.ones((1, cfg.image_size, cfg.image_size, cfg.channels)))
    tx = optax.adam(learning_rate=3E-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=tx
    )

    n_iter = 0
    for epoch in range(cfg.epochs):
        writer.add_scalar(tag="epoch", scalar_value=epoch, global_step=n_iter)
        rng, input_rng = jax.random.split(rng)
        pbar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=False)
        for batch in pbar:
            lr, hr = batch
            lr, hr = lr.permute(0, 2, 3, 1), hr.permute(0, 2, 3, 1)
            lr, hr = jnp.asarray(lr), jnp.asarray(hr)
            state, metrics, batch = train_step(state, model, {"lr": lr, "hr": hr})

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(metrics["loss"])))

            if n_iter % cfg.log_every_n == 0:
                for k, v in metrics.items():
                    writer.add_scalar(tag=k, scalar_value=float(v), global_step=n_iter)
                    baseline_image = jax.image.resize(
                        batch["lr"][0], shape=(cfg.image_size, cfg.image_size, cfg.channels), method="bicubic"
                    )
                    image = jnp.concatenate((baseline_image, batch["sr"][0], batch["hr"][0]), axis=1)
                    image = jnp.clip(image, 0.0, 1.0)
                    writer.add_image("image", np.array(image), global_step=n_iter, dataformats="HWC")

            n_iter += 1

        checkpoints.save_checkpoint(ckpt_dir=writer.log_dir, target=state, step=f"epoch_{epoch}", keep=5)


def compute_metrics(predictions, targets):
    return dict(
        mae=jax_enhance.metrics.mae(predictions,targets),
        mse=jax_enhance.metrics.mse(predictions,targets),
        psnr=jax_enhance.metrics.psnr(predictions,targets)
    )

#@jax.jit
def train_step(state, model, batch):
    def loss_fn(params):
        sr = model.apply({"params": params}, batch["lr"])
        loss = jax_enhance.losses.mse(sr, batch["hr"])
        return loss, sr

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, sr), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(sr, batch["hr"])
    metrics["loss"] = loss
    batch["sr"] = sr
    return state, metrics, batch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=jax_enhance.models.__all__, default="ESPCN")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--log_every_n", type=int, default=5)
    cfg = parser.parse_args()
    main(cfg)

