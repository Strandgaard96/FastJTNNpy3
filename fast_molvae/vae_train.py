import sys

sys.path.append("../")
import argparse
import math
import os
import pickle as pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_jtnn import *


def main_vae_train(
    train,
    vocab,
    save_dir,
    load_epoch=0,
    hidden_size=450,
    batch_size=32,
    latent_size=56,
    depthT=20,
    depthG=3,
    lr=1e-3,
    clip_norm=50.0,
    beta=0.0,
    step_beta=0.002,
    max_beta=1.0,
    warmup=1000,
    epoch=100,
    anneal_rate=0.9,
    anneal_iter=1000,
    kl_anneal_iter=500,
    print_iter=50,
    save_iter=1000,
):
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    model = JTNNVAE(
        vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG)
    ).cuda()
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.epoch-" + str(load_epoch)))

    print(
        (
            "Model #Params: %dK"
            % (sum([x.nelement() for x in model.parameters()]) / 1000,)
        )
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)

    def param_norm(m):
        return math.sqrt(sum([(p.norm().item() ** 2) for p in m.parameters()]))

    def grad_norm(m):
        return math.sqrt(
            sum(
                [
                    (p.grad.norm().item() ** 2)
                    for p in m.parameters()
                    if p.grad is not None
                ]
            )
        )

    total_step = load_epoch
    meters = np.zeros(4)

    print("Before epoch")
    for epoch in tqdm(list(range(epoch))):
        print(f"Epoch: {epoch}")
        loader = MolTreeFolder(train, vocab, batch_size)  # , num_workers=4)
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            except Exception as e:
                print(f"Batch {total_step} failed")
                print(e)
                continue

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

            if total_step % print_iter == 0:
                meters /= print_iter
                print(
                    (
                        "[%d] Loss: %.3f,Beta: %.3f,KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f"
                        % (
                            total_step,
                            loss.item(),
                            beta,
                            meters[0],
                            meters[1],
                            meters[2],
                            meters[3],
                            param_norm(model),
                            grad_norm(model),
                        )
                    )
                )
                print(f"Loss: {loss.item()}")
                sys.stdout.flush()
                meters *= 0

            if total_step % save_iter == 0:
                torch.save(
                    model.state_dict(), save_dir + "/model.iter-" + str(total_step)
                )

            if total_step % anneal_iter == 0:
                scheduler.step()
                print(("learning rate: %.6f" % scheduler.get_lr()[0]))

            if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                beta = min(max_beta, beta + step_beta)
    #         torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))
    torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(total_step))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--load_epoch", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--step_beta", type=float, default=0.002)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--kl_anneal_iter", type=int, default=300)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--save_iter", type=int, default=1000)

    args = parser.parse_args()
    print(args)

    main_vae_train(
        args.train,
        args.vocab,
        args.save_dir,
        args.load_epoch,
        args.hidden_size,
        args.batch_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.lr,
        args.clip_norm,
        args.beta,
        args.step_beta,
        args.max_beta,
        args.warmup,
        args.epoch,
        args.anneal_rate,
        args.anneal_iter,
        args.kl_anneal_iter,
        args.print_iter,
        args.save_iter,
    )
