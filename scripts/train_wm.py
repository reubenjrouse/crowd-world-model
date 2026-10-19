"""Stage 1d: train the deterministic crowd world model on synthetic scenes.

Each training window: `context` observed frames, then `horizon` imagined frames under
the known gate plan. Losses:
  density   squared error of imagined (and teacher-forced) next frames
  dest      which exit people in each cell head to (density-weighted cross-entropy)
  hurry     crowd's mean desired speed, estimated after the context

Baseline for comparison: "nothing moves" (repeat the last observed frame).

Saves to outputs/runs/<name>/:
  last.pt     full training state every --save_every steps (resume with --resume)
  best.pt     the model with the best held-out score so far
  log.jsonl   training losses every --log_every steps, held-out scores every --eval_every
Plot with scripts/plot_training.py.

Run:  python scripts/train_wm.py --data v1 --steps 20000 --name wm_v1
      python scripts/train_wm.py --data v1 --steps 20000 --name wm_v1 --resume   (after a crash)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crowdwm.data.sim_dataset import FAMILY_NAMES, SimWindows, augment, collate, make_batch
from crowdwm.model.world_model import CrowdWorldModel, load_compatible
from crowdwm.raster_torch import TorchSplatter

ROOT = Path(__file__).resolve().parents[1]


def losses(model, B, context, rollout_weight=1.0, beta=0.1, free_nats=0.02, sample=True,
           prior_step=False):
    """Training (stochastic model): z comes from the posterior, which sees the true future,
    and a KL term pulls the prior toward it. Evaluation: z is drawn from the prior
    (sample=False: prior mean).

    prior_step=True: a training step without the posterior. The model imagines with its
    own prior samples and is graded on that. Without such steps the model can route
    predictable events (e.g. scheduled arrivals) through the posterior's peek at the
    future instead of learning them, and then fails to imagine them."""
    stochastic_train = model.training and model.z_dim > 0 and not prior_step
    pred, dest_logits, hurry = model(B["density"], B["static"], B["gates"], B["exit_mask"], context,
                                     arrivals=B.get("arrivals"), use_posterior=stochastic_train,
                                     sample=sample)
    target = B["density"][:, 1:]
    err = (pred - target).pow(2).mean((2, 3, 4)) * 1000          # (B, L-1)
    teacher = err[:, :context - 1].mean()
    rollout = err[:, context - 1:].mean()

    dens = B["density"].squeeze(2)                                # (B, L, H, W)
    q = B["dest"] / dens.unsqueeze(2).clamp(min=1e-6)             # target fractions per exit
    logp = torch.log_softmax(dest_logits, 2)
    ce = -(q * logp).sum(2)                                       # (B, L, H, W)
    w = dens * (dens > 1e-3)
    dest = (ce * w).sum() / w.sum().clamp(min=1e-6)

    # NaN = no hurry label (e.g. Eindhoven). Replace it before any arithmetic: masking after
    # the fact still lets NaN reach the gradients (NaN * 0 = NaN) and poisons the weights.
    has_hurry = torch.isfinite(B["hurry"])
    diff = torch.where(has_hurry, hurry - torch.nan_to_num(B["hurry"]), torch.zeros_like(hurry))
    hurry_l = diff.pow(2).sum() / has_hurry.sum().clamp(min=1)
    total = teacher + rollout_weight * rollout + dest + 5 * hurry_l
    parts = dict(teacher=teacher.item(), rollout=rollout.item(), dest=dest.item(), hurry=hurry_l.item())
    if stochastic_train and model.kl is not None:
        # Free nats: don't penalise the first `free_nats` of information per latent value,
        # so z is used for the genuinely unpredictable part instead of collapsing to zero
        total = total + beta * torch.clamp(model.kl, min=free_nats)
        parts["kl"] = model.kl.item()
    return total, parts, (pred, dest_logits, hurry)


@torch.no_grad()
def evaluate(model, loader, splatter, length, context, max_batches=50):
    """Held-out scores. Errors are 'people misplaced' (sum of |pred - true| over the grid) at
    1 s, 4 s and the end of the horizon, next to the "nothing moves" baseline. peak_* is the
    predicted peak crowding divided by the real peak (1 = as sharp as reality)."""
    model.eval()
    horizon = length - context
    ahead = {"1s": 5, "4s": min(20, horizon), "end": horizon}
    m, by_family = {}, {}
    dest_hit = dest_all = 0.0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        B = make_batch(b, splatter, length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, (pred, dest_logits, hurry) = losses(model, B, context, sample=False)
            if model.z_dim:   # one random future, to measure how sharp individual samples are
                sampled, _, _ = model(B["density"], B["static"], B["gates"], B["exit_mask"], context,
                                      arrivals=B["arrivals"], sample=True)
        true = B["density"][:, 1:]                   # true[:, j] is frame j + 1
        last = B["density"][:, context - 1]          # last observed frame
        for name, n in ahead.items():
            j = context - 2 + n                      # n steps after the last observed frame
            err = (pred[:, j] - true[:, j]).abs().sum((1, 2, 3))
            base = (last - true[:, j]).abs().sum((1, 2, 3))
            m.setdefault(f"mae_{name}", []).append(err.mean().item())
            m.setdefault(f"persist_{name}", []).append(base.mean().item())
            peak_t = true[:, j].amax((1, 2, 3))
            ok = peak_t > 0.05
            if ok.any():
                m.setdefault(f"peak_{name}", []).append(
                    (pred[:, j].amax((1, 2, 3))[ok] / peak_t[ok]).mean().item())
                if model.z_dim:
                    m.setdefault(f"peak_sample_{name}", []).append(
                        (sampled[:, j].float().amax((1, 2, 3))[ok] / peak_t[ok]).mean().item())
            if name == "4s":
                for fam, e, p in zip(b["family"].tolist(), err.tolist(), base.tolist()):
                    by_family.setdefault(fam, [[], []])
                    by_family[fam][0].append(e)
                    by_family[fam][1].append(p)
        n_true = true[:, -1].sum((1, 2, 3))
        m.setdefault("count_err_end", []).append((pred[:, -1].sum((1, 2, 3)) - n_true).abs().mean().item())
        m.setdefault("persist_count_err_end", []).append((last.sum((1, 2, 3)) - n_true).abs().mean().item())
        dens = B["density"][:, context - 1, 0]
        correct = (dest_logits[:, context - 1].argmax(1) == B["dest"][:, context - 1].argmax(1)).float()
        dest_hit += (correct * dens).sum().item()
        dest_all += dens.sum().item()
        if torch.isfinite(B["hurry"]).any():
            m.setdefault("hurry_mae", []).append((hurry - B["hurry"]).abs().nanmean().item())
    model.train()
    out = {k: float(np.mean(v)) for k, v in m.items()}
    out["dest_acc"] = dest_hit / max(dest_all, 1e-6)
    for fam, (e, p) in by_family.items():   # < 1 means better than "nothing moves"
        out[f"score_4s_{FAMILY_NAMES[fam]}"] = float(np.mean(e) / max(np.mean(p), 1e-6))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="v1")
    p.add_argument("--name", default="wm_v1")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--context", type=int, default=10)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--eval_on_train", action="store_true",
                   help="score on the training scenes (for the can-it-overfit check)")
    p.add_argument("--resume", action="store_true", help="continue from outputs/runs/<name>/last.pt")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--init_from", default=None,
                   help="start from another run's best.pt weights (fresh optimiser and schedule)")
    p.add_argument("--no_augment", action="store_true", help="disable random rotations/mirroring")
    p.add_argument("--crop", type=float, default=0.0,
                   help="fraction of synthetic training windows seen through a smaller camera view")
    p.add_argument("--real", default=None,
                   help="real dataset(s) to mix in, comma-separated, e.g. juelich,crossing "
                        "(folders under data/processed/real)")
    p.add_argument("--real_holdout", default="",
                   help="comma-separated real run folders kept out of training (the test set)")
    p.add_argument("--real_frac", type=float, default=0.5, help="share of training windows that are real")
    p.add_argument("--z_dim", type=int, default=0, help="stochastic latent size per 16x16 cell (0 = deterministic)")
    p.add_argument("--beta", type=float, default=0.1, help="weight of the KL term (stochastic model)")
    p.add_argument("--free_nats", type=float, default=0.02, help="KL allowance per latent value")
    p.add_argument("--prior_every", type=int, default=0,
                   help="every N-th step trains without the posterior (imagine with the prior); 2 = alternate")
    p.add_argument("--prior_mean", action="store_true",
                   help="no-peek steps use the prior mean rather than a random sample (keeps futures varied)")
    a = p.parse_args()

    length = a.context + a.horizon
    root = [ROOT / "data" / "processed" / "sim" / d for d in a.data.split(",")]
    train = SimWindows(root, "train", length, stride=3, max_scenes=a.max_scenes, crop=a.crop)
    val = (SimWindows(root, "train", length, stride=10, max_scenes=a.max_scenes)
           if a.eval_on_train else SimWindows(root, "val", length, stride=10))
    print(f"train: {len(train):,} windows from {len(train.dirs)} scenes; val: {len(val):,} windows")
    sampler = None
    if a.real:
        from crowdwm.data.real_dataset import REAL_DIR
        from fnmatch import fnmatch
        held = list(filter(None, a.real_holdout.split(",")))   # names or wildcards, e.g. day7_*
        parts, weights = [train], [torch.full((len(train),), (1 - a.real_frac) / len(train))]
        names = a.real.split(",")
        for ds_name in names:   # each real dataset gets an equal share of the real fraction
            dirs = sorted(d for d in (REAL_DIR / ds_name).iterdir() if (d / "cache.npz").exists()
                          and not any(fnmatch(d.name, h) for h in held))
            real = SimWindows(dirs=dirs, length=length, stride=1)
            n_held = sum(any(fnmatch(d.name, h) for h in held) for d in (REAL_DIR / ds_name).iterdir())
            print(f"real {ds_name}: {len(real):,} windows from {len(dirs)} runs ({n_held} held out), "
                  f"{a.real_frac / len(names):.0%} of each batch")
            parts.append(real)
            weights.append(torch.full((len(real),), a.real_frac / len(names) / len(real)))
        w = torch.cat(weights)
        train = torch.utils.data.ConcatDataset(parts)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=10**8, replacement=True)
    tl = DataLoader(train, batch_size=a.batch, shuffle=sampler is None, sampler=sampler,
                    collate_fn=collate, drop_last=True, num_workers=2, persistent_workers=True)
    vl = DataLoader(val, batch_size=a.batch, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    splatter = TorchSplatter(device="cuda")
    model = CrowdWorldModel(z_dim=a.z_dim).cuda()
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
          + (f", stochastic (z: {a.z_dim} values per 16x16 cell)" if a.z_dim else ", deterministic"))
    if a.init_from and not a.resume:
        src = ROOT / "outputs" / "runs" / a.init_from / ("last.pt" if "_real" in a.init_from else "best.pt")
        resized = load_compatible(model, torch.load(src, weights_only=False)["model"])
        print(f"initialised from {src}" + (f" (expanded: {', '.join(resized)})" if resized else ""))
    aug_gen = torch.Generator().manual_seed(1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    run_dir = ROOT / "outputs" / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    step, best, running = 0, float("inf"), {}
    if a.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best, running = ck["step"], ck["best"], ck["running"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed from step {step} (best val score so far {best:.3f})")
    elif (run_dir / "log.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run; use --resume or a new --name")
    (run_dir / "args.json").write_text(json.dumps(vars(a)))
    log = open(run_dir / "log.jsonl", "a")

    def save(path, score=None):
        """Write to a temp file first so a crash mid-save can't corrupt the checkpoint."""
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "running": running, "args": vars(a),
                    "rng": torch.get_rng_state(), "score": score}, tmp)
        tmp.replace(path)

    t0, start_step, bad_batches = time.perf_counter(), step, 0
    while step < a.steps:
        for b in tl:
            B = make_batch(b, splatter, length)
            if not a.no_augment:
                B = augment(B, aug_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prior_step = a.z_dim > 0 and a.prior_every > 0 and step % a.prior_every == 0
                # --prior_mean: grade no-peek steps on the most likely future (z = prior mean).
                # Grading a single random sample instead punishes any spread and the model
                # learns to ignore z (all futures identical).
                loss, parts, _ = losses(model, B, a.context, beta=a.beta, free_nats=a.free_nats,
                                        prior_step=prior_step,
                                        sample=not (prior_step and a.prior_mean))
                if prior_step:   # log the no-peek steps separately
                    parts = {f"{k}_prior": v for k, v in parts.items() if k in ("rollout", "teacher")}
            if not torch.isfinite(loss):
                bad_batches += 1
                print(f"step {step}: non-finite loss, skipping batch ({bad_batches} in a row)", flush=True)
                opt.zero_grad(set_to_none=True)
                if bad_batches >= 20:   # the weights themselves are probably broken: stop, don't spin
                    raise SystemExit("20 non-finite batches in a row; stopping. Resume from last.pt "
                                     "after fixing the cause.")
                continue
            bad_batches = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                running[k] = 0.98 * running.get(k, v) + 0.02 * v
            if step % a.log_every == 0:
                speed = (time.perf_counter() - t0) / (step - start_step)
                log.write(json.dumps(dict(step=step, kind="train", lr=sched.get_last_lr()[0],
                                          grad_norm=float(gnorm), s_per_step=speed, **parts)) + "\n")
                log.flush()
            if step % 100 == 0:
                eta = (a.steps - step) * (time.perf_counter() - t0) / (step - start_step) / 60
                print(f"step {step:6d}  " + "  ".join(f"{k} {v:.3f}" for k, v in running.items())
                      + f"  ({(time.perf_counter() - t0) / (step - start_step):.2f} s/step, "
                        f"~{eta:.0f} min left)", flush=True)
            if step % a.save_every == 0:
                save(run_dir / "last.pt")
            if step % a.eval_every == 0 or step == a.steps:
                ev = evaluate(model, vl, splatter, length, a.context)
                score = ev["mae_4s"] / ev["persist_4s"]      # < 1 means better than "nothing moves"
                print("  eval: " + "  ".join(f"{k} {v:.3f}" for k, v in ev.items())
                      + f"  | vs nothing-moves {score:.3f}", flush=True)
                log.write(json.dumps(dict(step=step, kind="val", score=score, **ev)) + "\n")
                log.flush()
                if score < best:
                    best = score
                    save(run_dir / "best.pt", score)
                    print(f"  new best ({score:.3f}) saved", flush=True)
                save(run_dir / "last.pt")
            if step >= a.steps:
                break
    print(f"done: {step} steps in {(time.perf_counter() - t0) / 60:.1f} min; best score {best:.3f}")


if __name__ == "__main__":
    main()
