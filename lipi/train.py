"""Training. One recipe for every few-shot model in the paper.

Episodic ProtoNet training: 5-way 5-shot episodes with 10 queries per class,
80 episodes per epoch, at most 100 epochs. SGD (lr 0.01, momentum 0.9, weight
decay 5e-4), learning rate divided by 10 at epochs 50 and 75. After every epoch
the model is scored on 100 fixed source-disjoint validation episodes; the best
epoch is kept and training stops after 20 epochs without improvement.

An "epoch" here is just a block of 80 episodes (6,000 images), not a pass over
the data.
"""

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from easyfsl.methods import PrototypicalNetworks
from PIL import Image
from torchvision import transforms

from .data import GlyphSet, to_input
from .episodes import accuracy
from .models import embed


@dataclass(frozen=True)
class Recipe:
    n_way: int = 5
    n_shot: int = 5
    n_query: int = 10
    episodes_per_epoch: int = 80
    max_epochs: int = 100
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_drops: tuple = (50, 75)
    patience: int = 20
    val_episodes: int = 100


RECIPE = Recipe()

# Light geometric jitter for training images (paper white is 255).
AUGMENT = transforms.Compose([
    transforms.RandomRotation(degrees=10, fill=255),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=255),
])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augmented_batch(glyphs: GlyphSet, indices) -> torch.Tensor:
    images = [np.asarray(AUGMENT(Image.fromarray(glyphs.images[i].numpy()))) for i in indices]
    return torch.from_numpy(np.stack(images))


def train_episodic(model: nn.Module, train: GlyphSet, val: GlyphSet, seed: int, device, log=print,
                   recipe: Recipe = RECIPE):
    """Train `model` in place; returns (best_state_dict, info). Deterministic given the seed."""
    set_seed(seed)
    model.to(device)
    learner = PrototypicalNetworks(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe.lr, momentum=recipe.momentum,
                                weight_decay=recipe.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(recipe.lr_drops), gamma=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    by_class = {}
    for index, label in enumerate(train.labels):
        by_class.setdefault(label, []).append(index)
    classes = sorted(by_class)
    rng = random.Random(seed)
    per_class = recipe.n_shot + recipe.n_query
    s_lab = torch.arange(recipe.n_way).repeat_interleave(recipe.n_shot).to(device)
    q_lab = torch.arange(recipe.n_way).repeat_interleave(recipe.n_query).to(device)
    no_centre = torch.zeros(1)

    best_acc, best_state, best_epoch, history = -1.0, None, 0, []
    started = time.time()
    for epoch in range(1, recipe.max_epochs + 1):
        model.train()
        losses = []
        for _ in range(recipe.episodes_per_epoch):
            picked = [rng.sample(by_class[c], per_class) for c in rng.sample(classes, recipe.n_way)]
            support = [i for p in picked for i in p[:recipe.n_shot]]
            query = [i for p in picked for i in p[recipe.n_shot:]]
            x = to_input(augmented_batch(train, support + query).to(device))
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                learner.process_support_set(x[:len(support)], s_lab)
                loss = loss_fn(learner(x[len(support):]), q_lab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        val_acc = accuracy(embed(model, val.images, device), val, "validation", "protonet", no_centre,
                           recipe.n_way, recipe.n_shot, n_query=recipe.n_query,
                           n_episodes=recipe.val_episodes)["acc"]
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            log(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val {val_acc:.2f}  (best {best_acc:.2f} at {best_epoch})")
        if epoch - best_epoch >= recipe.patience:
            log(f"no improvement for {recipe.patience} epochs, stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    info = {"recipe": asdict(recipe), "epochs_run": len(history), "best_epoch": best_epoch,
            "best_val_acc": best_acc, "train_minutes": round((time.time() - started) / 60, 1),
            "history": history}
    return best_state, info


def train_supervised(model: nn.Module, glyphs: GlyphSet, seed: int, device, epochs: int = 30, log=print):
    """Closed-set classifier for one script (the within-script baseline).

    Trains on glyph pairs 0-19 of every class and tests on pairs 20-24, so no
    glyph and no augmented copy crosses the split. Returns test accuracy in %.
    """
    set_seed(seed)
    train_idx = [i for i, pair in enumerate(glyphs.pair_ids) if pair < 20]
    test_idx = [i for i, pair in enumerate(glyphs.pair_ids) if pair >= 20]
    labels = torch.tensor(glyphs.labels)
    n_classes = len(glyphs.class_names)

    model.to(device)
    with torch.no_grad():
        dim = model(to_input(glyphs.images[:2].to(device))).shape[1]
    head = nn.Linear(dim, n_classes).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        order = train_idx[:]
        rng.shuffle(order)
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            x = to_input(augmented_batch(glyphs, batch).to(device))
            optimizer.zero_grad()
            nn.functional.cross_entropy(head(model(x)), labels[batch].to(device)).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = head(model(to_input(glyphs.images[test_idx].to(device))))
    acc = 100 * (logits.argmax(1).cpu() == labels[test_idx]).float().mean().item()
    log(f"supervised: {acc:.2f}% on {len(test_idx)} held-out images, {n_classes} classes")
    return {"acc": round(acc, 2), "classes": n_classes, "train_images": len(train_idx),
            "test_images": len(test_idx), "epochs": epochs}
