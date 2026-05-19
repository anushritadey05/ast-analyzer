from __future__ import annotations

import argparse
import random
from typing import List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .dataset import BIG_O_CLASSES, Sample, SyntheticDatasetGenerator
from .gnn_dataset import BigODataset
from .model import BaselineGNN


def collate_graphs(samples):
    data_list = [s.data for s in samples]
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    batch = Batch.from_data_list(data_list)
    return batch, labels


def split_samples(samples: List[Sample], train_ratio: float = 0.8):
    random.shuffle(samples)
    split = int(len(samples) * train_ratio)
    return samples[:split], samples[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline GNN for Big-O prediction")
    parser.add_argument("--count", type=int, default=500, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    gen = SyntheticDatasetGenerator()
    samples = gen.generate(args.count)
    train_samples, val_samples = split_samples(samples)

    train_ds = BigODataset(train_samples)
    val_ds = BigODataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    num_node_types = len(train_ds.converter.node_type_vocab)
    num_classes = len(BIG_O_CLASSES)

    model = BaselineGNN(num_node_types=num_node_types, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch, labels in train_loader:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, labels in val_loader:
                logits = model(batch)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{args.epochs} loss={total_loss:.4f} val_acc={acc:.3f}")


if __name__ == "__main__":
    main()
