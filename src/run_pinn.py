"""Orquestador de la PINN."""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("pinn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--split-mode", default="random", choices=["random", "spatial"],
        help="Tipo de split train/val/test (random o spatial)")
    args = parser.parse_args()

    from src.pinn.train import train_pinn

    LOG.info("=" * 60)
    LOG.info("PINN-UTCI · epochs=%d · batch=%d · lr=%.1e · samples=%d",
             args.epochs, args.batch_size, args.lr, args.n_samples)
    LOG.info("=" * 60)

    t0 = time.time()
    train_pinn(epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, n_samples=args.n_samples,
        split_mode=args.split_mode)
    LOG.info("Entrenamiento terminado en %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()