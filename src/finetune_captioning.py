import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import CocoDataModule
from src.captioning_model import ImageCaptioningModel


def main(args):
    """
    Main function to run the fine-tuning process.
    """
    # 1. Setup Data Module
    datamodule = CocoDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 2. Setup Model from checkpoint
    model = ImageCaptioningModel(
        jepa_checkpoint_path=args.jepa_checkpoint_path,
        learning_rate=args.learning_rate
    )

    # 3. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="tb_logs/captioning_model",
        filename="best-checkpoint-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=False,
        mode="min"
    )

    # 4. Setup Logger
    logger = TensorBoardLogger("tb_logs", name="captioning_model")

    # 5. Setup Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # 6. Start Fine-tuning
    print("Starting model fine-tuning...")
    trainer.fit(model, datamodule)
    print("Fine-tuning finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an Image Captioning Model using a pre-trained JEPA encoder.")

    # Required arguments
    parser.add_argument(
        "--jepa_checkpoint_path",
        type=str,
        required=True,
        help="Path to the pre-trained JEPA model checkpoint (.ckpt file).",
    )

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    # Hardware parameters
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["cpu", "gpu", "mps", "auto"],
        help="Hardware accelerator to use ('cpu', 'gpu', 'mps', 'auto')."
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use (e.g., GPUs).")

    args = parser.parse_args()
    main(args)
