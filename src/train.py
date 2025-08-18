import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .data import CocoDataModule
from .model import JEPA


def main():
    """
    Main function to run the JEPA experiment.
    """
    # Initialize the DataModule
    data_module = CocoDataModule(batch_size=16)

    # Initialize the JEPA model
    model = JEPA(
        predictor_embed_dim=256,
    )

    # Configure the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="jepa_model")

    # Configure checkpointing to save every 10 epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="jepa-epoch={epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[checkpoint_callback],
    )

    # Start the training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
