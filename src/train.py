import pytorch_lightning as pl
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

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )

    # Start the training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
