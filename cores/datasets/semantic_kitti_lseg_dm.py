import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

class SemanticKITTILsegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        data_root,
        ann_file,
        pred_model,
        batch_size=1,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = dataset

        self.data_root = data_root
        self.ann_file = ann_file
        self.pred_model = pred_model

        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.predict_dataset = self.dataset(self.data_root, self.ann_file, self.pred_model, 'predict')
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)