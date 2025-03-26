import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

class SemanticKITTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        data_root,
        ann_file,
        pred_model,
        vlm_model=None,
        batch_size=1,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = dataset

        self.data_root = data_root
        self.ann_file = ann_file
        self.pred_model = pred_model
        self.vlm_model = vlm_model
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = self.dataset(self.data_root, self.ann_file, self.pred_model, 'train', vlm_model=self.vlm_model)
        self.test_dataset = self.dataset(self.data_root, self.ann_file, self.pred_model, 'val', vlm_model=self.vlm_model)
        self.val_dataset = self.dataset(self.data_root, self.ann_file, self.pred_model, 'test', vlm_model=self.vlm_model)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)
    