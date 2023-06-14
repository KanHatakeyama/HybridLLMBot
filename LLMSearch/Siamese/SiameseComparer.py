import torch
from sklearn.model_selection import train_test_split
from .SiameseNetwork import SiameseNetwork
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MemoryEfficientDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.labels[index]

    def __len__(self):
        return len(self.data1)

class SiameseComparer:
    def __init__(self,encoder):
        self.encoder=encoder

    def prepare_model(self,
                    data1,
                    data2,
                    labels,
                    val_ratio=0.1,
                    hidden_dim=256,
                    batch_size=32,
    ):
        print("encoding data1")
        vec1=[self.encoder(text) for text in tqdm(data1)]
        print("encoding data2")
        vec2=[self.encoder(text) for text in tqdm(data2)]

        print("preparing tensor")
        data1=torch.tensor(vec1)
        data2=torch.tensor(vec2)
        labels=torch.tensor(labels)
        input_dim=data1.shape[1]

        print("splitting data")        
        train_data1,valid_data1,train_data2,valid_data2,train_labels,valid_labels=train_test_split(
                data1,data2,labels,test_size=val_ratio,random_state=42)

        print("preparing dataloader")
        #train_dataset = torch.utils.data.TensorDataset(train_data1, train_data2, train_labels)
        train_dataset=MemoryEfficientDataset(train_data1, train_data2, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

        #val_dataset = torch.utils.data.TensorDataset(valid_data1, valid_data2, valid_labels)
        val_dataset=MemoryEfficientDataset(valid_data1, valid_data2, valid_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        print("preparing model and fit")
        model = SiameseNetwork(input_dim=input_dim,hidden_dim=hidden_dim)

        early_stop_callback = EarlyStopping(monitor='val_loss',patience=5)
        trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback])
        trainer.fit(model, train_loader, val_loader)

        self.model=model

    def compare(self,text1,text2):

        if type(text1)==str:
            text1=[text1]
            text2=[text2]

        input1=[self.encoder(text) for text in text1]
        input2=[self.encoder(text) for text in text2]

        input1=torch.tensor(input1)#.reshape(1,-1)
        input2=torch.tensor(input2)#.reshape(1,-1)


        self.model.eval()


        # 推論
        with torch.no_grad():
            output1, output2 = self.model(input1, input2)

        # コサイン類似度の計算
        cosine_similarity = F.cosine_similarity(output1, output2).numpy()
        return cosine_similarity