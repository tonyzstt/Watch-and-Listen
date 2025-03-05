import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from audio import AudioEncoder
from vision import VisionEncoder
from text import TextDecoder
from cross_att import CrossAttention
from data import VideoDataset

class VideoUnderstandingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.audio_encoder = AudioEncoder()
        self.cross_attention = CrossAttention()
        self.text_decoder = TextDecoder(model_name="gpt2", embed_dim=768, prefix_len=10)

    def forward(self, images, audio, texts=None):

        v_feats = self.vision_encoder(images)
        a_feats = self.audio_encoder(audio)
        fused = self.cross_attention(v_feats, a_feats)
        loss, generated = self.text_decoder(fused, input_texts=texts)

        return loss, generated

def train(train_data):

    dataset = VideoDataset(
        root_dir=train_data,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = VideoUnderstandingModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, audio, texts in dataloader:
            images = images.cuda()   
            audio = audio.cuda() 

            optimizer.zero_grad()

            loss, _ = model(images, audio, texts=texts)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    print("Training complete!")



if __name__ == "__main__":
    train("../data/train")