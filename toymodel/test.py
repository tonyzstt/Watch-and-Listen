import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data import VideoDataset
from train import VideoUnderstandingModel

def inference(test_data, ckpt):
    dataset = VideoDataset(
        root_dir=test_data,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = VideoUnderstandingModel().cuda()
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    for images, audio, texts in dataloader:
        images = images.cuda()
        audio = audio.cuda()
        with torch.no_grad():
            loss, generated_texts = model(images, audio, texts=None)

        for gen, text in zip(generated_texts, texts):
            print("=" * 100)
            print("Generated:", gen)
            print("GT:", text)
            print("=" * 100)

if __name__ == "__main__":
    inference("../data/test", "ckpts/checkpoint_epoch_54.pth")