from imagen_pytorch.dataset import CocoCaptionKarpathyDataset
from torch.utils.data import DataLoader
import IPython
test_dataset = CocoCaptionKarpathyDataset(
            data_dir="/sharefs/multimodel/lzk/adata",
            transform_keys = ["pixelbert"],
            split="train",
            image_size=384,
        )
loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    collate_fn=test_dataset.collate,
)
p = []
for batch_ndx, sample in enumerate(loader):
    if batch_ndx==1:
        break
    p = sample
IPython.embed()


from torchvision import transforms
toPIL = transforms.ToPILImage()
img = p["image"][0]
pic = toPIL(img)
pic.save('random.jpg')

from imagen_pytorch import t5
t5.t5_encode_text(t5)