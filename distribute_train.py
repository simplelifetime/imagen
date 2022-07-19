import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.dataset import CocoCaptionKarpathyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from imagen_pytorch import t5
import os
import datetime

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
torch.distributed.init_process_group(backend="nccl", init_method='env://')

# unet for imagen
epochs = 10

tme = datetime.date.today()

def get_imagen():
    unet1 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
    )

    unet2 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)
    return Imagen(
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()

imagen = get_imagen()
# wrap imagen with the trainer class
trainer = ImagenTrainer(imagen).cuda()

test_dataset = CocoCaptionKarpathyDataset(
            data_dir="/sharefs/multimodel/lzk/adata",
            transform_keys = ["pixelbert"],
            split="train",
            image_size=384,
        )

train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    # shuffle=True,
    pin_memory=False,
    drop_last=True,
    collate_fn=test_dataset.collate,
    sampler = train_sampler
)


imagen = torch.nn.parallel.DistributedDataParallel(imagen,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

def train_one_epoch():
    for batch_ndx, sample in enumerate(tqdm(loader)):
        text_embeds, text_masks = t5.t5_encode_text(sample["text"])
        images = sample["image"]
        for i in (1, 2):
            loss = trainer(
                images,
                text_embeds = text_embeds,
                text_masks = text_masks,
                unet_number = i,
                max_batch_size = 4        # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
            )
            trainer.update(unet_number = i)

        if batch_ndx % 500 == 0:
            print(f"avg_loss={loss}")
        
        if batch_ndx % 2000 == 0:
            save_path = "./logs/" + str(tme) + "/epochs_" + str(batch_ndx) + ".pt"
            trainer.save(save_path)
            print(f"save epochs to {save_path}")
        
for i in range(epochs):
    train_one_epoch()
