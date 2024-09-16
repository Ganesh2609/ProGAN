import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm 
import matplotlib.pyplot as plt
from typing import List
import json


def gradient_penalty(discriminator:torch.nn.Module,
                     real:torch.tensor,
                     fake:torch.tensor,
                     alpha:int,
                     step:int,
                     device:torch.device):
    
    N, C, W, H = real.shape
    epsilon = torch.rand(N, 1, 1, 1).repeat(1, C, W, H).to(device)
    
    interpolated_img = (epsilon * real) + ((1-epsilon) * fake)
    mixed_score = discriminator(interpolated_img, alpha, step)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_img,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(N, -1)
    grad_norm = gradient.norm(p=2, dim=1)
    grad_penalty = torch.mean((1-grad_norm)**2)
    return grad_penalty


def train_generator(generator:torch.nn.Module,
                    discriminator:torch.nn.Module,
                    alpha:float,
                    step:int,
                    optimizer:torch.optim.Optimizer,
                    scaler:torch.cuda.amp.GradScaler,
                    BATCH_SIZE:int,
                    LATENT_DIM:int,
                    device:torch.device):
    
    generator.train()
    discriminator.eval()
    
    with torch.amp.autocast(enabled=True, device_type='cuda'):
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
        fake_img = generator(noise, alpha, step)
        fake_probs = discriminator(fake_img, alpha, step)
        loss = -(fake_probs.mean())
        
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del noise, fake_img, fake_probs
    return loss.item()


def train_discriminator(generator:torch.nn.Module,
                        discriminator:torch.nn.Module,
                        alpha:float,
                        step:int,
                        batch:tuple,
                        optimizer:torch.optim.Optimizer,
                        scaler:torch.cuda.amp.GradScaler,
                        BATCH_SIZE:int,
                        LATENT_DIM:int,
                        LAMBDA_GP:int,
                        device:torch.device):
    
    generator.eval()
    discriminator.train()
    
    real_img = batch[0].to(device)
    
    with torch.amp.autocast(enabled=True, device_type='cuda'):
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
        fake_img = generator(noise, alpha, step)
        real_probs = torch.mean(discriminator(real_img, alpha, step))
        fake_probs = torch.mean(discriminator(fake_img, alpha, step))
        gp = gradient_penalty(discriminator=discriminator, real=real_img, fake=fake_img, alpha=alpha, step=step, device=device)
        loss = fake_probs - real_probs + (LAMBDA_GP*gp)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del noise, real_img, fake_img, real_probs, fake_probs, gp
    return loss.item()


def train_models(generator:torch.nn.Module,
                 discriminator:torch.nn.Module,
                 g_optimizer:torch.optim.Optimizer,
                 d_optimizer:torch.optim.Optimizer,
                 g_scaler:torch.cuda.amp.GradScaler,
                 d_scaler:torch.cuda.amp.GradScaler,
                 START_STEP:int,
                 END_STEP:int,
                 BATCH_SIZES:List,
                 LATENT_DIM:int,
                 LAMBDA_GP:int,
                 PROGRESSIVE_EPOCHS:List,
                 FIXED_NOISE:torch.tensor,
                 ROOT_DIR:str,
                 g_path:str,
                 d_path:str,
                 result_path:str,
                 device:torch.device):
    
    JSON_PATH = result_path + '/model_loss.json'
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
        
    for step in range(START_STEP, END_STEP+1):
        
        NUM_EPOCHS = PROGRESSIVE_EPOCHS[step]
        BATCH_SIZE = BATCH_SIZES[step]
        alpha = 0
        img_size = 2 ** (step+2)
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        data = datasets.ImageFolder(root=ROOT_DIR, transform=input_transform)
        dataloader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(1, NUM_EPOCHS+1):
            
            gen_loss = 0
            disc_loss = 0
            
            with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
                for i, batch in t:
                    
                    disc_batch_loss = train_discriminator(generator=generator,
                                                          discriminator=discriminator,
                                                          alpha=alpha,
                                                          step=step,
                                                          batch=batch,
                                                          optimizer=d_optimizer,
                                                          scaler=d_scaler,
                                                          BATCH_SIZE=BATCH_SIZE,
                                                          LATENT_DIM=LATENT_DIM,
                                                          LAMBDA_GP=LAMBDA_GP,
                                                          device=device)
                    
                    gen_batch_loss = train_generator(generator=generator,
                                                     discriminator=discriminator,
                                                     alpha=alpha,
                                                     step=step,
                                                     optimizer=g_optimizer,
                                                     scaler=g_scaler,
                                                     BATCH_SIZE=BATCH_SIZE,
                                                     LATENT_DIM=LATENT_DIM,
                                                     device=device)
                    
                    disc_loss += disc_batch_loss
                    gen_loss += gen_batch_loss
                    
                    alpha += (1/len(dataloader)*NUM_EPOCHS*0.5)
                    alpha = min(alpha, 1)
                    
                    t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}] ')
                    t.set_postfix({
                        'Step' : step,
                        'Alpha' : alpha,
                        'Gen Batch Loss' : gen_batch_loss,
                        'Gen Loss' : gen_loss/(i+1),
                        'Disc Batch Loss' : disc_batch_loss,
                        'Disc Loss' : disc_loss/(i+1),
                    })
                    
                    if i % 100 == 0:
                        if g_path:
                            torch.save(obj=generator.state_dict(), f=g_path)
                        if d_path:
                            torch.save(obj=discriminator.state_dict(), f=d_path)
                        
                    if i % 100 == 0 and result_path:
                        
                        json_data['Generator Loss'].append(gen_loss/(i+1))
                        json_data['Discriminator Loss'].append(disc_loss/(i+1))
                        with open(JSON_PATH, 'w') as f:
                            json.dump(json_data, f)
                            
                        RESULT_SAVE_NAME = result_path + f'/Step_{step}_Epoch_{epoch}'
                        generator.eval()
                        discriminator.eval()
                        with torch.inference_mode():
                            fake_img = generator(FIXED_NOISE, alpha, step).cpu()
                        
                        grid = make_grid(tensor=fake_img, nrow=4, normalize=True, padding=16, pad_value=1)
                        fig = plt.figure(figsize=(9,4))
                        plt.imshow(grid.permute(1,2,0))
                        plt.title(f'Step_{step}_Epoch_{epoch}')
                        plt.axis(False);
                        plt.savefig(RESULT_SAVE_NAME)
                        plt.close(fig)
                        