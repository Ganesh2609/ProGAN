import torch 
from torch import nn 
import torch.nn.functional as F
    

class WSConv2d(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=1, gain:int=2, transpose:bool=False):
        super(WSConv2d, self).__init__()
        self.conv = (
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            if transpose else
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        self.bias = self.conv.bias
        self.conv.bias = None
        self.scale = (gain / (in_channels * (kernel_size**2)) ) ** 0.5
    
    def forward(self, x):
        return self.conv(self.scale + x) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(  torch.mean(x**2, dim=1, keepdim=True) + self.epsilon )


class ConvBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, use_pixel_norm:bool=True):
        super(ConvBlock, self).__init__()
        self.use_pixel_norm = use_pixel_norm
        self.layers = nn.Sequential(
            self.conv_block(in_channels=in_channels, out_channels=out_channels),
            self.conv_block(in_channels=out_channels, out_channels=out_channels)
        )
    
    def conv_block(self, in_channels:int, out_channels:int):
        return nn.Sequential(
            WSConv2d(in_channels=in_channels, out_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            PixelNorm() if self.use_pixel_norm else nn.Identity()
        )
    
    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, factors=[1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]):
        super(Generator, self).__init__()
        self.initial_conv = nn.Sequential(
            PixelNorm(),
            WSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0, transpose=True),
            nn.LeakyReLU(),
            PixelNorm(),
            WSConv2d(in_channels=in_channels, out_channels=in_channels),
            nn.LeakyReLU(),
            PixelNorm()
        )
        initial_rgb = WSConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
        self.prog_blocks, self.rgb_blocks = nn.ModuleList(), nn.ModuleList([initial_rgb])
        for i in range(len(factors)-1):
            c_in = int(in_channels*factors[i])
            c_out = int(in_channels*factors[i+1])
            self.prog_blocks.append(
                ConvBlock(in_channels=c_in, out_channels=c_out)
            )
            self.rgb_blocks.append(
                WSConv2d(in_channels=c_out, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, generated, upscaled):
        return torch.tanh((alpha * generated) + ((1-alpha) * upscaled))
        
    def forward(self, x, alpha, steps): 
        out = self.initial_conv(x)
        if steps == 0:
            return self.rgb_blocks[0](out)
        
        for i in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[i](upscaled)
        
        final_upscaled = self.rgb_blocks[steps-1](upscaled)
        final_generated = self.rgb_blocks[steps](out)
        final__out = self.fade_in(alpha=alpha, generated=final_generated, upscaled=final_upscaled)
        return final__out         
        


class Discriminator(nn.Module):
    
    def __init__(self, latent_channels:int, img_channels:int, factors=[1/32, 1/16, 1/8, 1/4, 1/2, 1, 1, 1, 1]):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.from_rgb_block = nn.ModuleList(), nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        for i in range(len(factors)-1):
            c_in = int(factors[i]*latent_channels)
            c_out = int(factors[i+1]*latent_channels)
            self.from_rgb_block.append(WSConv2d(in_channels=img_channels, out_channels=c_in, kernel_size=1, stride=1, padding=0))
            self.prog_blocks.append(ConvBlock(in_channels=c_in, out_channels=c_out, use_pixel_norm=False))
        
        final_rgb = WSConv2d(in_channels=img_channels, out_channels=latent_channels, kernel_size=1, stride=1, padding=0)
        self.from_rgb_block.append(final_rgb)
    
        self.final_conv = nn.Sequential(
            WSConv2d(in_channels=latent_channels+1, out_channels=latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            WSConv2d(in_channels=latent_channels, out_channels=latent_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            WSConv2d(in_channels=latent_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        
    
    def fade_in(self, alpha, downscaled, out):
        return (alpha * out) + ((1-alpha) * downscaled)
    
    def minibatch_std(self, x):
        std = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)
        
    def forward(self, x, alpha, steps):
        curr_step = len(self.prog_blocks) - steps
        out = self.leaky_relu(self.from_rgb_block[curr_step](x))
        
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_conv(out).view(out.shape[0], -1)
    
        downscaled = self.leaky_relu(self.from_rgb_block[curr_step+1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[curr_step](out))
        out = self.fade_in(alpha=alpha, downscaled=downscaled, out=out)
        
        for i in range(curr_step+1, len(self.prog_blocks)):
            out = self.avg_pool(self.prog_blocks[i](out))
        
        out = self.minibatch_std(out)
        return self.final_conv(out).view(out.shape[0], -1)