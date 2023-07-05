import torch
import dnnlib
import legacy
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
epsilon = 1e-4

with dnnlib.util.open_url("out/stylegan3-r-afhqv2-512x512.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
    z = torch.zeros([1, G.z_dim])
    z[0][0] -= 5 * epsilon

    fig, axes = plt.subplots(1, 11)
    for i in range(11):
        print(i)
        z[0][0] += epsilon
        img = G(z, None)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.show()
