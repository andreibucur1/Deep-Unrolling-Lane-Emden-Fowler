import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from torch.utils.data import Dataset, DataLoader
import time

from model import LEFNet, get_potentials

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma_test = 0.09

model_eval = LEFNet(No_iterations=15).to(device)
model_eval.load_state_dict(torch.load("lefnet_bsds300_global.pth", map_location=device))
model_eval.eval()

transform_to_tensor = transforms.ToTensor()


def apply_nlm_uint8(img_uint8):
    return cv2.fastNlMeansDenoisingColored(img_uint8, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21)


def apply_median_uint8(img_uint8, ksize=5):
    return cv2.medianBlur(img_uint8, ksize=ksize)



def run_hjb_filter(image_np, iterations=10, dt=0.05):
    img_f = image_np.astype(np.float32) / 255.0
    for _ in range(iterations):
        for c in range(img_f.shape[2]):
            u = img_f[:, :, c]
            dy, dx = np.gradient(u)
            dyy, dyx = np.gradient(dy)
            dxy, dxx = np.gradient(dx)
            grad_mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
            u_nn = (dx ** 2 * dxx + 2 * dx * dy * dxy + dy ** 2 * dyy) / (grad_mag ** 2)
            u -= dt * np.sign(u_nn) * grad_mag
            img_f[:, :, c] = u
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)



folder_test = "test"
test_images_paths = glob.glob(os.path.join(folder_test, "*.jpg"))
num_images = len(test_images_paths)

if num_images == 0:
    raise FileNotFoundError(f"No test images found in the '{folder_test}' directory.")

print(f"Starting testing on the independent dataset ({num_images} images)...")


results = {
    "Noisy": {"mse": [], "psnr": [], "ssim": []},
    "NLM": {"mse": [], "psnr": [], "ssim": []},
    "Median": {"mse": [], "psnr": [], "ssim": []},
    "LEFNet": {"mse": [], "psnr": [], "ssim": []},
    "LEFNet+HJB": {"mse": [], "psnr": [], "ssim": []}
}

start_time = time.time()

with torch.no_grad():
    for i, img_path in enumerate(test_images_paths):
        img_pil = Image.open(img_path).convert("RGB")
        img_clean_tensor = transform_to_tensor(img_pil).unsqueeze(0).to(device)

        torch.manual_seed(42 + i)
        noise = sigma_test * torch.randn_like(img_clean_tensor)
        img_noisy_tensor = torch.clamp(img_clean_tensor + noise, 0.0, 1.0)

        _, _, H, W = img_noisy_tensor.shape
        p_test, q_test = get_potentials(H, W, device)

        img_lefnet_tensor = model_eval(img_noisy_tensor, p_test, q_test)
        img_lefnet_tensor = torch.clamp(img_lefnet_tensor, 0.0, 1.0)

        clean_np_uint8 = (img_clean_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        noisy_np_uint8 = (img_noisy_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lefnet_np_uint8 = (img_lefnet_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        lefnet_hjb_np_uint8 = run_hjb_filter(lefnet_np_uint8, iterations=10)

        noisy_bgr = cv2.cvtColor(noisy_np_uint8, cv2.COLOR_RGB2BGR)

        nlm_bgr = apply_nlm_uint8(noisy_bgr)
        nlm_np_uint8 = cv2.cvtColor(nlm_bgr, cv2.COLOR_BGR2RGB)

        median_bgr = apply_median_uint8(noisy_bgr)
        median_np_uint8 = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2RGB)

        gt = clean_np_uint8.astype(np.float64) / 255.0


        def calc_metrics(test_img_uint8):
            test_f = test_img_uint8.astype(np.float64) / 255.0
            m = np.mean((gt - test_f) ** 2)
            p = calculate_psnr(gt, test_f, data_range=1.0)
            s = calculate_ssim(gt, test_f, data_range=1.0, channel_axis=-1)
            return m, p, s


        m_n, p_n, s_n = calc_metrics(noisy_np_uint8)
        results["Noisy"]["mse"].append(m_n)
        results["Noisy"]["psnr"].append(p_n)
        results["Noisy"]["ssim"].append(s_n)

        m_nlm, p_nlm, s_nlm = calc_metrics(nlm_np_uint8)
        results["NLM"]["mse"].append(m_nlm)
        results["NLM"]["psnr"].append(p_nlm)
        results["NLM"]["ssim"].append(s_nlm)

        m_med, p_med, s_med = calc_metrics(median_np_uint8)
        results["Median"]["mse"].append(m_med)
        results["Median"]["psnr"].append(p_med)
        results["Median"]["ssim"].append(s_med)

        m_lef, p_lef, s_lef = calc_metrics(lefnet_np_uint8)
        results["LEFNet"]["mse"].append(m_lef)
        results["LEFNet"]["psnr"].append(p_lef)
        results["LEFNet"]["ssim"].append(s_lef)

        m_hjb, p_hjb, s_hjb = calc_metrics(lefnet_hjb_np_uint8)
        results["LEFNet+HJB"]["mse"].append(m_hjb)
        results["LEFNet+HJB"]["psnr"].append(p_hjb)
        results["LEFNet+HJB"]["ssim"].append(s_hjb)

        print(f"Processed {i + 1}/{num_images} images...")

total_time = time.time() - start_time
print(f"\nTesting completed in {total_time:.1f} seconds.")

print("\n" + "=" * 90)
print(f"{'Method':<16} | {'MSE (Mean ± Std)':<22} | {'PSNR (dB) (Mean ± Std)':<25} | {'SSIM (Mean ± Std)':<20}")
print("-" * 90)

for metoda in ["Noisy", "Median", "NLM", "LEFNet", "LEFNet+HJB"]:
    avg_mse = np.mean(results[metoda]["mse"])
    std_mse = np.std(results[metoda]["mse"])

    avg_psnr = np.mean(results[metoda]["psnr"])
    std_psnr = np.std(results[metoda]["psnr"])

    avg_ssim = np.mean(results[metoda]["ssim"])
    std_ssim = np.std(results[metoda]["ssim"])

    print(
        f"{metoda:<16} | {avg_mse:.5f} ± {std_mse:.5f} | {avg_psnr:.4f} ± {std_psnr:.4f}   | {avg_ssim:.4f} ± {std_ssim:.4f}")

print("=" * 90)
print(f"Note: Evaluation performed on {num_images} independent images (BSDS300 Test)")