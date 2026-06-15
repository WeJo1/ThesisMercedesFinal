import torch
import lpips
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='vgg').to(device)

# Standard-Transformation: Bild laden, Größe anpassen, zu Tensor [0, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class SimpleImageDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return transform(img)

def calibrate_lpips(folder_path):
    """Berechnet das 95-Perzentil der Distanz zwischen zufälligen Bildern im Ordner."""
    dataset = SimpleImageDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    all_dists = []
    
    print(f"-> Starte Kalibrierung mit Bildern aus: {folder_path}")
    
    with torch.no_grad():
        for batch in dataloader:
            if batch.size(0) < 2: continue
            
            # Bilder auf [-1, 1] bringen
            img1 = batch[:-1].to(device) * 2.0 - 1.0
            img2 = batch[1:].to(device) * 2.0 - 1.0
            
            dist = loss_fn(img1, img2)
            all_dists.append(dist.view(-1).cpu().numpy())
    
    all_dists = np.concatenate(all_dists)
    d_max = np.percentile(all_dists, 95)
    return d_max

def compare_two_images(path_a, path_b, d_max):
    """Vergleicht zwei spezifische Bilder und gibt die %-Ähnlichkeit aus."""
    img_a = transform(Image.open(path_a).convert('RGB')).unsqueeze(0).to(device) * 2.0 - 1.0
    img_b = transform(Image.open(path_b).convert('RGB')).unsqueeze(0).to(device) * 2.0 - 1.0
    
    with torch.no_grad():
        dist = loss_fn(img_a, img_b).item()
    
    # Formel: Ähnlichkeit = (1 - (aktuelle_dist / maximal_dist)) * 100
    similarity = max(0, (1 - (dist / d_max)) * 100)
    return dist, similarity

# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    # 1. SCHRITT: Ordner mit Referenzbildern (dein Datensatz)
    # Ändere diesen Pfad zu deinem Bilder-Ordner!
    my_data_folder = "./reference" 
    
    if os.path.exists(my_data_folder):
        val_max = calibrate_lpips(my_data_folder)
        print(f"\nERGEBNIS KALIBRIERUNG:")
        print(f"Dein d_max Wert ist: {val_max:.4f}")
        print("-" * 30)
        
        # 2. SCHRITT: Test-Vergleich
        # Hier Pfade zu zwei Bildern einfügen, die du vergleichen willst
        test_img1 = "reference/cla.png"
        test_img2 = "Generated/cla.png"
        
        if os.path.exists(test_img1) and os.path.exists(test_img2):
            d, sim = compare_two_images(test_img1, test_img2, val_max)
            print(f"Vergleich: {test_img1} vs {test_img2}")
            print(f"LPIPS Distanz: {d:.4f}")
            print(f"Berechnete Ähnlichkeit: {sim:.2f}%")
        else:
            print("Info: Testbilder nicht gefunden. Passe die Pfade im Code an.")
    else:
        print(f"Fehler: Ordner '{my_data_folder}' nicht gefunden!")