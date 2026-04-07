
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr

# ==========================================
# CHARGEMENT DES DONNÉES (Xarray)
# ==========================================

"""
Dossier de données intéresant: 
-"Mes_Scans_AD3/Scan_20260331_140002/donnees_completes.nc"
-"Mes_Scans_AD3/Scan_20260331_153011/donnees_completes.nc" avec une pointe de chauffe froide
-"Mes_Scans_AD3/Scan_20260324_170009/donnees_completes.nc"
-"Mes_Scans_AD3/Scan_20260402_142109/donnees_completes.nc"
"""
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260324_170009/donnees_completes.nc"
ds = xr.open_dataset(chemin_fichier_nc, engine="netcdf4")

mat = ds["signal_mesure"].values        # Matrice 3D (X, Y, Temps)
signal_emetteur = ds["signal_source"].values # Matrice 3D (X, Y, Temps)
x_value = ds["x"].values 
y_value = ds["y"].values
t = ds["temps"].values

num_time_steps = len(t)
nb_x = len(x_value)
nb_y = len(y_value)

sample_frequency = ds.attrs.get("sample_frequency_Hz", num_time_steps)

print(f"Chargement terminé : Grille {nb_x}x{nb_y}, {num_time_steps} points temporels.")

vx, vy = np.meshgrid(x_value, y_value, indexing='ij')

#%%

# ==========================================
# CALCUL DE L'ANALYSE MODALE
# ==========================================

nb_aver = 3
step_time = num_time_steps // nb_aver
demi_n = step_time // 2  

# Création de l'axe des fréquences (uniquement positives)
freqs = np.fft.fftfreq(step_time, d=1/sample_frequency)

# On applique une fenêtre de Hanning sur chaque tronçon pour éviter le "leakage"
window = np.hanning(step_time)

# 1. Pré-allocation des matrices d'accumulation (Finis les tableaux 4D géants !)
Sxx = np.zeros((nb_x, nb_y, step_time), dtype=np.float64)
Sxy = np.zeros((nb_x, nb_y, step_time), dtype=complex)
rep_source_accum = np.zeros((nb_x, nb_y, step_time), dtype=np.float64)

for i in range(nb_aver):
    # Extraction du tronçon et application de la fenêtre spatio-temporelle
    mesure_troncon = mat[:, :, i*step_time:(i+1)*step_time] * window
    source_troncon = signal_emetteur[:, :, i*step_time:(i+1)*step_time] * window

    # FFT sur l'axe du temps (axis=2) et troncature à demi_n
    fft_mesure = np.fft.fft(mesure_troncon, n=step_time, axis=2)
    fft_source = np.fft.fft(source_troncon, n=step_time, axis=2)

    # 2. Accumulation à la volée (on ajoute au total)
    Sxx += np.real(np.conjugate(fft_source) * fft_source)
    Sxy += np.conjugate(fft_source) * fft_mesure
    rep_source_accum += np.abs(fft_source)
    
    # 3. Nettoyage manuel de la RAM pour supprimer ces gros tableaux temporaires
    del mesure_troncon, source_troncon, fft_mesure, fft_source

# 4. Division finale pour obtenir la moyenne
Sxx /= nb_aver
Sxy /= nb_aver
rep_source_mean = (rep_source_accum / nb_aver) * 2 / step_time

# Fonction de Transfert (Estimateur H1) : H = Sxy / Sxx
# On ajoute 1e-12 pour éviter une division par zéro mathématique
H = Sxy / (Sxx)

# L'amplitude de la déformée modale P est la valeur absolue de la fonction de transfert
P = np.real(H)

print(f"Shape de la matrice de transfert P : {P.shape}")

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep = np.mean(np.abs(P), axis=(0, 1))
rep_source_mean_1D = np.mean(rep_source_mean, axis=(0, 1))

#%%

#### Figure interactive

#%matplotlib inline

fig_fft2D = plt.figure(figsize=(10, 6))
spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig_fft2D, height_ratios=[5, 1])

# --- Graphe du haut : Carte de chaleur ---
ax_fft2D = fig_fft2D.add_subplot(spec[0, 0])
idx_freq_initial = 100 

extent_physique = (x_value.min(), x_value.max(), y_value.min(), y_value.max())



from matplotlib.colors import LinearSegmentedColormap

colors = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']
cmap_name = '9 class spectral'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

im = ax_fft2D.imshow(P[:, :, idx_freq_initial].T, 
                     extent=extent_physique, 
                     origin='lower', aspect='auto', cmap="jet", interpolation="bicubic",
                     vmin = -np.pi, vmax = np.pi
                     )

cbar = fig_fft2D.colorbar(im, ax=ax_fft2D)
cbar.set_label("Fonction de transfert |H|")

ax_fft2D.set_title(f"Amplitude spatiale à Freq = {freqs[idx_freq_initial]:.1f} Hz")
ax_fft2D.set_xlabel("Position X (mm)")
ax_fft2D.set_ylabel("Position Y (mm)")

# --- Graphe du bas : Spectre global ---
ax_fft2D_mean = fig_fft2D.add_subplot(spec[1, 0])

ax_fft2D_mean.loglog(freqs, rep, color='black', label='Plaque (Transfert H1)')
ax_fft2D_mean.loglog(freqs, rep_source_mean_1D, color='green', linestyle='--', label='Source Brute')

ax_fft2D_mean.set_title("Maintenez le clic sur la barre rouge pour la glisser le long du spectre")
ax_fft2D_mean.set_xlabel("Fréquence (Hz)")
ax_fft2D_mean.set_ylabel("Amplitude")
ax_fft2D_mean.grid(True, which="both", ls="--", alpha=0.5)
ax_fft2D_mean.legend(loc="upper right") 

# Création de la ligne rouge avec picker=5 (Tolérance de 5 pixels pour l'attraper)
vline = ax_fft2D_mean.axvline(x=freqs[idx_freq_initial], color='red', linestyle='-', linewidth=2, picker=5)

# --- INTERACTION : Classe de Glisser-Déposer ---
class DraggableLine:
    def __init__(self, line):
        self.line = line
        self.is_dragging = False
        self.canvas = line.figure.canvas
        
        # Connexion des événements de la souris
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        # Vérifie si on a cliqué sur la ligne (ou à moins de 5 pixels)
        contains, _ = self.line.contains(event)
        if contains:
            self.is_dragging = True

    def on_release(self, event):
        # On lâche la ligne
        self.is_dragging = False

    def on_motion(self, event):
        # Si on ne tient pas la ligne, ou qu'on sort du graphe, on annule
        if not self.is_dragging or event.inaxes != ax_fft2D_mean:
            return
            
        mouse_freq = event.xdata
        if mouse_freq is None:
            return
            
        # On trouve l'indice de la vraie fréquence la plus proche
        idx = np.argmin(np.abs(freqs - mouse_freq))
        freq_cible = freqs[idx]
        
        # 1. Mise à jour de la position de la ligne rouge
        self.line.set_xdata([freq_cible, freq_cible])
        
        # 2. Mise à jour de la carte de chaleur 2D
        im.set_data(P[:, :, idx].T)
        ax_fft2D.set_title(f"Amplitude spatiale à Freq = {freq_cible:.1f} Hz")
        
        # 3. Redessine l'écran
        self.canvas.draw_idle()

# On active notre classe sur la ligne rouge
drag_logic = DraggableLine(vline)

plt.show()

# %%
