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
-"Mes_Scans_AD3/Scan_20260409_151849/donnees_completes.nc" : plaque 240*300
"""
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260409_151849/donnees_completes.nc"
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
H = Sxy / Sxx

# L'amplitude de la déformée modale P est la valeur absolue de la fonction de transfert
P = np.abs(H)

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep = np.sqrt(np.mean(P**2, axis=(0, 1)))
rep_source_mean_1D = np.mean(rep_source_mean, axis=(0, 1))

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
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260413_151410/donnees_completes.nc"
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
nb_aver = 5
step_time = num_time_steps // nb_aver
demi_n_h = step_time // 2  

# Création de l'axe des fréquences (uniquement positives)
freqs_h = np.fft.fftfreq(step_time, d=1/sample_frequency)

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
H = Sxy / Sxx

# L'amplitude de la déformée modale P est la valeur absolue de la fonction de transfert
P = np.abs(H)

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep_h = np.sqrt(np.mean(P**2, axis=(0, 1)))
rep_source_mean_1D_h = np.mean(rep_source_mean, axis=(0, 1))

#%%
#Création de la figure
import matplotlib
matplotlib.use('Qt5Agg')

fft_sup = plt.figure(figsize=(12, 6))
ax = fft_sup.add_subplot(1, 1, 1) 

#%%
#Affichage de la plaque vertical
ax.loglog(freqs[5:demi_n], rep[5:demi_n], label="Réponse modale verticale", color='black')

#%%
#Affichage de la plaque horizontal
ax.loglog(freqs_h[20:demi_n_h], rep_h[20:demi_n_h], label="Réponse modale horizontale (Sxx)", color='orange')

#%%
#Legende de la figure

ax.set_title("Réponse modale et de la source en fonction de la fréquence")
ax.set_xlabel("Fréquence (Hz)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid()
plt.tight_layout()

# %%
#Trouvé les pics
import scipy.signal as signal

# Détection des pics dans la réponse modale verticale
peaks_v, _ = signal.find_peaks(rep[5:demi_n],prominence=0.1, width=1)  # Ajustez le seuil de hauteur selon vos données
peaks_h, _ = signal.find_peaks(rep_h[5:demi_n_h], prominence=0.2, width=1)

# %%
# Pic affichage : Freqences des modes propres de la plaque verticale
plt.loglog(freqs[5:demi_n][peaks_v], rep[5:demi_n][peaks_v], 'x', label='Pics plaque verticale', color='red')

#%%
# Pic affichage : Freqences des modes propres de la plaque horizontale
plt.loglog(freqs_h[5:demi_n_h][peaks_h], rep_h[5:demi_n_h][peaks_h], 'x', label='Pics plaque horizontale', color='black')
plt.legend()

#%%
# Affichage des fréquences MEF

freq_mef = [np.float64(45.558754277369374), np.float64(56.28676782150183), np.float64(94.85772877245577), np.float64(109.70305103988925), np.float64(128.7427450927569), np.float64(167.28879135078958), np.float64(213.16803826622947), np.float64(218.12810452912768)]
for f in freq_mef:
    plt.axvline(f, color='green', linestyle='--', alpha=0.5, label='Fréquences MEF' if f == freq_mef[0] else "")



#%%

### Pic : Affichage des différences
if False :
    dif_freqs = freqs[5:demi_n][peaks_v][:8] - freqs_h[5:demi_n_h][peaks_h][:8]
    plt.figure()
    plt.plot(dif_freqs, 'o-')
    plt.title("Différence de fréquences entre les pics verticaux et horizontaux")
    plt.xlabel("Mode")
    plt.ylabel("Différence de fréquence (Hz)")
    plt.grid()
    plt.tight_layout()

# %%

plt.show(block=False)

# %%
