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
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260420_164158/donnees_completes.nc"
ds = xr.open_dataset(chemin_fichier_nc, engine="netcdf4")

SiS = ds["signal_mesure"]        # Matrice 3D (X, Y, Temps)
SiE = ds["signal_source"] # Matrice 3D (X, Y, Temps)
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
 

# Création de l'axe des fréquences (uniquement positives)
freqs = np.fft.rfftfreq(step_time, d=1/sample_frequency)

demi_n = len(freqs)  # Nombre de fréquences positives (incluant la DC)

# On applique une fenêtre de Hanning sur chaque tronçon pour éviter le "leakage"
window = np.hanning(step_time)

# 1. Pré-allocation des matrices d'accumulation (Finis les tableaux 4D géants !)
H = np.zeros((nb_x, nb_y, demi_n), dtype=complex)
rep_source_accum = np.zeros((nb_x, nb_y, demi_n), dtype=np.float64)

for i in range(nb_x):
    for j in range(nb_y):
        sig_s = SiS[i, j, :].values
        sig_e = SiE[i, j, :].values

        sxx_pt = np.zeros(demi_n, dtype=np.float64)
        sxy_pt = np.zeros(demi_n, dtype=complex)
        src_pt = np.zeros(demi_n, dtype=np.float64)

    # Extraction du tronçon et application de la fenêtre spatio-temporelle
        for k in range(nb_aver):
            tronc_s = sig_s[k*step_time:(k+1)*step_time] * window
            tronc_e = sig_e[k*step_time:(k+1)*step_time] * window

            fft_s = np.fft.rfft(tronc_s)  # On ne garde que les fréquences positives
            fft_e = np.fft.rfft(tronc_e)

            sxy_pt += np.conjugate(fft_e) * fft_s
            sxx_pt += np.real(np.conjugate(fft_e) * fft_e) 
            src_pt += np.abs(fft_e)
        
        H[i, j, :] = sxy_pt / sxx_pt  # On ajoute une petite valeur pour éviter la division par zéro
        rep_source_accum[i, j, :] = src_pt / nb_aver  # Moyenne de la source brute sur les tronçons

# 4. Division finale pour obtenir la moyenne

# L'amplitude de la déformée modale P est la valeur absolue de la fonction de transfert
P = np.real(H)

print(f"Shape de la matrice de transfert P : {P.shape}")

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep = np.sqrt(np.mean(np.abs(H)**2, axis=(0, 1)))
rep_source_mean_1D = np.mean(rep_source_accum, axis=(0, 1))

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
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260416_093014/donnees_completes.nc"
ds = xr.open_dataset(chemin_fichier_nc, engine="netcdf4")

SiS = ds["signal_mesure"]        # Matrice 3D (X, Y, Temps)
SiE = ds["signal_source"] # Matrice 3D (X, Y, Temps)
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
 

# Création de l'axe des fréquences (uniquement positives)
freqs_h = np.fft.rfftfreq(step_time, d=1/sample_frequency)

demi_n_h = len(freqs_h)  # Nombre de fréquences positives (incluant la DC)

# On applique une fenêtre de Hanning sur chaque tronçon pour éviter le "leakage"
window = np.hanning(step_time)

# 1. Pré-allocation des matrices d'accumulation (Finis les tableaux 4D géants !)
H = np.zeros((nb_x, nb_y, demi_n_h), dtype=complex)
rep_source_accum = np.zeros((nb_x, nb_y, demi_n_h), dtype=np.float64)

for i in range(nb_x):
    for j in range(nb_y):
        sig_s = SiS[i, j, :].values
        sig_e = SiE[i, j, :].values

        sxx_pt = np.zeros(demi_n_h, dtype=np.float64)
        sxy_pt = np.zeros(demi_n_h, dtype=complex)
        src_pt = np.zeros(demi_n_h, dtype=np.float64)

    # Extraction du tronçon et application de la fenêtre spatio-temporelle
        for k in range(nb_aver):
            tronc_s = sig_s[k*step_time:(k+1)*step_time] * window
            tronc_e = sig_e[k*step_time:(k+1)*step_time] * window

            fft_s = np.fft.rfft(tronc_s)  # On ne garde que les fréquences positives
            fft_e = np.fft.rfft(tronc_e)

            sxy_pt += np.conjugate(fft_e) * fft_s
            sxx_pt += np.real(np.conjugate(fft_e) * fft_e) 
            src_pt += np.abs(fft_e)
        
        H[i, j, :] = sxy_pt / sxx_pt  # On ajoute une petite valeur pour éviter la division par zéro
        rep_source_accum[i, j, :] = src_pt / nb_aver  # Moyenne de la source brute sur les tronçons

# 4. Division finale pour obtenir la moyenne

# L'amplitude de la déformée modale P est la valeur absolue de la fonction de transfert
P = np.real(H)

print(f"Shape de la matrice de transfert P : {P.shape}")

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep_h = np.sqrt(np.mean(np.abs(H)**2, axis=(0, 1)))
rep_source_mean_1D_h = np.mean(rep_source_accum, axis=(0, 1))

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

freq_mef = [np.float64(45.955418770460696), np.float64(57.6294273677844), np.float64(95.63643497988748), np.float64(109.71497113892785), np.float64(128.60051094271873), np.float64(167.76571844921006), np.float64(213.3017142133707), np.float64(218.217841095426)]
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
