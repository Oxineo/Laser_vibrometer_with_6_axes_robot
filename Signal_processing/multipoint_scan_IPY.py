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
chemin_fichier_nc = "/home/adm-discohbot/Documents/Stage_Recherche_M2_Arthur/Mes_Scans_AD3/Scan_20260402_142109/donnees_completes.nc"
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
P = np.real(H)

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep = np.mean(np.abs(P), axis=(0, 1))
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
P = np.real(H)

# Moyenne spatiale de la fonction de transfert pour le graphe 1D
rep_h = np.mean(np.abs(P), axis=(0, 1))
rep_source_mean_1D_h = np.mean(rep_source_mean, axis=(0, 1))

#%%
import matplotlib
matplotlib.use('Qt5Agg')

fft_sup = plt.figure(figsize=(12, 6))
ax = fft_sup.add_subplot(1, 1, 1) 

ax.loglog(freqs[5:demi_n], rep[5:demi_n], label="Réponse modale verticale", color='blue')
ax.loglog(freqs_h[20:demi_n_h], rep_h[20:demi_n_h], label="Réponse modale horizontale (Sxx)", color='orange')


ax.set_title("Réponse modale et de la source en fonction de la fréquence")
ax.set_xlabel("Fréquence (Hz)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid()
plt.tight_layout()

# %%
import scipy.signal as signal

# Détection des pics dans la réponse modale verticale
peaks_v, _ = signal.find_peaks(rep[5:demi_n],prominence=0.1, width=1)  # Ajustez le seuil de hauteur selon vos données
peaks_h, _ = signal.find_peaks(rep_h[5:demi_n_h], prominence=0.2, width=1)

# %%

### Pic : Freqences des modes propres

plt.loglog(freqs[5:demi_n][peaks_v], rep[5:demi_n][peaks_v], 'x', label='Pics verticaux', color='red')
plt.loglog(freqs_h[5:demi_n_h][peaks_h], rep_h[5:demi_n_h][peaks_h], 'x', label='Pics horizontaux', color='black')
plt.legend()

#%%

freq_mef = [np.float64(49.0501679967059), np.float64(49.0501679967059), np.float64(62.0479004000161), np.float64(102.04770487083248), np.float64(118.16828544240893), np.float64(138.55703062301328), np.float64(180.26071164496904), np.float64(229.81703008587203), np.float64(235.02937381633674), np.float64(277.9705322492294), np.float64(317.04630462138664), np.float64(350.3789852609248), np.float64(367.866038757654), np.float64(395.1279315867118), np.float64(425.7379570147936), np.float64(537.8247545619424), np.float64(556.8510518020956), np.float64(566.701507980634), np.float64(567.9539297225444), np.float64(594.3120568662456), np.float64(631.832438300778), np.float64(685.2328311588914), np.float64(762.632409150582), np.float64(777.8659549219907), np.float64(846.2088755116199), np.float64(872.0956220100869), np.float64(898.72878617357), np.float64(898.72878617357), np.float64(927.6978709232011), np.float64(937.6610681343418), np.float64(1008.6422864891476), np.float64(1043.7874004695168), np.float64(1046.5841601824784), np.float64(1074.4384205405875), np.float64(1201.1020758192792), np.float64(1220.4778066410481), np.float64(1271.4659287380257), np.float64(1303.936985810302), np.float64(1307.5365951901717), np.float64(1362.1386078754274), np.float64(1384.1443552653993), np.float64(1421.0606086133412), np.float64(1427.8251613062312), np.float64(1489.8170447570644), np.float64(1615.7728701233348), np.float64(1636.9524179125062), np.float64(1654.9295628411471), np.float64(1655.1912852072423), np.float64(1676.5742883224718), np.float64(1691.9967772221296), np.float64(1826.715662159472), np.float64(1880.1005620188105)]

for f in freq_mef:
    plt.axvline(f, color='green', linestyle='--', alpha=0.5, label='Fréquences MEF' if f == freq_mef[0] else "")



#%%

### Pic : Affichage des différences

dif_freqs = freqs[5:demi_n][peaks_v][:8] - freqs_h[5:demi_n_h][peaks_h][:8]
plt.figure()
plt.plot(dif_freqs, 'o-')
plt.title("Différence de fréquences entre les pics verticaux et horizontaux")
plt.xlabel("Mode")
plt.ylabel("Différence de fréquence (Hz)")
plt.grid()
plt.tight_layout()

# %%

plt.show()

# %%
