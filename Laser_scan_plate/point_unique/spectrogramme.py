import numpy as np
import sys
from collections import deque
from datetime import datetime # ### ENREGISTREMENT 1/3 : Pour le nom du fichier ###

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import QTimer

from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfAnalogOutNode, 
                   DwfAnalogOutFunction, DwfAcquisitionMode)
from pydwf.utilities.open_dwf_device import openDwfDevice

def analog_out_noise(analogOut, periode, sample_frequency, channel=0, amplitude=2.2, bandwidth_hz=5000.0, seed=420):
    """
    Génère un bruit blanc matériel sur l'Analog Discovery 3.
    """
    node_carrier = DwfAnalogOutNode.Carrier
    analogOut.reset(channel)
    _, buffer_max = analogOut.nodeDataInfo(channel, node_carrier)
    
    step_time = int(periode * sample_frequency)
    freq = np.fft.rfftfreq(step_time, d=1/sample_frequency)

    A = np.zeros(freq.shape)

    np.random.seed(seed)
    masque = (freq >= 100) & (freq <= bandwidth_hz)
    A[masque] = 1.0  
    phase = np.random.uniform(0, 2*np.pi, size=A.shape)
    A_complex = A * np.exp(1j * phase)
    waveform = np.fft.irfft(A_complex, n=step_time)

    # Normalisation
    waveform = waveform / np.max(np.abs(waveform))

    analogOut.nodeEnableSet(channel, node_carrier, True)
    analogOut.nodeFunctionSet(channel, node_carrier, DwfAnalogOutFunction.Custom)
    analogOut.nodeDataSet(channel, node_carrier, waveform)
    analogOut.nodeFrequencySet(channel, node_carrier, 1/periode) 
    analogOut.nodeAmplitudeSet(channel, node_carrier, amplitude)
    analogOut.nodeOffsetSet(channel, node_carrier, 0.0)

    analogOut.configure(channel, True)

def main():
    # ----------------------------------------------------------------------
    # 1. PRÉPARATION DE L'INTERFACE GRAPHIQUE
    # ----------------------------------------------------------------------
    app = pg.mkQApp("Oscilloscope AD3 - Spectrogramme Longue Durée")

    win = pg.GraphicsLayoutWidget(show=True, title="Acquisition AD3")
    win.resize(1000, 700)

    plot = win.addPlot(title="Spectrogramme de la FRF (Dérive temporelle)")
    plot.setLabel('bottom', "Fréquence", units='Hz')
    plot.setLabel('left', "Temps (Trames affichées)")
    plot.setLimits(xMin=0)

    # Paramètres de base
    freq_echantillonnage = 8192.0 
    taille_buffer = 8192 * 2
    freqs = np.fft.rfftfreq(taille_buffer, d=1/freq_echantillonnage)

    # Configuration du Spectrogramme
    img = pg.ImageItem()
    plot.addItem(img)

    colormap = pg.colormap.get('inferno')
    img.setLookupTable(colormap.getLookupTable())

    # Taille visuelle : combien de lignes on garde à l'écran
    history_size = 1000 

    # Matrice circulaire pour l'affichage (Fréquence, Temps)
    img_buffer = np.full((len(freqs), history_size), -80.0)
    img.setRect(QtCore.QRectF(0, 0, freqs[-1], history_size))

    # Légende colorée
    color_bar = pg.ColorBarItem(values=(-80, 20), colorMap=colormap)
    color_bar.setImageItem(img)
    win.addItem(color_bar)

    # ----------------------------------------------------------------------
    # 2. CONNEXION ET CONFIGURATION DE L'AD3
    # ----------------------------------------------------------------------
    dwf = DwfLibrary()
    print("Connexion à l'Analog Discovery en cours...")
    
    with openDwfDevice(dwf) as device:
        print("Connecté avec succès.")
        
        record_length = taille_buffer / freq_echantillonnage
        bandwidth_hz = 2500
        CH1 = 0
        CH2 = 1

        # Historiques limités pour la moyenne glissante (lissage mathématique)
        nb_aver = 10
        historique_fft_CH1 = deque(maxlen=nb_aver)
        historique_fft_CH2 = deque(maxlen=nb_aver)
        
        ### ENREGISTREMENT 2/3 : Liste pour stocker les données en mémoire ###
        historique_sauvegarde = []
        ######################################################################

        sample_frequency = freq_echantillonnage
        analog_out_noise(device.analogOut, record_length, sample_frequency, channel=CH1, amplitude=2.2, bandwidth_hz=bandwidth_hz, seed=420)

        analogIn = device.analogIn
        analogIn.reset()
        analogIn.channelEnableSet(CH1, True)
        analogIn.channelRangeSet(CH1, 5.0) 
        analogIn.channelEnableSet(CH2, True)
        analogIn.channelRangeSet(CH2, 5.0)
        
        analogIn.acquisitionModeSet(DwfAcquisitionMode.ScanScreen)
        analogIn.frequencySet(freq_echantillonnage)
        analogIn.bufferSizeSet(taille_buffer)
        analogIn.configure(False, True)

        # ----------------------------------------------------------------------
        # 3. LA BOUCLE DE MISE À JOUR (Calcul rapide + Affichage ralenti)
        # ----------------------------------------------------------------------
        
        TRAMES_AVANT_AFFICHAGE = 30 
        compteur_trames = 0
        
        def update_graph():
            nonlocal img_buffer, compteur_trames

            # 1. Acquisition à pleine vitesse
            analogIn.status(True)
            data_ch1 = analogIn.statusData(CH1, taille_buffer)
            data_ch2 = analogIn.statusData(CH2, taille_buffer)
            
            fft_CH1 = np.fft.rfft(data_ch1)
            fft_CH2 = np.fft.rfft(data_ch2)

            historique_fft_CH1.append(fft_CH1)  
            historique_fft_CH2.append(fft_CH2)

            # 2. Calcul mathématique à pleine vitesse (pour la précision)
            if len(historique_fft_CH1) > 0:
                Sxy = np.mean(np.conjugate(historique_fft_CH2) * historique_fft_CH1, axis=0)
                Sxx = np.mean(np.conjugate(historique_fft_CH2) * historique_fft_CH2, axis=0) 

                # Calcul de la fonction de transfert en dB
                H = Sxy / (Sxx + 1e-12) 
                amplitude_H_dB = 20 * np.log10(np.abs(H) + 1e-12)

                # 3. Mise à jour de l'écran et sauvegarde au ralenti
                compteur_trames += 1
                
                if compteur_trames >= TRAMES_AVANT_AFFICHAGE:
                    
                    ### ENREGISTREMENT 3A/3 : Ajout discret à la mémoire ###
                    # On sauvegarde en float32 pour diviser le poids par 2
                    historique_sauvegarde.append(amplitude_H_dB.astype(np.float32).copy())
                    ########################################################
                    
                    # Défilement du tampon visuel : on décale et on met la nouvelle ligne
                    img_buffer[:, :-1] = img_buffer[:, 1:]
                    img_buffer[:, -1] = amplitude_H_dB

                    # Application à l'image
                    img.setImage(img_buffer, autoLevels=False)
                    
                    # On remet le compteur à zéro
                    compteur_trames = 0

        # Le timer tourne très vite (~30 fois par seconde)
        timer = QTimer()
        timer.timeout.connect(update_graph)
        timer.start(33) 

        # Blocage ici pendant que la fenêtre est ouverte
        app.exec()

        ### ENREGISTREMENT 3B/3 : Génération du fichier à la fermeture ###
        print("\nCréation du fichier NPZ en cours...")
        if len(historique_sauvegarde) > 0:
            matrice_finale = np.array(historique_sauvegarde)
            nom_fichier = f"Derive_Freq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            
            # Sauvegarde compressée (spectro, fréquences, et intervalle de temps)
            np.savez_compressed(nom_fichier, 
                                spectro=matrice_finale, 
                                freqs=freqs,
                                trames_moyennees=TRAMES_AVANT_AFFICHAGE)
            
            print(f"✅ Fichier sauvegardé : {nom_fichier}")
            print(f"📊 {matrice_finale.shape[0]} mesures enregistrées (soit {matrice_finale.shape[0]} x {TRAMES_AVANT_AFFICHAGE} trames réelles).")
        else:
            print("Aucune donnée collectée.")
        ##################################################################

    print("Interface fermée, connexion à l'AD3 clôturée proprement.")

if __name__ == "__main__":
    main()