import numpy as np
import sys
from collections import deque
from datetime import datetime

import pyqtgraph as pg
from PyQt5.QtCore import QTimer

from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfAnalogOutNode, 
                   DwfAnalogOutFunction, DwfAcquisitionMode)
from pydwf.utilities.open_dwf_device import openDwfDevice

def analog_out_noise(analogOut, periode, sample_frequency, channel=0, amplitude=2.2, bandwidth_hz=5000.0, seed=420):
    """
    Démarre un générateur de bruit blanc matériel continu et 100% aléatoire.
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

    ### --- Configuration de la Porteuse en Bruit Blanc --- ###
    analogOut.nodeEnableSet(channel, node_carrier, True)
    analogOut.nodeFunctionSet(channel, node_carrier, DwfAnalogOutFunction.Custom)
    analogOut.nodeDataSet(channel, node_carrier, waveform)
    analogOut.nodeFrequencySet(channel, node_carrier, 1/periode) 
    
    analogOut.nodeAmplitudeSet(channel, node_carrier, amplitude)
    analogOut.nodeOffsetSet(channel, node_carrier, 0.0)

    analogOut.configure(channel, True)

def main():
    # ----------------------------------------------------------------------
    # 1. PRÉPARATION DE L'INTERFACE GRAPHIQUE (PyQtGraph)
    # ----------------------------------------------------------------------
    app = pg.mkQApp("Oscilloscope AD3 - Essais d'Impact")

    win = pg.GraphicsLayoutWidget(show=True, title="Acquisition AD3")
    win.resize(1000, 600)

    plot = win.addPlot(title="Voie 2 : Spectre (en direct)")
    plot.setLabel('bottom', "Fréquence", units='Hz')
    plot.setLabel('left', "Amplitude")
    # plot.setYRange(0, 5) # À ajuster selon l'amplitude de votre FFT
    plot.showGrid(x=True, y=True, alpha=0.3)

    # Création du curseur vertical
    curseur = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
    curseur.setPos(1000) # Position initiale à 1000 Hz
    plot.addItem(curseur)

    label_valeur = pg.TextItem(text="", color="r", anchor=(0, 1))
    plot.addItem(label_valeur)

    # Réglage de la fréquence et du nombre de points
    freq_echantillonnage = 8192.0 # 100 kHz
    taille_buffer = 8192 * 2

    # L'axe X des fréquences (calculé une seule fois)
    freqs = np.fft.rfftfreq(taille_buffer, d=1/freq_echantillonnage)

    def curseur_deplace(ligne):
        freq_visee = ligne.value()
        label_valeur.setText(f"Curseur X : {freq_visee:.1f} Hz")
        # On place le texte légèrement au-dessus du curseur
        label_valeur.setPos(freq_visee, plot.viewRange()[1][1] * 0.9) 

    curseur.sigPositionChanged.connect(curseur_deplace)
    curseur_deplace(curseur)

    courbe = plot.plot(pen='y') 

    # ----------------------------------------------------------------------
    # 2. CONNEXION ET CONFIGURATION DE L'AD3
    # ----------------------------------------------------------------------
    dwf = DwfLibrary()

    print("Connexion à l'Analog Discovery en cours...")
    
    # Le bloc 'with' garantit la fermeture propre de l'AD3 en cas de crash
    with openDwfDevice(dwf) as device:
        print("Connecté avec succès.")
        
        record_length = taille_buffer / freq_echantillonnage
        bandwidth_hz = 2500
        CH1 = 0
        CH2 = 1

        # File d'attente optimisée pour garder les 5 dernières FFT (Moyenne glissante)
        nb_aver = 10
        historique_fft_CH1 = deque(maxlen=nb_aver)
        historique_fft_CH2 = deque(maxlen=nb_aver)

        sample_frequency = freq_echantillonnage
        # Démarrage de la sortie analogique
        analog_out_noise(device.analogOut, record_length, sample_frequency, channel=CH1, amplitude=2.2, bandwidth_hz=bandwidth_hz, seed=420)

        analogIn = device.analogIn

        # Reset et configuration de l'entrée analogique
        analogIn.reset()
        analogIn.channelEnableSet(CH1, True)
        analogIn.channelRangeSet(CH1, 5.0) 
        analogIn.channelEnableSet(CH2, True)
        analogIn.channelRangeSet(CH2, 5.0)
        
        # Mode continu
        analogIn.acquisitionModeSet(DwfAcquisitionMode.ScanScreen)
        analogIn.frequencySet(freq_echantillonnage)
        analogIn.bufferSizeSet(taille_buffer)
        
        # Démarre l'acquisition
        analogIn.configure(False, True)

        # ----------------------------------------------------------------------
        # 3. LA BOUCLE DE MISE À JOUR (Le "Cerveau" du temps réel)
        # ----------------------------------------------------------------------
        def update_graph():
            # Met à jour le statut interne de l'AD3
            analogIn.status(True)
            
            # On lit tout le buffer d'un coup (taille_buffer points)
            data_ch1 = analogIn.statusData(CH1, taille_buffer)
            data_ch2 = analogIn.statusData(CH2, taille_buffer)
            
            # 1. Calcul de la FFT COMPLEXE (SANS np.abs ! On garde la phase)
            fft_CH1 = np.fft.rfft(data_ch1)
            fft_CH2 = np.fft.rfft(data_ch2)

            # Ajout à l'historique
            historique_fft_CH1.append(fft_CH1)  
            historique_fft_CH2.append(fft_CH2)

            # On vérifie qu'on a au moins une donnée pour éviter les erreurs au démarrage
            if len(historique_fft_CH1) > 0:
                
                # 2. Calcul des densités spectrales moyennes
                # Supposons que CH2 est l'entrée (le marteau/bruit) et CH1 la sortie (le capteur)
                Sxy = np.mean(np.conjugate(historique_fft_CH2) * historique_fft_CH1, axis=0)
                Sxx = np.mean(np.conjugate(historique_fft_CH2) * historique_fft_CH2, axis=0) 

                # 3. Calcul de la fonction de transfert H
                H = Sxy / (Sxx+1e-12) 
                           
                
                # 4. On calcule l'amplitude (le module) pour pouvoir l'afficher sur l'écran
                amplitude_H = np.abs(H)

                # Mise à jour de la courbe
                courbe.setData(x=freqs, y=amplitude_H)

        timer = QTimer()
        timer.timeout.connect(update_graph)
        timer.start(33) # ~30 fps

        # Lance l'application. Le script bloque ici tant que la fenêtre est ouverte.
        app.exec()

    # Quand la fenêtre est fermée, on sort du bloc 'with'
    print("Application terminée. Connexion à l'AD3 fermée proprement.")

if __name__ == "__main__":
    main()