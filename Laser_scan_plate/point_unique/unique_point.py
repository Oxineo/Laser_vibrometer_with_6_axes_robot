import numpy as np
import time
import os
import json
from datetime import datetime

import pyqtgraph as pg # Assurez-vous de bien avoir importé pyqtgraph

from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfTriggerSource, 
                   DwfAnalogOutNode, DwfAnalogOutFunction, DwfAcquisitionMode,
                   DwfState)

import time

import sys
import pyqtgraph as pg
from PyQt5.QtCore import QTimer

# Importation de la bibliothèque Digilent

from pydwf.utilities.open_dwf_device import openDwfDevice


def analog_out_noise(analogOut, periode, sample_frequency , channel=0, amplitude=2.2 , bandwidth_hz=5000.0 , seed=420 ):
    """
    Démarre un générateur de bruit blanc matériel continu et 100% aléatoire.
    
    :param analogOut: L'objet device.analogOut
    :param periode: La durée du signal en secondes
    :param sample_frequency: La fréquence d'échantillonnage
    :param channel: Le canal de sortie (0 = W1, 1 = W2)
    :param amplitude: L'amplitude crête en Volts
    :param bandwidth_hz: Le taux de rafraîchissement du signal aléatoire
    """
    node_carrier = DwfAnalogOutNode.Carrier

    analogOut.reset(channel)

    _, buffer_max = analogOut.nodeDataInfo(channel, node_carrier)
    buffer_size = int(buffer_max)

    step_time = int(periode * sample_frequency)  # Pas de temps
    freq = np.fft.rfftfreq(step_time, d=1/sample_frequency)

    A = np.zeros(freq.shape)

    np.random.seed(seed)
    masque = (freq >= 100) & (freq <= bandwidth_hz)  # Masque pour les fréquences entre 100 Hz et la bande passante souhaitée
    A[masque] = 1.0  # Appliquer le masque pour créer un signal dans la bande de fréquences souhaitée
    phase = np.random.uniform(0, 2*np.pi, size=A.shape)  # Phase aléatoire pour chaque composante fréquentielle
    A_complex = A * np.exp(1j * phase)  # Signal complexe avec amplitude et phase
    waveform = np.fft.irfft(A_complex, n=step_time)

    waveform = waveform / np.max(np.abs(waveform))

    ### --- Configuration de la Porteuse en Bruit Blanc --- ###
    analogOut.nodeEnableSet(channel, node_carrier, True)
    analogOut.nodeFunctionSet(channel, node_carrier, DwfAnalogOutFunction.Custom)
    analogOut.nodeDataSet(channel, node_carrier, waveform)
    # Pour le bruit, la 'fréquence' définit la vitesse à laquelle l'AD3 génère une nouvelle valeur aléatoire.
    # 100 000 Hz donne un bruit blanc d'excellente qualité couvrant tout votre spectre d'intérêt.
    analogOut.nodeFrequencySet(channel, node_carrier, 1/periode) 
    
    analogOut.nodeAmplitudeSet(channel, node_carrier, amplitude)
    analogOut.nodeOffsetSet(channel, node_carrier, 0.0)

    analogOut.configure(channel, True)

def main() :
    # ----------------------------------------------------------------------
    # 1. PRÉPARATION DE L'INTERFACE GRAPHIQUE (PyQtGraph)
    # ----------------------------------------------------------------------

    app = pg.Qt.mkQApp("Oscilloscope AD3 - Essais d'Impact")

    win = pg.GraphicsLayoutWidget(show=True, title="Acquisition AD3")
    win.resize(1000, 600)

    plot = win.addPlot(title="Voie 1 : Mesure de déformation (en direct)")
    plot.setLabel('bottom', "Fréquence", units='Hz')
    plot.setLabel('left', "Tension", units='V')
    plot.setYRange(-5, 5) # Force l'affichage entre -5V et +5V
    plot.showGrid(x=True, y=True, alpha=0.3)

    # Création de la courbe (vide pour l'instant)
    plot.setLogMode(x=True, y=True)

    # ... (Création du plot) ...

    # Création du curseur vertical (InfiniteLine)
    # angle=90 : ligne verticale. movable=True : on peut la glisser à la souris
    curseur = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=2))
    
    # Position initiale du curseur (par exemple à 0.04 secondes)
    curseur.setPos(0.04)
    
    # Ajout du curseur au graphique
    plot.addItem(curseur)

    # Ajoute un label en haut du graphique pour afficher la valeur
    label_valeur = pg.TextItem(text="", color="r", anchor=(0, 1))
    plot.addItem(label_valeur)

    # Réglage de la fréquence et du nombre de points
    freq_echantillonnage = 100000.0 # 100 kHz
    taille_buffer = 8192*4
    
    freqs = np.fft.rfftfreq(taille_buffer, d=1/freq_echantillonnage)

    # Fonction appelée à chaque fois que le curseur est déplacé
    def curseur_deplace(ligne):
        freq_visee = ligne.value() # Récupère la position X du curseur

        # On récupère les vraies valeurs à cet indice

        label_valeur.setText(f"Curseur X : {freq_visee:.4f} s")
        # On peut aussi repositionner le texte à côté du curseur
        label_valeur.setPos(freq_visee, 4.5) 

    # Connecte le mouvement du curseur à la fonction
    curseur.sigPositionChanged.connect(curseur_deplace)
    
    # Initialisation
    curseur_deplace(curseur)

    courbe = plot.plot(pen='y') 

    # ----------------------------------------------------------------------
    # 2. CONNEXION ET CONFIGURATION DE L'AD3
    # ----------------------------------------------------------------------
    dwf = DwfLibrary()

    def maximize_analog_out_buffer_size(configuration_parameters):
                """Select the configuration with the highest possible analog out buffer size."""
                return configuration_parameters[DwfEnumConfigInfo.AnalogOutBufferSize]
    with openDwfDevice(dwf) as device:
        try:
        # Ouvre le premier appareil détecté (-1)
            record_length = 2 
            nb_aver = 5
            bandwidth_hz = 2500
            CH1 = 0

            data_save = []

            sample_frequency = 21300.0
            analog_out_noise(device.analogOut, record_length, sample_frequency, channel=CH1, amplitude=2.2, bandwidth_hz=bandwidth_hz, seed=420)

            analogIn = device.analogIn

            # Reset et configuration de la voie 1
            analogIn.reset()
            analogIn.channelEnableSet(0, True) # 0 correspond à CH1
            analogIn.channelRangeSet(0, 5.0)   # Calibre +/- 5V
            
            # Mode "ScanScreen" : le buffer se remplit en continu comme un oscilloscope
            # (Vérifiez les noms exacts des Enums selon votre version de pydwf)
            analogIn.acquisitionModeSet(DwfAcquisitionMode.ScanScreen)
            


            analogIn.frequencySet(freq_echantillonnage)
            analogIn.bufferSizeSet(taille_buffer)
            
            # Démarre l'acquisition sur l'AD3
            analogIn.configure(False, True)

        # ----------------------------------------------------------------------
        # 3. LA BOUCLE DE MISE À JOUR (Le "Cerveau" du temps réel)
        # ----------------------------------------------------------------------
            def update_graph():
                """Fonction appelée automatiquement par le Timer"""
                # Demande à l'AD3 de mettre à jour son statut interne
                analogIn.status(True)
                
                # Récupère les 8192 derniers points du buffer de la Voie 1 (CH1)
                data = analogIn.statusData(0, taille_buffer)
                fft = np.abs(np.fft.rfft(data))

                # Met à jour la courbe à l'écran instantanément
                courbe.setData(x = freqs , y = fft)

            # Création du Timer
            timer = QTimer()
            timer.timeout.connect(update_graph)
            timer.start(33) # 33 millisecondes = ~30 Images par seconde

            # Lance l'application (le script reste bloqué ici tant que la fenêtre est ouverte)
            sys.exit(app.exec())

            # ----------------------------------------------------------------------
            # 4. FERMETURE PROPRE (Très important pour l'AD3)
            # ----------------------------------------------------------------------
        except Exception as e:
            print(f"Erreur lors de l'acquisition : {e}")

        finally:
            # Ce bloc s'exécute quand vous fermez la fenêtre PyQtGraph
            if device is not None:
                device.close()
                print("Connexion à l'AD3 fermée proprement.")

if __name__ == "__main__":
    main()