import aim_laser as las
import rclpy

import numpy as np

import time
import matplotlib.pyplot as plt



from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfAnalogOutIdle, DwfTriggerSource, 
                   DwfAnalogOutNode, DwfAnalogOutFunction, DwfAcquisitionMode,
                   DwfState)

from pydwf.utilities import openDwfDevice

import time

from pydwf.utilities.open_dwf_device import openDwfDevice

def antiveille(analogOut , duree_totale=10.0, vitesse_note=0.15, amplitude=1.0, canal=0):
    """
    Génère un arpège chiptune sur l'Analog Discovery 3.
    
    Paramètres :
    - duree_totale (float) : Durée totale de l'effet en secondes.
    - vitesse_note (float) : Temps de maintien de chaque note en secondes.
    - amplitude (float) : Tension crête (volume), par défaut 1.0 Volt.
    - canal (int) : 0 pour W1, 1 pour W2.
    """
    # Gamme pentatonique de base (Do4 à Do5)
    notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    
    dwf = DwfLibrary()
    node = DwfAnalogOutNode.Carrier

    try:
            # 2. CONFIGURATION DE BASE
            analogOut.nodeEnableSet(canal, node, True)
            analogOut.nodeFunctionSet(canal, node, DwfAnalogOutFunction.Triangle)
            analogOut.nodeAmplitudeSet(canal, node, amplitude)
            
            # 3. FRÉQUENCE INITIALE ET DÉLAIS
            analogOut.nodeFrequencySet(canal, node, notes[0])
            analogOut.runSet(canal, 0.0)  # 0.0 = Temps d'exécution infini
            analogOut.repeatSet(canal, 0) # 0 = Répétition infinie

            # 4. DÉMARRAGE FORCÉ
            analogOut.configure(canal, True)
            print(f"Générateur DÉMARRÉ sur W{canal + 1} ({duree_totale}s)...")

            start_time = time.time()
            index = 0
            
            # 5. BOUCLE DE JEU
            while (time.time() - start_time) < duree_totale:
                frequence_actuelle = notes[index % len(notes)]
                
                # Mise à jour de la note
                analogOut.nodeFrequencySet(canal, node, frequence_actuelle)
                
                # Forcer l'appareil à appliquer cette nouvelle note immédiatement
                analogOut.configure(canal, True)
                
                time.sleep(vitesse_note)
                index += 1
                
                if index % len(notes) == 0:
                    notes.reverse()

            # 6. EXTINCTION PROPRE
            analogOut.configure(canal, False)
            print("Lecture terminée !")

    except Exception as e:
        print(f"Une erreur est survenue avec l'instrument : {e}")

def pulse_generation(nb_oscillation=.5, n_point_per_oscillation=1000):
    """
    Génère un buffer d'onde sinusoïdale pour un générateur de signaux arbitraires.
    """
    # 1. On s'assure que le nombre total de points est bien un entier (int)
    total_points = int(nb_oscillation * n_point_per_oscillation)
    
    # 2. On crée un vecteur "temps" normalisé (de 0 à 1.5)
    # np.linspace est parfait pour générer des points uniformément espacés
    t = np.linspace(0, 1, total_points, endpoint=False)
    
    # 3. Calcul vectoriel : Numpy applique le sinus sur tout le tableau instantanément
    waveform =(np.sin(2 * np.pi * t * nb_oscillation)*np.sin(np.pi * t))
    
    return waveform

def custom_analog_out_waveform(analogOut, waveform, waveform_duration, wait_duration):
    """Put the given waveform on the first analog output channel."""

    CH1 = 0

    channel = CH1
    node = DwfAnalogOutNode.Carrier

    analogOut.reset(channel)

    # Show the run of the AnalogOut device on trigger pin #0.
    analogOut.device.triggerSet(0, DwfTriggerSource.AnalogOut1)

    analogOut.nodeEnableSet(channel, node, True)
    analogOut.nodeFunctionSet(channel, node, DwfAnalogOutFunction.Custom)

    # Determine offset and amplitude values to use for the requested waveform.

    (amplitude_min, amplitude_max) = analogOut.nodeAmplitudeInfo(channel, node)  # pylint: disable=unused-variable

    analogOut.nodeAmplitudeSet(channel, node, amplitude_max/2)
    analogOut.nodeOffsetSet(channel, node, 0.0)

    samples = waveform 
    analogOut.nodeDataSet(channel, node, samples)

    # Wait duration before each waveform emission.
    analogOut.waitSet(channel, wait_duration)

    # The frequency of a custom waveform is (1 / waveform_duration).
    analogOut.nodeFrequencySet(channel, node, 1.0 / waveform_duration)

    # Emit precisely one custom waveform.
    analogOut.runSet(channel, waveform_duration)

    # Keep going indefinitely.
    analogOut.repeatSet(channel, 0)

    analogOut.idleSet(channel, DwfAnalogOutIdle.Initial)

    analogOut.configure(False, True)

def acquisition(analogIn, sample_frequency, record_length , nb_aver=1):
     
    ### --- Analog input configuration --- ###
    CH1, CH2 = 0, 1
    channels = (CH1, CH2)

    for channel_index in channels:
        analogIn.channelEnableSet(channel_index, True)
        analogIn.channelRangeSet(channel_index, 5.0)

    analogIn.acquisitionModeSet(DwfAcquisitionMode.Record)
    analogIn.frequencySet(sample_frequency)
    analogIn.recordLengthSet(record_length)


    ### --- Trigger configuration --- ###
    trigger_position = 0
    trigger_level = 0.2
    
    analogIn.triggerSourceSet(DwfTriggerSource.AnalogOut1)
    analogIn.triggerPositionSet(trigger_position)


    wave_ch1_saves = []
    for i in range(nb_aver):
        try:
            samples = []

            # Start acquisition.
            analogIn.configure(False, True)

            # Wait for acquisition to complete.
            while True:
                status = analogIn.status(True)

                (current_samples_available, current_samples_lost, current_samples_corrupted) = analogIn.statusRecord()
                if current_samples_available != 0:
                    current_samples = np.vstack([analogIn.statusData(channel_index, current_samples_available)
                                                for channel_index in channels]).transpose()
                    samples.append(current_samples)

                if status == DwfState.Done:
                    break
                
            samples = np.concatenate(samples)
            
            wave_ch1 = samples[:, 0]
            wave_ch2 = samples[:, 1]

            wave_ch1_saves.append(wave_ch1)
        except KeyboardInterrupt:
            print("Acquisition stopped by user.")
            break

    avg_signal = np.median(wave_ch1_saves , axis=0)
    return avg_signal, wave_ch2  # <-- On renvoie un tuple (Mesure, Source)

def main(record_length = 1 , nb_aver=1 , args=None ) :
    freq_sin = 2000
    nb_oscillation = 0.5
    WAVEFORM_DURATION = nb_oscillation/freq_sin
    waveform = pulse_generation(nb_oscillation=nb_oscillation , n_point_per_oscillation=5000)
    WAIT_DURATION = record_length *3/2

    rclpy.init(args=args)
    node = las.Point_Aimer_Ur7e()
    dwf = DwfLibrary()

    

    def maximize_analog_out_buffer_size(configuration_parameters):
            """Select the configuration with the highest possible analog out buffer size."""
            return configuration_parameters[DwfEnumConfigInfo.AnalogOutBufferSize]

    with openDwfDevice(dwf,
                        score_func=maximize_analog_out_buffer_size) as device:

        time0 = time.time()

        custom_analog_out_waveform(device.analogOut, waveform, WAVEFORM_DURATION , WAIT_DURATION)
            
        # Configuration parameters for the acquisition.
        sample_frequency = 28000
        record_length    = record_length
        trigger_flag     = True   
        signal_amplitude = 5.0    


        nb_x_point = 15
        nb_y_point = 15

        mat_point = {}

        x_point = np.linspace(0.05,0.95 , nb_x_point)
        y_point = np.linspace(0.05,0.95, nb_y_point)
        
        nb_point = nb_x_point * nb_y_point
        # Initialisation de deux dictionnaires
        mat_signal = {}
        mat_source = {}

        mat_dep = []
        for i in (np.array(range(nb_x_point))+1):
            ind_x = i
            for j in (np.array(range(nb_y_point))+1):
                mat_dep.append((i , j*(-1)**(i+1) % (nb_y_point+1)))
                
        mat_dep = np.array(mat_dep) -(1,1)

        av_point = 0

        for (i,j) in mat_dep :
            
            node.aim_UR7e(x_point[i], y_point[j])
            
            # On récupère les deux signaux
            sig_mesure, sig_source = acquisition(device.analogIn, sample_frequency, record_length,nb_aver=nb_aver)
            
            av_point += 1

            mat_signal[(i, j)] = sig_mesure
            mat_source[(i, j)] = sig_source

            print(f"{av_point}/{nb_point} acquis. \n X={100 * x_point[i]:.2f}%, Y={100 * y_point[j]:.2f}%")

            if (time.time()-time0)>1200 :
                device.analogOut.reset(-1)
                antiveille(device.analogOut )
                device.analogOut.reset(-1)
                time.sleep(WAIT_DURATION)
                time0 = time.time()
                custom_analog_out_waveform(device.analogOut, waveform, WAVEFORM_DURATION , WAIT_DURATION)
                    

        # 3. SÉCURITÉ : Extinction du générateur et du robot
        device.analogOut.reset(-1)  
        node.destroy_node()
        rclpy.shutdown()
        
    print(5*"-"+"\nScan terminé")

    # ==========================================
    # SAUVEGARDE CENTRALISÉE DES DONNÉES
    # ==========================================
    import os
    import json
    from datetime import datetime
    import xarray as xr

    # 1. Création de l'arborescence des dossiers
    dossier_principal = "Mes_Scans_AD3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_sauvegarde = os.path.join(dossier_principal, f"Scan_{timestamp}")
    os.makedirs(chemin_sauvegarde, exist_ok=True)
    print(f"Sauvegarde en cours dans : {chemin_sauvegarde}/")

    # 2. Sauvegarde des paramètres en JSON
    parametres = {
        "sample_frequency": sample_frequency,
        "freq_sin": freq_sin,
        "nb_oscillation": nb_oscillation,
        "record_length": record_length,
        "nb_x_point": nb_x_point,
        "nb_y_point": nb_y_point,
        "x_point_mm": x_point.tolist(),
        "y_point_mm": y_point.tolist()
    }
    with open(os.path.join(chemin_sauvegarde, "parametres.json"), "w") as f:
        json.dump(parametres, f, indent=4)

    # 3. Préparation des matrices Numpy pour Xarray
    # MODIFICATION ICI : On construit les matrices en (X, Y, Temps)
    # L'axe i (X) est la boucle extérieure, l'axe j (Y) est la boucle intérieure
    matrice_signal_3d = np.array([[mat_signal[(i,j)] for j in range(nb_y_point)] for i in range(nb_x_point)])
    matrice_source_3d = np.array([[mat_source[(i,j)] for j in range(nb_y_point)] for i in range(nb_x_point)])
    
    # Création du vecteur temps
    t_array = np.linspace(0, record_length, int(sample_frequency * record_length), endpoint=False)

    # 4. Création du Dataset Xarray
    ds = xr.Dataset(
        # --- Les données mesurées ---
        data_vars={
            "signal_mesure": (
                ["x", "y", "temps"], # <-- L'ordre est maintenant [X, Y, Temps]
                matrice_signal_3d, 
                {"units": "Volts", "description": "Signal acquis sur la plaque (CH1)"}
            ),
            "signal_source": (
                ["x", "y", "temps"], # <-- Idem pour la source
                matrice_source_3d, 
                {"units": "Volts", "description": "Signal source généré (CH2)"}
            )
        },
        # --- Les axes physiques ---
        coords={
            "x": ("x", x_point, {"units": "mm"}), # On met X en premier par convention visuelle
            "y": ("y", y_point, {"units": "mm"}),
            "temps": ("temps", t_array, {"units": "s"})
        },
        # --- Les métadonnées globales ---
        attrs={
            "sample_frequency_Hz": sample_frequency,
            "source_frequency_Hz": freq_sin,
            "robot_model": "UR7e",
            "date_scan": timestamp
        }
    )

    # 5. Sauvegarde au format NetCDF4
    fichier_nc = os.path.join(chemin_sauvegarde, "donnees_completes.nc")
    ds.to_netcdf(fichier_nc, engine="netcdf4")
    print(f"Données Xarray/NetCDF et JSON enregistrées avec succès.")

    plt.show()

    return

if __name__ == "__main__":
    main()

