import aim_laser as las
import rclpy
import numpy as np
import time
import os
import json
from datetime import datetime

from pydwf import (DwfLibrary, DwfEnumConfigInfo, DwfTriggerSource, 
                   DwfAnalogOutNode, DwfAnalogOutFunction, DwfAcquisitionMode,
                   DwfState)

import time

from pydwf.utilities import openDwfDevice

from pydwf.utilities.open_dwf_device import openDwfDevice

def analog_out_noise(analogOut, periode, sample_frequency , channel=0, amplitude=3.0 , bandwidth_hz=5000.0 , seed=420 ):
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
    masque = (freq >= 20) & (freq <= bandwidth_hz)  # Masque pour les fréquences entre 100 Hz et la bande passante souhaitée
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

def acquisition(analogIn, sample_frequency, record_length):
     
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
    trigger_position = 0.0  

    analogIn.triggerSourceSet(DwfTriggerSource.AnalogOut1)
    analogIn.triggerPositionSet(trigger_position)

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

    except KeyboardInterrupt:
        print("Acquisition stopped by user.")

    
    return wave_ch1, wave_ch2  # (Mesure, Source)

import netCDF4 as nc 

def main(record_length=10, nb_aver=5, args=None):

    bandwidth_hz = 5000.0

    rclpy.init(args=args)
    node = las.Point_Aimer_Ur7e()
    dwf = DwfLibrary()

    def maximize_analog_out_buffer_size(configuration_parameters):
        return configuration_parameters[DwfEnumConfigInfo.AnalogOutBufferSize]

    with openDwfDevice(dwf, score_func=maximize_analog_out_buffer_size) as device:
        
        CH1 = 0

        sample_frequency = 21300.0
        
        analog_out_noise(device.analogOut, record_length /nb_aver, sample_frequency, channel=CH1, amplitude=5.0, bandwidth_hz=bandwidth_hz, seed=42)

        nb_x_point = 24
        nb_y_point = 30
        x_point = np.linspace(0.01, 0.99, nb_x_point)
        y_point = np.linspace(0.01, 0.99, nb_y_point)
        nb_point = nb_x_point * nb_y_point
        
        # =======================================================
        # CRÉATION DU FICHIER ET DE L'ARBORESCENCE AVANT LA BOUCLE
        # =======================================================
        dossier_principal = "Mes_Scans_AD3"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin_sauvegarde = os.path.join(dossier_principal, f"Scan_{timestamp}")
        os.makedirs(chemin_sauvegarde, exist_ok=True)
        print(f"Fichier créé, sauvegarde en direct dans : {chemin_sauvegarde}/")

        # Sauvegarde JSON
        parametres = {
            "sample_frequency": sample_frequency,
            "record_length": record_length,
            "nb_aver": nb_aver,
            "nb_x_point": nb_x_point,
            "nb_y_point": nb_y_point,
            "x_point_mm": x_point.tolist(),
            "y_point_mm": y_point.tolist()
        }
        with open(os.path.join(chemin_sauvegarde, "parametres.json"), "w") as f:
            json.dump(parametres, f, indent=4)

        fichier_nc = os.path.join(chemin_sauvegarde, "donnees_completes.nc")
        
        # Calcul du vecteur temps
        num_time_steps = int(sample_frequency * record_length)
        t_array = np.linspace(0, record_length, num_time_steps, endpoint=False)

        # -------------------------------------------------------
        # PRÉPARATION DU FICHIER NETCDF SUR LE DISQUE
        # -------------------------------------------------------
        dataset = nc.Dataset(fichier_nc, 'w', format='NETCDF4')

        # 1. Dimensions
        dataset.createDimension('x', nb_x_point)
        dataset.createDimension('y', nb_y_point)
        dataset.createDimension('temps', num_time_steps)

        # 2. Variables de coordonnées
        var_x = dataset.createVariable('x', 'f4', ('x',))
        var_y = dataset.createVariable('y', 'f4', ('y',))
        var_temps = dataset.createVariable('temps', 'f4', ('temps',))
        var_x.units = "mm"
        var_y.units = "mm"
        var_temps.units = "s"

        # 3. Variables de données (f4 = float32 pour économiser de la place)
        var_mesure = dataset.createVariable('signal_mesure', 'f8', ('x', 'y', 'temps'))
        var_mesure.units = "Volts"
        var_mesure.description = "Signal acquis sur la plaque (CH1)"

        var_source = dataset.createVariable('signal_source', 'f8', ('x', 'y', 'temps'))
        var_source.units = "Volts"
        var_source.description = "Signal source généré (CH2)"

        # 4. Attributs globaux (Métadonnées)
        dataset.sample_frequency_Hz = sample_frequency
        dataset.source_frequency_Hz = bandwidth_hz
        dataset.nb_aver = nb_aver
        dataset.robot_model = "UR7e"
        dataset.date_scan = timestamp

        # 5. Remplissage des coordonnées (Axes)
        var_x[:] = x_point
        var_y[:] = y_point
        var_temps[:] = t_array
        
        # On force la création de l'entête sur le disque
        dataset.sync() 
        # =======================================================

        # Génération du parcours
        mat_dep = []
        for i in (np.array(range(nb_x_point))+1):
            for j in (np.array(range(nb_y_point))+1):
                mat_dep.append((i , j*(-1)**(i+1) % (nb_y_point+1)))
        mat_dep = np.array(mat_dep) - (1, 1)

        av_point = 0
        time.sleep(4)  # Petit délai pour s'assurer que le régime transitoire est passé

        # =======================================================
        # BOUCLE D'ACQUISITION
        # =======================================================
        for (i, j) in mat_dep:
            
            node.aim_UR7e(x_point[i], y_point[j])
            
            # Acquisition
            sig_mesure, sig_source = acquisition(device.analogIn, sample_frequency, record_length)
            
            var_mesure[i, j, :] = sig_mesure
            var_source[i, j, :] = sig_source
            dataset.sync()  
            
            av_point += 1
            print(f"📍 {av_point}/{nb_point} acquis et sauvegardés. \n X={100 * x_point[i]:.2f}%, Y={100 * y_point[j]:.2f}%")

        # 3. SÉCURITÉ : Extinction du générateur et du robot
        device.analogOut.reset(-1)  
        node.destroy_node()
        rclpy.shutdown()
        
        # Fermeture propre du fichier
        dataset.close() 

    print(5*"-" + "\nScan terminé et fichier verrouillé avec succès !")
    return

if __name__ == "__main__":
    main()