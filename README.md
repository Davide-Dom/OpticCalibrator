La libreria OpticCalibrator.py è una soluzione autonoma per la calibrazione ottica di telecamere, specializzata nella correzione della distorsione della lente. È progettata per essere modulare e riutilizzabile, gestendo l'intero flusso di lavoro, dall'acquisizione delle immagini del pattern ChArUco al calcolo dei parametri e alla correzione finale dei fotogrammi.

Architettura della Libreria
L'architettura del modulo è orientata agli oggetti ed è composta da due classi principali:

EdgeRecovery: Una classe interna che si occupa di gestire i movimenti della telecamera (specificamente PTZ) in modalità automatica. Il suo scopo è garantire che il pattern di calibrazione rimanga sempre visibile e al centro del campo visivo, effettuando movimenti correttivi quando i marker si avvicinano ai bordi del fotogramma.

OpticCalibrator: Questa è la classe principale che contiene tutta la logica di calibrazione. Gestisce l'intero ciclo di vita del processo:

Acquisizione Dati: Fornisce metodi per l'acquisizione delle immagini sia in modo manuale (per telecamere fisse) che automatico (per telecamere PTZ).

Calcolo e Filtraggio: Utilizza il modulo cv2.aruco di OpenCV per rilevare gli angoli del pattern e calcolare i parametri della telecamera. Include una strategia di filtraggio per scartare le "viste degeneri" che potrebbero compromettere l'accuratezza del risultato.

Gestione File: Salva e carica i parametri di calibrazione su un file .npz, eliminando la necessità di calibrare nuovamente la telecamera a ogni avvio dell'applicazione.

Correzione: Applica i parametri calcolati per correggere in tempo reale la distorsione ottica di un fotogramma.

Metodi Principali
calibrate_optics_automatically(): Avvia il processo di calibrazione automatica, ideale per telecamere PTZ.

calibrate_optics_manually(): Avvia il processo di calibrazione manuale.

load_calibration(): Carica i parametri di calibrazione da un file.

save_calibration(): Salva i parametri su un file.

correct_frame(): Applica la correzione della distorsione a un fotogramma.

#--------------------------------------------------------------------------------------------

# Esempio 1: Eseguire una calibrazione automatica per una telecamera PTZ

from Cal_ReolinkE1_Camera_Controller import TestedONVIFController, RealCameraConfig
from Cal_ReolinkE1_Config import Direction
from OpticCalibrator import OpticCalibrator
import tkinter as tk

class MockCamera:
    def capture_frame(self):
        # Sostituisci con la logica della tua camera per catturare un frame
        return None
    def get_fresh_frame(self):
        return None
    def move_ptz(self, direction, duration):
        pass
    def move_to_software_home(self):
        return True

def run_auto_calibration_example():
    mock_camera_controller = MockCamera()
    root = tk.Tk()
    calibrator = OpticCalibrator(mock_camera_controller, root)
    if calibrator.calibrate_optics_automatically():
        print("Calibrazione automatica completata e salvata.")
    else:
        print("Calibrazione automatica fallita o annullata.")
    root.destroy()

# Chiamata all'esempio
run_auto_calibration_example()

# -----------------------------------------------------------------------------

# Esempio 2: Caricare una calibrazione esistente e correggere un fotogramma

from OpticCalibrator import OpticCalibrator
import cv2
import numpy as np
import os
import tkinter as tk

def run_correction_example():
    # Per questo esempio, assumiamo che il file lens_calibration_data.npz esista
    if not os.path.exists("lens_calibration_data.npz"):
        print("Errore: file 'lens_calibration_data.npz' non trovato. Esegui prima la calibrazione.")
        return

    # Mock del controller della camera (non necessario se si usa solo per correzione)
    class MockCamera:
        def capture_frame(self): return np.zeros((720, 1280, 3), dtype=np.uint8)
        def get_fresh_frame(self): return self.capture_frame()

    mock_camera_controller = MockCamera()
    root = tk.Tk() # Necessario per l'inizializzazione della classe OpticCalibrator
    calibrator = OpticCalibrator(mock_camera_controller, root)
    root.destroy()

    if calibrator.is_calibrated:
        print("Parametri di calibrazione caricati con successo.")
        # Simula un frame acquisito dalla telecamera
        distorted_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Correggi il frame
        corrected_frame = calibrator.correct_frame(distorted_frame)
        
        # Mostra i frame (in un'applicazione reale, dovresti usare una GUI)
        # cv2.imshow("Frame Corretto", corrected_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Frame corretto con successo.")
    else:
        print("Nessun parametro di calibrazione valido trovato.")

# Chiamata all'esempio
run_correction_example()

# -----------------------------------------------------------------------------

# Esempio 3: Eseguire una calibrazione manuale e mostrare un'anteprima

from OpticCalibrator import OpticCalibrator
import tkinter as tk
import cv2
import numpy as np
import threading
import os

def run_manual_calibration_and_preview():
    # Mock del controller della camera per questo esempio
    class MockCamera:
        def __init__(self):
            # Crea un'immagine di test con un cerchio per simulare un pattern
            self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.circle(self.test_frame, (640, 360), 100, (255, 255, 255), -1)
            self.lock = threading.Lock()
        def capture_frame(self):
            with self.lock:
                return self.test_frame.copy()
        def get_fresh_frame(self):
            with self.lock:
                return self.test_frame.copy()
        def move_ptz(self, direction, duration):
            pass
        def move_to_software_home(self):
            return True

    mock_camera_controller = MockCamera()
    root = tk.Tk()
    
    # Inizializza il calibratore in un thread per non bloccare la GUI
    def calibration_thread():
        calibrator = OpticCalibrator(mock_camera_controller, root)
        if calibrator.calibrate_optics_manually():
            print("Calibrazione manuale completata e salvata.")
        else:
            print("Calibrazione manuale fallita o annullata.")
    
    # Avvia il thread di calibrazione
    thread = threading.Thread(target=calibration_thread)
    thread.start()
    
    # L'utente deve interagire con le finestre di OpenCV per "Acquisire" e "Calcolare".
    print("Avvia la calibrazione manuale e interagisci con le finestre di OpenCV che si apriranno.")
    print("Premi 'Acquisisci' nella finestra della GUI e poi 'Calcola' per terminare.")

    # La GUI deve rimanere in esecuzione
    root.mainloop()

# Chiamata all'esempio
# Nota: questo esempio aprirà una finestra di dialogo,
# la quale ha bisogno dell'interazione dell'utente per terminare.
run_manual_calibration_and_preview()

