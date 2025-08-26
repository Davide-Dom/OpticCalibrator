# OpticCalibrator.py (v5.0)
# FIX: Riaggiunto il metodo _draw_text e rimosso i riferimenti esterni
# per rendere questa libreria completamente autonoma e riutilizzabile.

import time
import cv2
import numpy as np
import logging
import os
from tkinter import messagebox, Toplevel, Label, Button, ttk
from typing import Optional, List, Tuple
from collections import defaultdict

# Importa i moduli di dipendenza necessari
from Cal_ReolinkE1_Camera_Controller import TestedONVIFController
from Cal_ReolinkE1_Config import Direction

class EdgeRecovery:
    """
    CLASSE INTERNA: Analizza la posizione dei marker rilevati e, se sono su un bordo,
    esegue un movimento PTZ correttivo per ricentrare il target.
    """
    def __init__(self, camera_controller: TestedONVIFController, image_width: int, image_height: int):
        self.camera_controller = camera_controller
        self.image_width = image_width
        self.image_height = image_height
        self.RECOVERY_STEP = 0.2
        
        self.x_left_threshold = self.image_width * 0.25
        self.x_right_threshold = self.image_width * 0.75
        self.y_top_threshold = self.image_height * 0.25
        self.y_bottom_threshold = self.image_height * 0.75

    def attempt_recovery(self, marker_corners: List[np.ndarray]) -> bool:
        if not (0 < len(marker_corners) <= 5):
            return False

        all_points = np.concatenate(marker_corners, axis=1).reshape(-1, 2)
        center_x, center_y = np.mean(all_points, axis=0)
        
        moved = False
        if center_x < self.x_left_threshold and center_y < self.y_top_threshold:
            logging.debug("[EdgeRecovery] Target nell'angolo ALTO-SINISTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.UP, self.RECOVERY_STEP)
            self.camera_controller.move_ptz(Direction.LEFT, self.RECOVERY_STEP)
            moved = True
        elif center_x > self.x_right_threshold and center_y < self.y_top_threshold:
            logging.debug("[EdgeRecovery] Target nell'angolo ALTO-DESTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.UP, self.RECOVERY_STEP)
            self.camera_controller.move_ptz(Direction.RIGHT, self.RECOVERY_STEP)
            moved = True
        elif center_x < self.x_left_threshold and center_y > self.y_bottom_threshold:
            logging.debug("[EdgeRecovery] Target nell'angolo BASSO-SINISTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.DOWN, self.RECOVERY_STEP)
            self.camera_controller.move_ptz(Direction.LEFT, self.RECOVERY_STEP)
            moved = True
        elif center_x > self.x_right_threshold and center_y > self.y_bottom_threshold:
            logging.debug("[EdgeRecovery] Target nell'angolo BASSO-DESTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.DOWN, self.RECOVERY_STEP)
            self.camera_controller.move_ptz(Direction.RIGHT, self.RECOVERY_STEP)
            moved = True
        elif center_x < self.x_left_threshold:
            logging.debug("[EdgeRecovery] Target sul bordo SINISTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.LEFT, self.RECOVERY_STEP)
            moved = True
        elif center_x > self.x_right_threshold:
            logging.debug("[EdgeRecovery] Target sul bordo DESTRO. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.RIGHT, self.RECOVERY_STEP)
            moved = True
        elif center_y < self.y_top_threshold:
            logging.debug("[EdgeRecovery] Target sul bordo SUPERIORE. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.UP, self.RECOVERY_STEP)
            moved = True
        elif center_y > self.y_bottom_threshold:
            logging.debug("[EdgeRecovery] Target sul bordo INFERIORE. Eseguo recupero...")
            self.camera_controller.move_ptz(Direction.DOWN, self.RECOVERY_STEP)
            moved = True
            
        return moved


class OpticCalibrator:
    def __init__(self, camera_controller: TestedONVIFController, root):
        self.camera_controller = camera_controller
        self.root = root
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.charuco_board = cv2.aruco.CharucoBoard((7, 5), 0.025, 0.015, self.aruco_dictionary)
        aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, aruco_parameters)
        self.edge_recovery: Optional[EdgeRecovery] = None
        self.manual_stop_request = False
        self.marker_counts = defaultdict(int)

        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        self.new_camera_matrix: Optional[np.ndarray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.filename = "lens_calibration_data.npz"

        self.load_calibration()
    
    # Metodi Pubblici di Calibrazione Ottica
    
    def calibrate_optics_automatically(self) -> bool:
        logging.info("Avvio calibrazione ottica automatica (modalità PTZ)...")
        self.manual_stop_request = False
        all_charuco_corners, all_charuco_ids, last_gray = self._run_capture_loop_auto()

        if last_gray is None:
            logging.info("Calibrazione annullata dall'utente.")
            return False
            
        return self._process_and_calculate(all_charuco_corners, all_charuco_ids, last_gray.shape[::-1])
        
    def calibrate_optics_manually(self) -> bool:
        logging.info("Avvio calibrazione ottica manuale (modalità fissa)...")
        self.manual_stop_request = False
        all_charuco_corners, all_charuco_ids, last_gray = self._run_capture_loop_manual()

        if last_gray is None:
            logging.info("Calibrazione annullata dall'utente.")
            return False
            
        return self._process_and_calculate(all_charuco_corners, all_charuco_ids, last_gray.shape[::-1])

    # Metodi Interni per l'Acquisizione
    
    def _draw_text(self, img, text, pos):
        """Disegna un testo con contorno per una migliore leggibilità. Specifico per questa libreria."""
        font_scale = 1.2; font_thickness = 2; font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), font_thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    def _run_capture_loop_auto(self) -> Tuple[Optional[list], Optional[list], Optional[np.ndarray]]:
        all_charuco_corners, all_charuco_ids = [], []
        PAN_STEP = 0.25; TILT_STEP = 0.25
        
        win_name = "Calibrazione a Serpentina (Premi 's' per iniziare)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1280, 720)

        messagebox.showinfo("Avvio Calibrazione a Serpentina", 
                            "1. Centra manualmente il pattern ChArUco.\n"
                            "2. Premi 's' nella finestra video per avviare la scansione automatica.")

        state = "WAITING"; pan_direction = Direction.RIGHT; tilt_direction = Direction.DOWN
        consecutive_pan_failures = 0; pan_moves_on_this_line = 0
        consecutive_tilts_in_one_direction = 0; total_moves = 0
        recovery_attempts = 0
        last_valid_gray = None
        
        self.marker_counts.clear()
        
        progress_dialog = self._create_progress_dialog()
        
        while True:
            frame = self.camera_controller.capture_frame()
            if frame is None: continue

            if self.edge_recovery is None:
                h, w, _ = frame.shape
                self.edge_recovery = EdgeRecovery(self.camera_controller, w, h)
            
            vis_frame = frame.copy()
            key = cv2.waitKey(30) & 0xFF

            if self.manual_stop_request:
                logging.info("Interruzione manuale del processo di acquisizione.")
                break
            
            if len(all_charuco_ids) >= 100:
                logging.info("Raggiunto il numero massimo di 100 viste valide. Termino l'acquisizione.")
                break
            
            if key == ord('s') and state == "WAITING":
                state = "SCANNING"
                
            if state == "SCANNING":
                logging.info(f"--- Inizio scansione PAN: {pan_direction.name} ---")
                self.camera_controller.move_ptz(pan_direction, PAN_STEP)
                total_moves += 1; time.sleep(1.0)
                
                frame_after_move = self.camera_controller.get_fresh_frame()
                if frame_after_move is None: continue
                
                vis_frame = frame_after_move.copy()
                gray_after_move = cv2.cvtColor(frame_after_move, cv2.COLOR_BGR2GRAY); last_valid_gray = gray_after_move
                marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray_after_move)
                pan_moves_on_this_line += 1
                
                self.marker_counts[len(marker_corners)] += 1
                
                is_valid_capture = False
                if len(marker_corners) > 4:
                    retval, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray_after_move, self.charuco_board)
                    if retval and len(corners) > 4:
                        all_charuco_corners.append(corners); all_charuco_ids.append(ids)
                        is_valid_capture = True
                
                if not is_valid_capture:
                    consecutive_pan_failures += 1
                    if len(marker_corners) == 0:
                        pass
                    elif 1 <= len(marker_corners) <= 4:
                        logging.debug(f"Rilevamento parziale ({len(marker_corners)} markers). Tento recupero dai bordi...")
                        if self.edge_recovery.attempt_recovery(marker_corners):
                            recovery_attempts += 1
                            time.sleep(1.0)
                    else:
                        pass
                else:
                    consecutive_pan_failures = 0
                
                if consecutive_pan_failures >= 2 or pan_moves_on_this_line >= 6:
                    logging.warning(f"Transizione TILT. Causa: {'Pattern perso' if consecutive_pan_failures >= 2 else 'Limite PAN raggiunto'}")
                    pan_moves_on_this_line = 0; consecutive_pan_failures = 0
                    
                    if consecutive_tilts_in_one_direction >= 3:
                        tilt_direction = Direction.UP if tilt_direction == Direction.DOWN else Direction.DOWN
                        logging.warning(f"Limite di 3 TILT raggiunto. Inversione TILT a: {tilt_direction.name}")
                        consecutive_tilts_in_one_direction = 0
                    
                    pan_direction = Direction.LEFT if pan_direction == Direction.RIGHT else Direction.RIGHT
                    logging.info(f"Eseguo TILT {tilt_direction.name} (Tentativo N.{consecutive_tilts_in_one_direction + 1}). Prossima scansione PAN: {pan_direction.name}")
                    self.camera_controller.move_ptz(tilt_direction, TILT_STEP)
                    total_moves += 1
                    consecutive_tilts_in_one_direction += 1
                    time.sleep(1.0)
            
            if progress_dialog.winfo_exists():
                self._update_progress_dialog(len(all_charuco_ids), progress_dialog.update_idletasks)

            y_pos = [50, 110, 170, 230, 290, 350]
            gray_display = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2GRAY)
            marker_corners_display, ids_display, _ = self.aruco_detector.detectMarkers(gray_display)
            
            if len(marker_corners_display) > 4:
                retval, corners_display, ids_display_charuco = cv2.aruco.interpolateCornersCharuco(marker_corners_display, ids_display, gray_display, self.charuco_board)
                if retval and len(corners_display) > 4:
                    cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners_display, ids_display, borderColor=(0, 255, 0))
                    cv2.aruco.drawDetectedCornersCharuco(vis_frame, corners_display, ids_display_charuco, cornerColor=(0, 255, 0))
                else:
                    cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners_display, ids_display, borderColor=(0, 165, 255))
            elif len(marker_corners_display) > 0:
                cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners_display, ids_display, borderColor=(0, 0, 255))

            if state == "WAITING": self._draw_text(vis_frame, "STEP 1: POSIZIONA E PREMI 's'", (20, y_pos[0]))
            else: self._draw_text(vis_frame, f"STEP 2: SCANSIONE PAN ({pan_direction.name})", (20, y_pos[0]))
            self._draw_text(vis_frame, f"Viste ChArUco valide: {len(all_charuco_ids)}", (20, y_pos[2]))
            self._draw_text(vis_frame, f"Marker ArUco trovati: {len(marker_corners_display)}", (20, y_pos[3]))
            self._draw_text(vis_frame, f"Movimenti PTZ totali: {total_moves}", (20, y_pos[4]))
            self._draw_text(vis_frame, f"Tentativi di recupero: {recovery_attempts}", (20, y_pos[5]))
            
            cv2.imshow(win_name, vis_frame)
        
        cv2.destroyWindow(win_name)
        if progress_dialog.winfo_exists():
            progress_dialog.destroy()
        return all_charuco_corners, all_charuco_ids, last_valid_gray

    def _run_capture_loop_manual(self) -> Tuple[Optional[list], Optional[list], Optional[np.ndarray]]:
        all_charuco_corners, all_charuco_ids = [], []
        
        def create_manual_dialog():
            dialog = Toplevel(self.root)
            dialog.title("Modalità Manuale")
            dialog.geometry("300x150")
            dialog.protocol("WM_DELETE_WINDOW", lambda: self._set_manual_stop())
            
            Label(dialog, text="Muovi il pattern e clicca Acquisisci.", font=('Arial', 10)).pack(pady=5)
            self.status_label = Label(dialog, text="Catture valide: 0", font=('Arial', 10, 'bold'))
            self.status_label.pack(pady=5)

            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            
            def acquire_and_update():
                if self._capture_single_frame(all_charuco_corners, all_charuco_ids):
                    self.status_label.config(text=f"Catture valide: {len(all_charuco_ids)}")
                
            Button(button_frame, text="Acquisisci", command=acquire_and_update).pack(side='left', padx=10)
            Button(button_frame, text="Calcola", command=lambda: self._set_manual_stop()).pack(side='right', padx=10)
            return dialog

        win_name = "Calibrazione Manuale"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1280, 720)
        
        manual_dialog = create_manual_dialog()
        last_valid_gray = None
        
        while not self.manual_stop_request:
            frame = self.camera_controller.capture_frame()
            if frame is None:
                continue
            
            vis_frame = frame.copy()
            last_valid_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(last_valid_gray)
            if len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(vis_frame, marker_corners, marker_ids)
            
            self._draw_text(vis_frame, "Premi 'Acquisisci' nella finestra per salvare il frame.", (20, 50))
            cv2.imshow(win_name, vis_frame)
            cv2.waitKey(1)
            
            manual_dialog.update_idletasks()
            
        cv2.destroyWindow(win_name)
        manual_dialog.destroy()
        
        return all_charuco_corners, all_charuco_ids, last_valid_gray

    def _capture_single_frame(self, all_corners: list, all_ids: list) -> bool:
        frame = self.camera_controller.get_fresh_frame()
        if frame is None: return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if len(marker_corners) > 4:
            retval, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, self.charuco_board)
            if retval and len(corners) > 4:
                all_corners.append(corners)
                all_ids.append(ids)
                logging.info(f"Cattura valida aggiunta. Totale: {len(all_ids)}")
                return True
        logging.warning("Cattura non valida. Assicurati che il pattern sia interamente visibile.")
        return False
        
    def _create_progress_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Calibrazione in corso...")
        dialog.geometry("350x450")
        dialog.protocol("WM_DELETE_WINDOW", self._on_close_dialog)
        
        Label(dialog, text="Acquisizione dei dati in corso...", font=('Arial', 10, 'bold')).pack(pady=5)
        
        self.status_label = Label(dialog, text="", font=('Arial', 10), justify='left')
        self.status_label.pack(pady=5)
        
        Label(dialog, text="Statistiche Marker Rilevati", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        table_frame = ttk.Frame(dialog)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ('Markers', 'Conteggio')
        self.marker_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        self.marker_tree.heading('Markers', text='Markers Rilevati')
        self.marker_tree.heading('Conteggio', text='Conteggio')
        self.marker_tree.column('Markers', width=120, anchor='center')
        self.marker_tree.column('Conteggio', width=80, anchor='center')
        self.marker_tree.pack(fill='both', expand=True)
        
        Button(dialog, text="Ferma e Calcola", command=self._set_manual_stop).pack(pady=10)
        
        return dialog
        
    def _on_close_dialog(self):
        self.manual_stop_request = True
        
    def _set_manual_stop(self):
        self.manual_stop_request = True
        
    def _update_progress_dialog(self, total_valid_captures, update_func):
        status_text = (
            f"Catture valide: {total_valid_captures}\n"
            f"Fallite (nessun marker): {self.marker_counts.get(0, 0)}\n"
            f"Fallite (parziali): {sum(self.marker_counts[i] for i in range(1, 5))}\n"
            f"Fallite (degenere): {sum(self.marker_counts[i] for i in range(5, 17))}"
        )
        self.status_label.config(text=status_text)
        
        self.marker_tree.delete(*self.marker_tree.get_children())
        for i in range(1, 18):
            self.marker_tree.insert('', 'end', values=(i, self.marker_counts[i]))
        
        update_func()
        
    def _process_and_calculate(self, all_corners: list, all_ids: list, image_size: tuple) -> bool:
        logging.info(f"Ricevute {len(all_corners)} catture totali per il calcolo.")

        if not all_ids:
            logging.warning("Elenco delle catture vuoto. Annullamento del calcolo.")
            return False

        logging.debug("Inizio fase di filtraggio dei dati...")
        filtered_corners, filtered_ids = self._filter_degenerate_views(all_corners, all_ids)
        
        logging.info(f"Filtraggio completato. Viste originali: {len(all_ids)}. Viste valide (post-filtro): {len(filtered_ids)}.")

        if len(filtered_ids) < 5:
            logging.warning(f"Numero di viste valide ({len(filtered_ids)}) insufficiente per una calibrazione affidabile.")
            if not messagebox.askyesno("Calibrazione Sconsigliata", f"Sono rimaste solo {len(filtered_ids)} viste di alta qualità...\n\nVuoi ricominciare la cattura?"):
                logging.info("Calibrazione annullata dall'utente post-filtraggio.")
                return False
            else:
                logging.info("L'utente ha scelto di ricominciare la cattura.")
                return False

        logging.info(f"Procedo con il calcolo finale usando {len(filtered_ids)} viste valide.")
        
        logging.debug("Dettagli delle viste filtrate:")
        for i, corners in enumerate(filtered_corners):
            logging.debug(f"  - Vista {i+1}: {len(corners)} punti.")
            if len(corners) < 5:
                 logging.warning(f"  - Vista {i+1} ha meno di 5 punti. Potrebbe essere la causa dell'errore di asserzione.")
        
        logging.debug("Calcolo parametri di correzione con cv2.aruco.calibrateCameraCharuco...")
        try:
            ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(filtered_corners, filtered_ids, self.charuco_board, image_size, None, None)
            if ret:
                self.set_calibration_params(mtx, dist, image_size)
                return True
            else:
                messagebox.showerror("Errore", "Impossibile calcolare i parametri di calibrazione.")
                return False
        except cv2.error as e:
            logging.error(f"Errore durante la calibrazione: {e}")
            if "Assertion failed" in str(e):
                logging.warning("L'errore suggerisce dati di calibrazione degeneri. Richiedo il riavvio.")
                if messagebox.askyesno("Errore Calibrazione", 
                                        "Si è verificato un errore critico durante il calcolo. "
                                        "I dati acquisiti potrebbero essere non validi. "
                                        "Vuoi ricominciare il processo di cattura da capo?"):
                    return False
                else:
                    return False
            else:
                messagebox.showerror("Errore OpenCV", f"Errore durante il calcolo della calibrazione:\n{e}")
                return False
                
    def _filter_degenerate_views(self, all_corners: list, all_ids: list) -> Tuple[list, list]:
        filtered_corners, filtered_ids = [], []
        DEGENERATE_ASPECT_RATIO = 20.0
        MIN_CORNERS_THRESHOLD = 10
        
        for i, corners in enumerate(all_corners):
            if not isinstance(corners, np.ndarray) or len(corners) < MIN_CORNERS_THRESHOLD:
                logging.debug(f"Filtraggio vista {i}: numero di punti insufficiente ({len(corners)}) o formato non valido.")
                continue

            points = np.array(corners, dtype=np.float32)
            _, _, w, h = cv2.boundingRect(points)
            if w == 0 or h == 0:
                logging.debug(f"Filtraggio vista {i}: larghezza o altezza zero.")
                continue
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < DEGENERATE_ASPECT_RATIO:
                filtered_corners.append(corners)
                filtered_ids.append(all_ids[i])
            else:
                logging.debug(f"Filtraggio vista {i}: rapporto d'aspetto degenere ({aspect_ratio:.2f}).")
        return filtered_corners, filtered_ids

    # Metodi di Gestione File e Correzione
    
    def set_calibration_params(self, matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple):
        self.camera_matrix = matrix
        self.distortion_coeffs = dist_coeffs
        self.is_calibrated = True
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, image_size, 1, image_size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_coeffs, None, self.new_camera_matrix, image_size, 5)
        logging.info("Parametri di correzione ottica impostati in memoria.")
        logging.debug("Inizio salvataggio parametri su file NPZ.")
        self.save_calibration()

    def save_calibration(self):
        if self.is_calibrated:
            logging.info(f"Salvataggio parametri di calibrazione su '{self.filename}'...")
            np.savez(self.filename, camera_matrix=self.camera_matrix, distortion_coeffs=self.distortion_coeffs, 
                     new_camera_matrix=self.new_camera_matrix, roi=self.roi)
            logging.info("Salvataggio completato.")

    def load_calibration(self):
        if os.path.exists(self.filename):
            try:
                logging.info(f"Trovato file di calibrazione '{self.filename}'. Caricamento in corso...")
                data = np.load(self.filename)
                self.camera_matrix = data['camera_matrix']
                self.distortion_coeffs = data['distortion_coeffs']
                self.new_camera_matrix = data['new_camera_matrix']
                self.roi = data['roi']
                
                if self.new_camera_matrix is not None and self.roi is not None:
                    h, w = self.roi[3], self.roi[2]
                    image_size = (w, h)
                    self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.distortion_coeffs, None, self.new_camera_matrix, image_size, 5)
                
                self.is_calibrated = True
                logging.info("Parametri di calibrazione caricati con successo.")
                return True
            except KeyError as e:
                logging.error(f"Errore durante il caricamento del file di calibrazione: {e}")
                logging.info("Il file NPZ è obsoleto. Proseguire senza caricare.")
                self.is_calibrated = False
                return False
            except Exception as e:
                logging.error(f"Errore imprevisto durante il caricamento: {e}")
                self.is_calibrated = False
                return False
        logging.info("Nessun file di calibrazione trovato.")
        return False

    def correct_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.is_calibrated and frame is not None and self.map1 is not None and self.map2 is not None:
            dst = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
            x, y, w, h = self.roi
            dst = dst[y:y+h, x:x+w]
            return dst
        return frame
