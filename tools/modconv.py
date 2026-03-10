# source: https://github.com/zeittresor/protracker_mod_choral_generator
import sys
import struct
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QFileDialog, QComboBox, QLabel, QMessageBox)
from PyQt6.QtCore import Qt

class ModChiper(QWidget):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Amiga MOD Instrument Replacer')
        self.setFixedWidth(400)
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI'; }
            QPushButton { background-color: #444; border: 1px solid #666; padding: 10px; border-radius: 4px; }
            QPushButton:hover { background-color: #555; }
            QComboBox { background-color: #333; color: white; padding: 5px; }
        """)

        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("1. Original MOD Datei auswählen:"))
        self.btn_load = QPushButton("DATEI ÖFFNEN")
        self.btn_load.clicked.connect(self.load_file)
        layout.addWidget(self.btn_load)

        self.lbl_status = QLabel("Keine Datei geladen")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        layout.addWidget(QLabel("2. Neues Einheits-Instrument wählen:"))
        self.combo_inst = QComboBox()
        # Hier sind deine gewünschten Instrumente
        self.combo_inst.addItems(["Piano", "Bajo", "Panflute", "Ziehharmonika", "Theremin", "Tuba", "8-Bit Square"])
        layout.addWidget(self.combo_inst)

        self.btn_save = QPushButton("3. ALS NEUE MOD SPEICHERN")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.process_and_save)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)

    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "MOD laden", "", "Amiga Tracker MOD (*.mod)")
        if file:
            self.input_file = file
            self.lbl_status.setText(f"Geladen: {file.split('/')[-1]}")
            self.btn_save.setEnabled(True)

    def generate_wave(self, name, length):
        # Basis-Parameter für hörbare Töne
        sr = 8287 # Amiga C-3 Frequenz
        t = np.arange(length) / sr
        freq = 440.0 # Standard A
        
        if name == "Piano":
            # Sinus mit perkussivem Decay
            data = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)
        elif name == "Bajo":
            # Tiefer Sinus mit extrem kurzem Decay
            data = np.sin(2 * np.pi * (freq/2) * t) * np.exp(-8 * t)
        elif name == "Panflute":
            # Sinus gemischt mit weißem Rauschen
            noise = np.random.uniform(-0.3, 0.3, length)
            data = (np.sin(2 * np.pi * freq * t) * 0.7) + noise
        elif name == "Ziehharmonika":
            # Summe aus zwei Rechteckschwingungen für Reeds-Sound
            data = 0.5 * (np.sign(np.sin(2 * np.pi * freq * t)) + 0.3 * np.sign(np.sin(2 * np.pi * freq * 2 * t)))
        elif name == "Theremin":
            # Sinus mit starkem Pitch-Vibrato (5Hz)
            vibrato = np.sin(2 * np.pi * 5 * t) * 15
            data = np.sin(2 * np.pi * (freq + vibrato) * t)
        elif name == "Tuba":
            # Tiefe Sägezahnschwingung
            data = 2 * ((t * (freq/2)) % 1.0) - 1
        else: # 8-Bit Square
            data = np.sign(np.sin(2 * np.pi * freq * t))

        # Normalisierung auf 8-Bit Signed (-128 bis 127)
        if np.max(np.abs(data)) > 0:
            data = (data / np.max(np.abs(data)) * 127).astype(np.int8)
        else:
            data = np.zeros(length, dtype=np.int8)
            
        return data.tobytes()

    def process_and_save(self):
        output_file, _ = QFileDialog.getSaveFileName(self, "MOD Speichern", "output_chip.mod", "MOD (*.mod)")
        if not output_file: return
        
        try:
            with open(self.input_file, 'rb') as f:
                mod_data = bytearray(f.read())

            # Versuche Pattern-Anzahl zu bestimmen (Offset 950-1080)
            num_patterns = max(mod_data[952:1080]) + 1
            sample_data_start = 1084 + (num_patterns * 1024)
            
            selected_instrument = self.combo_inst.currentText()
            current_pos = sample_data_start
            
            # Alle 31 Samples im Header durchlaufen
            for i in range(31):
                h_off = 20 + (i * 30)
                # Länge ist in Words gespeichert (*2 für Bytes)
                len_words = struct.unpack('>H', mod_data[h_off+22 : h_off+24])[0]
                len_bytes = len_words * 2
                
                if len_bytes > 0:
                    # Generiere das gewählte Instrument in der exakten Länge des Originals
                    new_sample = self.generate_wave(selected_instrument, len_bytes)
                    
                    # Sicherheitscheck für Dateiende
                    if current_pos + len_bytes <= len(mod_data):
                        mod_data[current_pos : current_pos + len_bytes] = new_sample
                    
                    current_pos += len_bytes

            with open(output_file, 'wb') as f:
                f.write(mod_data)
            
            QMessageBox.information(self, "Erfolg", f"Alle Instrumente wurden durch '{selected_instrument}' ersetzt!")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Da lief was schief: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ModChiper()
    ex.show()
    sys.exit(app.exec())
