import serial
import pandas as pd
from datetime import datetime
import time

PORT = "COM5"
BAUD = 115200
OUTPUT_FILE = f"swing_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

EXPECTED_COLUMNS = ["t_ms","ax","ay","az","gx","gy","gz","yaw","pitch","roll"]

def is_numeric_row(parts):
    if len(parts) != len(EXPECTED_COLUMNS):
        return False
    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    rows = []
    header_found = False
    columns = EXPECTED_COLUMNS

    print("Leyendo serial... Ctrl+C para detener.")

    try:
        while True:
            raw = ser.readline()
            print("RAW:", raw)

            line = raw.decode(errors="ignore").strip()
            print("LINE:", repr(line))

            if not line:
                continue

            if line.startswith("#"):
                print("Comentario:", line)
                continue

            normalized = line.replace(" ", "")

            # Caso 1: llegó el header
            if not header_found and normalized == ",".join(EXPECTED_COLUMNS):
                header_found = True
                print("Header detectado.")
                continue

            parts = [p.strip() for p in line.split(",")]

            # Caso 2: no llegó header, pero llegó una fila válida
            if not header_found and is_numeric_row(parts):
                header_found = True
                print("No llegó header, pero se detectó fila numérica válida. Continuando...")

            if not is_numeric_row(parts):
                print("Fila descartada:", parts)
                continue

            row = [float(x) for x in parts]
            rows.append(row)
            print("Fila guardada.")

    except KeyboardInterrupt:
        print("\nGuardando archivo...")

    finally:
        ser.close()

    if rows:
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Archivo guardado: {OUTPUT_FILE}")
        print(f"Filas guardadas: {len(rows)}")
    else:
        print("No se guardaron datos.")

if __name__ == "__main__":
    main()