import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
DEVICE_NAME_DEFAULT = "ESP32_IMU_GOLF"
EXPECTED_COLUMNS = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"]



def is_numeric_row(parts):
    if len(parts) != len(EXPECTED_COLUMNS):
        return False
    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False


async def find_device_address(device_name):
    print(f"Buscando BLE: {device_name}...")
    devices = await BleakScanner.discover(timeout=8.0)
    for device in devices:
        if device.name == device_name:
            print(f"Dispositivo encontrado: {device.name} ({device.address})")
            return device.address
    return None


async def run_capture(device_name):
    address = await find_device_address(device_name)
    if not address:
        print(f"No se encontró el dispositivo BLE '{device_name}'.")
        return

    rows = []
    header_found = False
    partial = ""

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"swing_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def handle_notification(_, data):
        nonlocal partial, header_found, rows

        chunk = data.decode(errors="ignore")
        partial += chunk

        while "\n" in partial:
            line, partial = partial.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                print("Comentario:", line)
                continue

            normalized = line.replace(" ", "")
            if not header_found and normalized == ",".join(EXPECTED_COLUMNS):
                header_found = True
                print("Header detectado.")
                continue

            parts = [p.strip() for p in line.split(",")]

            if not header_found and is_numeric_row(parts):
                header_found = True
                print("No llegó header, pero se detectó fila numérica válida. Continuando...")

            if not is_numeric_row(parts):
                print("Fila descartada:", parts)
                continue

            row = [float(x) for x in parts]
            rows.append(row)
            print(f"Fila guardada ({len(rows)}).")

    print("Conectando BLE...")
    async with BleakClient(address) as client:
        print("Conectado. Escuchando notificaciones... Ctrl+C para detener.")
        await client.start_notify(TX_CHAR_UUID, handle_notification)

        try:
            while True:
                await asyncio.sleep(0.2)
        except KeyboardInterrupt:
            print("\nGuardando archivo...")
        finally:
            await client.stop_notify(TX_CHAR_UUID)

    if rows:
        df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
        df.to_csv(output_file, index=False)
        print(f"Archivo guardado: {output_file}")
        print(f"Filas guardadas: {len(rows)}")
    else:
        print("No se guardaron datos.")



def main():
    parser = argparse.ArgumentParser(description="Captura IMU por BLE desde ESP32-C3")
    parser.add_argument("--name", default=DEVICE_NAME_DEFAULT, help="Nombre BLE del ESP32")
    args = parser.parse_args()

    asyncio.run(run_capture(args.name))


if __name__ == "__main__":
    main()
