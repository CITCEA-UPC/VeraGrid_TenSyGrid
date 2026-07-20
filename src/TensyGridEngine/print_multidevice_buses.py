from __future__ import annotations

import sys

import VeraGridEngine.api as vge


def main(grid_path: str) -> int:
    grid = vge.open_file(grid_path)

    bus_to_devices = {bus: [] for bus in grid.buses}
    for dev in grid.get_injection_devices_iter():
        if dev.bus is not None:
            bus_to_devices[dev.bus].append(dev)

    found = False
    for bus, devices in bus_to_devices.items():
        if len(devices) > 2:
            found = True
            names = ", ".join(f"{dev.device_type.value}:{dev.name}" for dev in devices)
            print(f"{bus.name}: {len(devices)} injection devices -> {names}")

    if not found:
        print("No buses with more than two injection devices found.")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 print_multidevice_buses.py <grid_path>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
