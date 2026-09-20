from localforge.hardware import detect_hardware


def test_detect_hardware_returns_sane_values():
    hw = detect_hardware()
    assert hw.ram_gb > 0
    assert hw.free_disk_gb > 0
    assert hw.cpu_cores >= 1
    assert hw.os in {"Darwin", "Linux", "Windows"}
