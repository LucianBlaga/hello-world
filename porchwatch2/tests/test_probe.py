import json
import threading

from porchwatch2.tools import probe


def test_probe_runs_against_the_virtual_camera(make_vcam, tmp_path, monkeypatch):
    cam = make_vcam(detect_vehicles=True)
    monkeypatch.setattr(probe, "AF_WINDOW_S", 0.2)
    with cam.lock:
        person = cam.scene.add_person(1, x=-40.0)

    def walk_in():                            # someone walks into view during the alert test
        import time
        while not cam.subscriptions:          # wait for the probe's alert step
            time.sleep(0.05)
        time.sleep(0.2)
        with cam.lock:
            person.x = 0.0
        cam.tick(0.05)
    threading.Thread(target=walk_in, daemon=True).start()
    rc = probe.main(["--host", cam.host, "--port", str(cam.port), "--password", "123456", "--setup",
                     "--alerts", "2", "--alert-port", "0", "--out", str(tmp_path)])
    assert rc == 0
    rep = json.loads((tmp_path / "probe-report.json").read_text())
    assert rep["default_password"] is True
    assert rep["detection_types"] == ["HUMAN", "CAR"]
    assert rep["ptz_position_readback"] == "yes"
    assert all("idle_after_s" in m for m in rep["ptz_moves"])
    assert "TrackAutoOff" in rep["builtin_tracking_off"]
    assert any(a["sub"] == "HumanShapeDetect" and a["active"] for a in rep["alerts"])
    assert (tmp_path / "snapshot-ch2.jpg").exists()


def test_probe_spots_missing_readback(make_vcam, tmp_path, monkeypatch):
    cam = make_vcam(report_position=False)
    monkeypatch.setattr(probe, "AF_WINDOW_S", 0.1)
    probe.main(["--host", cam.host, "--port", str(cam.port), "--password", "123456",
                "--alerts", "0", "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "probe-report.json").read_text())
    assert rep["ptz_position_readback"].startswith("no")
