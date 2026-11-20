# PorchWatch 2

New software for the **Besview BC-IQ56YDW-8MQ** dual-lens PTZ camera
(fixed 3.6 mm wide lens + 20x zoom ball). Built from the vendor's HTTP API
document (ver. 1.7, 109 pages), the datasheet and the vendor's answers.

## Status: part 1 of 4

| Part | What | State |
|---|---|---|
| 1 | Camera connection (HTTP API, ONVIF PTZ, alerts), virtual camera, probe tool | done |
| 2 | Recording: both lenses copied from the camera, 5 s pre-roll, event folders for Google Drive | next |
| 3 | Detection on the wide lens, aiming the zoom lens, best plate / face picture | |
| 4 | Web page: live view, events, settings | |

Every event will be one folder: `wide.mp4`, `ptz.mp4`, `plate.jpg` (or `face.jpg`).

## When the camera arrives

1. Connect it to your network, find its IP address, open its web page and **change the password**.
2. In your router, **block the camera's internet access** (its firmware has known security holes).
3. Run the probe from this folder:

   ```
   pip install -r requirements.txt
   python -m porchwatch2.tools.probe --host CAMERA_IP --password YOUR_PASSWORD --setup
   ```

   The zoom lens moves during the test. Walk in front of the camera during the
   last 30 s. Send back `probe-report/probe-report.json` and the snapshots.

## Layout

- `porchwatch2/camera/hapi.py`: HTTP API client (all endpoints from the PDF)
- `porchwatch2/camera/onvif.py`: ONVIF client (absolute PTZ moves, status, exposure)
- `porchwatch2/camera/events.py`: receives the camera's alerts, keeps the subscription alive
- `porchwatch2/sim/`: virtual camera (same API over a simulated street)
- `porchwatch2/tools/probe.py`: first-contact check of the real camera

Virtual camera: `python -m porchwatch2.sim.camera --port 8080`. Tests: `pytest`.
