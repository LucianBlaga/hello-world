# PorchWatch

Auto-tracking security camera software for the **OBSBOT Tiny 2** (works with any UVC pan/tilt/zoom webcam).

- Detects **people** and **vehicles** (YOLO).
- Follows a person with the gimbal, **zooms in on the face** and saves the sharpest close-up.
- Ignores parked cars. Follows **moving vehicles**, zooms toward the plate, and **reads the licence plate** (fast-alpr). A plate only counts as confirmed after the same text has been read several times.
- Records video **on events** (with pre/post-roll) or **continuously**, at its own resolution and frame rate.
- Web dashboard: live view, camera pad, event gallery (faces and plates), recordings, and a **Settings page**.
- Watch zone and ignore zones drawn on the picture. Automatic retention (delete after N days / above N GB).

```
watching (wide, "home") ──person or moving car──► tracking: centre it, zoom until face/plate is big
      ▲                                              │  collect sharpest face / vote on plate text
      └────── save snapshots, return home ◄──────────┘  (done, plate confirmed, lost, or timed out)
```

## 1. Install

Needs Python 3.10+. On Windows, install Python from python.org and tick "Add to PATH".

**Windows: keep the folder path short** (for example `C:\PorchWatch`). PyTorch has very long file names, and Windows refuses paths over 260 characters, so installing from a deeply nested folder such as `Desktop\...\hello-world-...\hello-world-...\porchwatch` fails with `No such file or directory ... Long Path`.

```bash
cd porchwatch
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/) and detection becomes much faster.
The detection, face and plate models download automatically the first time you run the program.

Linux only: `sudo apt install v4l-utils` (used for pan/tilt/zoom).

## 2. Prepare the camera

1. **Turn off OBSBOT's own AI tracking** (OBSBOT Center → AI mode off, or hand gesture/remote). Otherwise the camera and PorchWatch fight over the gimbal.
2. Close OBSBOT Center, Zoom, Teams and anything else using the camera. Only one program can own it.
3. Check PorchWatch can see and move it:

```bash
python -m porchwatch probe       # lists the camera's controls and resolutions
python -m porchwatch test-ptz    # pans right/left, tilts up/down, zooms 2x/4x, returns home
```

If it turns the wrong way (e.g. mounted upside down), tick *Invert pan / Invert tilt* in Settings.
If it moves too little or too much, compare the `pan_absolute` / `tilt_absolute` / `zoom_absolute` ranges from `probe` with `raw_pan_range` / `raw_tilt_range` / `raw_zoom_range` in `config.yaml`.

## 3. Run

```bash
python -m porchwatch             # or double-click start-windows.bat
```

Open **http://localhost:8080** for the dashboard. A local preview window also opens (keys: `q` quit, `w/a/s/d` move, `+/-` zoom, `h` home, `p` pause).

First-time setup in the dashboard:
1. **Live**: use the arrows and zoom to frame the street, then click **Set current view as home**.
2. **Settings → Detection → Zones**: drag a watch zone around the street, and add ignore zones over your own driveway, neighbours' windows, a road sign that flaps in the wind, and so on.
3. **Settings → Recording**: choose events or continuous, resolution, fps, codec, and pre/post-record.
4. **Settings → Storage**: set how many days to keep footage and snapshots.

All settings are saved in `config.yaml`. Settings marked *restart* briefly reopen the camera. Everything else applies immediately.

To watch from your phone at home: Settings → Web & access → set a **password**, set *Listen on* to `0.0.0.0`, restart, then open `http://<pc-ip>:8080`. Don't expose it to the internet with port forwarding. Use a VPN such as Tailscale if you need remote access.

Test without the camera: `python -m porchwatch --video some_street_clip.mp4`.

## 4. Settings reference

| Section | What you can set |
|---|---|
| Camera | capture resolution (up to 4K), capture fps, device, PTZ backend, home position, invert axes |
| Recording | mode (events / continuous / off), recording resolution, recording fps, codec, pre-record, post-record, file length, timestamp overlay, folder |
| Storage | days to keep recordings / snapshots, max GB for recordings |
| Detection | person / vehicle sensitivity, moving-vehicle threshold, model size, image size, CPU/GPU, watch & ignore zones |
| Tracking | on/off, prioritise vehicles, follow speed, lead, face / plate zoom target, time zoomed in, max chase, give-up time, re-capture interval |
| Snapshots | min face size, blur filter, plate OCR confidence, plate confirmations, save full scene |
| Web & access | listen address, port, username/password, live-view fps/width, local preview window |

Output: `captures/YYYY-MM-DD/*.jpg` plus `captures/events.jsonl` (one JSON line per event, including plate text), and `recordings/YYYY-MM-DD/*.mp4`.

## 5. What to expect (honest limits)

- **One camera = one view.** While it's zoomed on someone, it doesn't see the rest of the street. Chases are kept short (a few seconds) for that reason. A second, fixed wide camera is the usual fix if you need both.
- **Zoom is digital (up to 4×).** Set capture to **3840×2160** for plates and faces. At 1080p the 4× zoom is mostly upscaling. Rough rule: for reliable OCR a plate needs ~100+ px width in the 4K frame, so about ≤ 10–15 m away.
- **Fast cars:** the gimbal is quick, but USB control has latency. Cars passing at 50 km/h close to the house cross the view in about a second. Expect plate reads mostly from slower cars, cars turning, or cars approaching head-on. Test with your own street and tune *Lead moving targets* and *Follow speed*.
- **Vegas sun:** keep the camera out of direct sun behind the glass. The window heats up and the camera will overheat or throttle, so check OBSBOT's operating-temperature spec. Afternoon sun shining into the lens washes out faces, so turn on HDR in OBSBOT Center before closing it. Tinted or low-E windows add a colour cast and reflections.
- **Plates:** Nevada requires front *and* rear plates, so cars coming toward the house show a plate too. Out-of-state cars (e.g. Arizona, rear plate only) are only readable from behind.
- **Night:** the Tiny 2 has no infrared, so at night you need street lighting or a porch light. Headlights will glare plates out.
- **Through a window:** put the lens right against the glass and turn off inside lights, or reflections ruin the image. The Tiny 2 isn't weatherproof, so keep it indoors.
- **It captures faces, it doesn't identify people.** There is no face recognition. You get dated close-ups you can review or hand to the police.
- Tracking was developed against a simulated camera. Expect to tune *Follow speed* on real hardware.

## 6. Privacy / legal (Las Vegas / Nevada)

This is a practical summary, not legal advice.

- **Filming the street from your own property is generally legal in the US.** People on a public street or sidewalk have no reasonable expectation of privacy, and faces and licence plates in public view can be recorded.
- **Don't aim into private spaces.** Keep neighbours' windows, back yards and pool areas out of the picture. Nevada's voyeurism law (NRS 200.604) is about capturing people in private settings. Use **ignore zones** to mask anything like that.
- **No audio.** PorchWatch records video only, which keeps you clear of wiretap and eavesdropping rules.
- **HOA / lease rules** may limit where cameras can go or which way they can point. Check yours if you have one.
- **Footage use:** giving clips to Las Vegas Metro (LVMPD) for a crime is fine. Posting strangers' faces or plates online invites trouble (defamation, harassment claims). Keep it for evidence.
- No retention period is required, so pick what suits you in Settings → Storage.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests include a closed-loop simulation. Virtual people and cars move through the world, the virtual camera's pan/tilt/zoom decides where they appear, and the controller has to follow them, zoom in, capture, and return home.

| File | Role |
|---|---|
| `porchwatch/camera.py` | frame grabbing thread, pan/tilt/zoom backends (DirectShow via OpenCV, v4l2) |
| `porchwatch/detectors.py` | YOLO people/vehicles, YuNet faces, fast-alpr plates |
| `porchwatch/tracker.py` | tells moving cars from parked ones while at home |
| `porchwatch/controller.py` | watch → track → zoom → capture → home state machine |
| `porchwatch/storage.py` | snapshots, event log, video recorder, retention |
| `porchwatch/web.py`, `static/index.html` | dashboard and settings page |
| `porchwatch/app.py` | main loop wiring it together |
