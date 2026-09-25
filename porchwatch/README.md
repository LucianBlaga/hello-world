# PorchWatch

Auto-tracking security camera software for the **OBSBOT Tiny 2** (works with any UVC pan/tilt/zoom webcam).

- Detects **people** and **vehicles** (YOLO26 / YOLO11).
- Follows a person with the gimbal, **zooms in on the face** and saves the sharpest close-up.
- Ignores parked cars. Follows **moving vehicles**, zooms toward the plate, and **reads the licence plate** (fast-alpr). A plate only counts as confirmed after the same text has been read several times.
- **Patrol**: sweeps between two edges you set, stopping to look at each position.
- **Follow until it leaves** (optional): stays on a person or car until it's out of the picture.
- Records video **on events** (people and/or moving vehicles, with pre/post-roll), **continuously**, or with the **REC button**. **H.264 / H.265** with a compression slider, encoded on an NVIDIA GPU when available.
- Optional **audio** with boost, rumble filter, noise gate and limiter, plus live listening.
- Web dashboard: live view, camera pad, REC button, event gallery (faces and plates), recordings, and a **Settings page**.
- Watch zone and ignore zones drawn on the picture. Automatic retention (delete after N days / above N GB).

```
watching / patrolling ──person or moving car──► tracking: turn, stop, measure, turn to where it will be,
      ▲                                           │  zoom until the face/plate is big; keep the sharpest
      └──── save snapshots, back to watching ◄────┘  face / vote on plate text (done, lost, timed out, left)
```

## 1. Install

Needs Python 3.10+. On Windows, install Python from python.org and tick "Add python.exe to PATH".

**Windows: keep the folder path short** (for example `C:\PorchWatch`). PyTorch has very long file names, and Windows refuses paths over 260 characters, so installing from a deeply nested folder such as `Desktop\...\hello-world-...\hello-world-...\porchwatch` fails with `No such file or directory ... Long Path`.

```bash
cd porchwatch
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Or double-click `start-windows.bat`, which does this the first time.

With an NVIDIA GPU, install the CUDA build of PyTorch (https://pytorch.org/get-started/locally/) and pick `cuda:0` in Settings → Detection. Detection then becomes several times faster.
Models download automatically the first time they're used. Detection and face models go into `models/`; the plate reader keeps its own cache.

Linux only: `sudo apt install v4l-utils` (used for pan/tilt/zoom).

## 2. Prepare the camera

1. **Turn off OBSBOT's own AI tracking** (OBSBOT Center → AI mode off, or hand gesture/remote). Otherwise the camera and PorchWatch fight over the gimbal.
2. Close OBSBOT Center, Zoom, Teams and anything else using the camera. Only one program can own it.
3. Check PorchWatch can see and move it:

```bash
python -m porchwatch devices     # lists cameras + microphones, and which cameras can pan/tilt/zoom
python -m porchwatch test-ptz    # pans right/left, tilts up/down, zooms 2x/4x, returns home
python -m porchwatch probe       # the camera's control ranges and resolutions
```

PCs with NDI, vMix or OBS have many virtual cameras. Set **Settings → Camera → Camera device** to the camera's **name**, for example `OBSBOT Tiny 2`, not a number. Numbers can change, but a name always finds the real camera and skips "OBSBOT Virtual Camera".

On Windows, PorchWatch reads the pan/tilt/zoom ranges from the camera itself. If it turns the wrong way (e.g. mounted upside down), tick *Invert pan / Invert tilt* in Settings.

## 3. Run

```bash
python -m porchwatch --no-preview      # or double-click start-windows.bat
```

Open **http://localhost:8080**. `--no-preview` skips the extra local video window; everything is in the browser. The same option is in Settings → Web & access.

First-time setup:
1. **Live**: use the arrows and zoom to frame the street, then click **Set current view as home**.
2. Optional patrol: aim at the left end, click **Set as left edge**; aim at the right end, click **Set as right edge**; tick **Patrol**.
3. **Settings → Detection → Zones**: drag a watch zone around the street, and ignore zones over neighbours' windows, your own driveway, a flag that moves in the wind, and so on.
4. **Settings → Recording**: when to record, resolution, fps, codec (h264/h265), compression.
5. **Settings → Storage**: folders (can be another drive, e.g. `D:\PorchWatch\recordings`) and how long to keep things.

All settings are saved in `config.yaml`. Settings marked *restart* briefly reopen the camera. Everything else applies immediately. Values are range-checked, and an invalid value in `config.yaml` is reset to its default at start-up, with a warning in the console.

**● REC** on the Live page records until you press **■ STOP REC**. It works in every mode, even while you steer the camera by hand.

Test without the camera: `python -m porchwatch --video some_street_clip.mp4`.

## 4. Settings reference

| Section | What you can set |
|---|---|
| Camera | device (name or number), capture resolution (up to 4K), capture fps, PTZ backend, home position, camera response delay / turn speed / zoom speed, invert axes |
| Recording | mode (events / continuous / off), record on people, record on moving vehicles, resolution, fps, codec (h264, h265, legacy), encoder (auto/NVIDIA/CPU), compression, pre/post-record, file length, timestamp, folder |
| Audio | on/off, microphone, boost, rumble/wind filter, noise gate + threshold, limiter, quality, sync offset |
| Storage | days to keep recordings / snapshots, max GB for recordings, folders |
| Detection | person / vehicle sensitivity, moving-vehicle threshold, model (YOLO26/YOLO11, n…x), detection image size, CPU/GPU, watch & ignore zones |
| Tracking | on/off, prioritise vehicles, follow until it leaves (+ time limit), follow speed, lead, face / plate zoom target, time zoomed in, max chase, give-up time, re-capture interval, cool-down |
| Patrol | on/off, left / right edge, number of stops, look time per stop |
| Snapshots | min face size, blur filter, plate OCR confidence, plate confirmations, save full scene |
| Web & access | listen address, port, username/password, live-view fps/width, local preview window |

Less common options are only in `config.yaml`, for example `tracking.min_sightings` / `min_avg_conf` (how sure it must be before chasing), `recording.trigger_frames`, and `web.allowed_hosts`.

**Files:**
- `captures/YYYY-MM-DD/*.jpg` and `captures/events.jsonl` (one JSON line per event, including plate text)
- `recordings/YYYY-MM-DD/*.mp4` (`.mkv` for MJPG/XVID with audio)
- `logs/tracking.log` (every tracking decision, see below)

## 5. How tracking works, and when it struggles

- **Stop and measure.** While the gimbal turns, the picture is smeared and doesn't show where things really are. The camera turns to where the target *will be*, confirms from the picture that it has stopped, measures, then moves again. It moves in short steps rather than one smooth pan.
- **No ghosts.** It only chases something detected 6+ times with decent confidence: people for at least half a second, vehicles only once they're clearly moving. Recording triggers on 3 person sightings within a second, not on a single frame.
- **Fast cars** get a wider view (less zoom), so the car can't drive out of the picture while the camera catches up.
- **One camera = one view.** While it follows someone, it doesn't see the rest of the street. A second, fixed wide camera is the usual fix if you need both.
- **Zoom is digital (up to 4×).** Capture at **3840×2160** for plates and faces. Rough rule: reliable OCR needs a plate ~100+ px wide in the 4K frame, so about ≤ 10–15 m away.
- **Frame rate matters.** Below ~10 fps the camera gets fewer measurements and follows cars in bigger steps. If *Processing* fps on the Live page is low, lower the detection image size first, then the model size.
- **Night:** no infrared, so it needs street or porch lights. Slow night shutter speeds blur anything moving.
- **Vegas sun:** keep the camera out of direct sun behind the glass, where it can overheat. Turn on HDR in OBSBOT Center before closing it. Tinted/low-E windows add a colour cast and reflections; put the lens right against the glass.
- **Plates:** Nevada requires front and rear plates, so approaching cars can be read too. Arizona cars (rear plate only) can only be read from behind.
- **It captures faces, it doesn't identify people.** There is no face recognition.

**If tracking misbehaves**, open `logs/tracking.log` and look from `START` to `END`:
- `START` says what was chosen, how often it was seen, and with what confidence.
- `MEASURE` shows where the target and camera were and how much the scene slid (`scene shift`). A large shift while the camera should be still means trees or lights are fooling it.
- `COMMAND` shows each move and zoom.
- `NO MATCH` shows frames where the target wasn't found.

If the camera overshoots, raise **Camera response delay**. If it lags behind, lower it or raise **Camera turn speed**.

## 6. Security

- The dashboard only listens on this PC (`127.0.0.1`) by default. To use it from your phone at home, first set a **password**, then set *Listen on* to `0.0.0.0` and restart. Open it by the PC's IP address, e.g. `http://192.168.1.20:8080`.
- Don't port-forward it to the internet. For remote access use a VPN such as Tailscale. To open it by a name instead of an IP, add the name to `web.allowed_hosts` in `config.yaml`.
- Other websites you visit can't control the camera through your browser: requests must be JSON from this page, and unknown host names are refused.
- The password is never sent back to the browser. On the Settings page, leave the field empty to keep it, or type `NONE` to remove it.

## 7. Privacy / legal (Las Vegas / Nevada)

This is a practical summary, not legal advice.

- **Filming the street from your own property is generally legal in the US.** People on a public street or sidewalk have no reasonable expectation of privacy, and faces and licence plates in public view can be recorded.
- **Don't aim into private spaces.** Keep neighbours' windows, back yards and pool areas out of the picture. Nevada's voyeurism law (NRS 200.604) is about capturing people in private settings. Use **ignore zones** to mask anything like that.
- **Audio is off by default, for a reason.** Nevada law (NRS 200.650) restricts recording private conversations you're not part of. A microphone that picks up people talking on the sidewalk or at a neighbour's door is much less clear-cut than video of a public street. If you turn audio on, don't use it to listen in on conversations; a big boost makes that more likely.
- **HOA / lease rules** may limit where cameras can go or which way they can point.
- **Footage use:** giving clips to Las Vegas Metro (LVMPD) for a crime is fine. Posting strangers' faces or plates online invites trouble (defamation, harassment claims).
- No retention period is required, so pick what suits you in Settings → Storage.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests include a closed-loop simulation. Virtual people and cars move through the world, and the simulated gimbal responds with a delay and limited turn/zoom speed (faster and slower than PorchWatch assumes). In *field mode* the simulator matches what real hardware shows: ~11 fps, no detections while the camera smears the picture, and 20% of frames with nothing detected. The controller has to follow targets, zoom in, capture and return. There's also an end-to-end run of the whole app on a generated video.

| File | Role |
|---|---|
| `porchwatch/camera.py` | frame grabbing, pan/tilt/zoom backends (DirectShow, OpenCV, v4l2), gimbal timing model |
| `porchwatch/detectors.py` | YOLO people/vehicles, YuNet faces, fast-alpr plates |
| `porchwatch/tracker.py` | tells moving cars from parked ones and ghosts from real sightings |
| `porchwatch/controller.py` | watch/patrol → track → zoom → capture state machine, motion meter |
| `porchwatch/storage.py` | snapshots, event log, video recorder, retention |
| `porchwatch/video.py`, `audio.py` | H.264/H.265 via ffmpeg (NVENC), microphone processing and muxing |
| `porchwatch/web.py`, `static/index.html` | dashboard, settings page, validation, request guards |
| `porchwatch/app.py` | main loop wiring it together |
