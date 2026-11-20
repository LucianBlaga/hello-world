"""Minimal ONVIF client: device, media, PTZ and imaging, nothing else.

The vendor says PTZ positioning works "only with absolute coordinates via
ONVIF", so this is how we aim the zoom lens: AbsoluteMove to an exact
pan/tilt/zoom in ONVIF's generic spaces (pan/tilt -1..1, zoom 0..1). GetStatus
may or may not report the real position on this camera; the probe tool checks.

SOAP 1.2 with a WS-Security UsernameToken (password digest). The camera's clock
is read first (GetSystemDateAndTime, no auth needed) so the token's timestamp
matches the camera even if its clock is off.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import requests

NS = {
    "s": "http://www.w3.org/2003/05/soap-envelope",
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tptz": "http://www.onvif.org/ver20/ptz/wsdl",
    "timg": "http://www.onvif.org/ver20/imaging/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
}
_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
_WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
_DIGEST = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
_B64 = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary"


class OnvifError(RuntimeError):
    pass


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find(el: ET.Element | None, *names: str) -> ET.Element | None:
    """Namespace-agnostic path lookup: find(el, 'Position', 'PanTilt')."""
    for name in names:
        if el is None:
            return None
        el = next((c for c in el.iter() if c is not el and _local(c.tag) == name), None)
    return el


def find_all(el: ET.Element | None, name: str) -> list[ET.Element]:
    return [] if el is None else [c for c in el.iter() if _local(c.tag) == name]


def _float(el: ET.Element | None, attr: str, default: float | None = None) -> float | None:
    try:
        return float(el.get(attr)) if el is not None and el.get(attr) is not None else default
    except ValueError:
        return default


def make_security_header(username: str, password: str, created: str, nonce: bytes) -> str:
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return (f'<wsse:Security xmlns:wsse="{_WSSE}" xmlns:wsu="{_WSU}" s:mustUnderstand="1">'
            f"<wsse:UsernameToken><wsse:Username>{escape(username)}</wsse:Username>"
            f'<wsse:Password Type="{_DIGEST}">{digest}</wsse:Password>'
            f'<wsse:Nonce EncodingType="{_B64}">{base64.b64encode(nonce).decode()}</wsse:Nonce>'
            f"<wsu:Created>{created}</wsu:Created></wsse:UsernameToken></wsse:Security>")


def check_security_header(root: ET.Element, users: dict[str, str], now: dt.datetime,
                          max_skew_s: float = 300) -> bool:
    """Server side of the digest check (used by the virtual camera)."""
    token = find(root, "UsernameToken")
    if token is None:
        return False
    user = (find(token, "Username").text or "") if find(token, "Username") is not None else ""
    pw_el, nonce_el, created_el = find(token, "Password"), find(token, "Nonce"), find(token, "Created")
    if user not in users or pw_el is None or nonce_el is None or created_el is None:
        return False
    try:
        nonce = base64.b64decode(nonce_el.text or "")
        created = dt.datetime.strptime((created_el.text or "")[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return False
    if abs((now - created).total_seconds()) > max_skew_s:
        return False
    expect = base64.b64encode(hashlib.sha1(nonce + created_el.text.encode() + users[user].encode()).digest()).decode()
    return expect == (pw_el.text or "")


@dataclass
class Profile:
    token: str
    name: str
    video_source: str | None
    width: int | None
    height: int | None
    ptz_config: str | None


@dataclass
class PTZSpaces:
    pan: tuple[float, float] = (-1.0, 1.0)
    tilt: tuple[float, float] = (-1.0, 1.0)
    zoom: tuple[float, float] = (0.0, 1.0)
    raw: dict = field(default_factory=dict)


@dataclass
class PTZStatus:
    pan: float | None
    tilt: float | None
    zoom: float | None
    pan_tilt_moving: bool | None
    zoom_moving: bool | None
    utc: str | None


class OnvifClient:
    def __init__(self, host: str, username: str, password: str, *, port: int = 80,
                 timeout: float = 5.0, session: requests.Session | None = None):
        self.base = f"http://{host}:{port}"
        self.username = username
        self.password = password
        self.timeout = timeout
        self.http = session or requests.Session()
        self.xaddr = {"device": self.base + "/onvif/device_service"}
        self.clock_offset = 0.0          # camera UTC minus our UTC, seconds
        self._ready = False

    # ------------------------------------------------------------------ plumbing
    def _created(self) -> str:
        t = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=self.clock_offset)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    def call(self, service: str, body_xml: str, auth: bool = True) -> ET.Element:
        url = self.xaddr.get(service)
        if url is None:
            raise OnvifError(f"camera has no ONVIF {service} service")
        header = make_security_header(self.username, self.password, self._created(), os.urandom(16)) if auth else ""
        ns = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
        envelope = f'<?xml version="1.0" encoding="UTF-8"?><s:Envelope {ns}><s:Header>{header}</s:Header><s:Body>{body_xml}</s:Body></s:Envelope>'
        resp = self.http.post(url, data=envelope.encode(), timeout=self.timeout,
                              headers={"Content-Type": "application/soap+xml; charset=utf-8"})
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            raise OnvifError(f"{service}: HTTP {resp.status_code}, reply is not XML") from None
        fault = find(root, "Fault")
        if fault is not None or resp.status_code >= 400:
            text = " ".join(t.strip() for t in fault.itertext() if t.strip()) if fault is not None else ""
            raise OnvifError(f"{service}: {text or 'HTTP %d' % resp.status_code}")
        body = find(root, "Body")
        return body[0] if body is not None and len(body) else root

    def connect(self) -> None:
        """Sync clock, discover service addresses."""
        r = self.call("device", "<tds:GetSystemDateAndTime/>", auth=False)
        utc = find(r, "UTCDateTime")
        if utc is not None:
            d, t = find(utc, "Date"), find(utc, "Time")
            try:
                cam = dt.datetime(int(find(d, "Year").text), int(find(d, "Month").text), int(find(d, "Day").text),
                                  int(find(t, "Hour").text), int(find(t, "Minute").text), int(find(t, "Second").text),
                                  tzinfo=dt.timezone.utc)
                self.clock_offset = (cam - dt.datetime.now(dt.timezone.utc)).total_seconds()
            except (AttributeError, TypeError, ValueError):
                pass
        caps = self.call("device", '<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>')
        for svc in ("Media", "PTZ", "Imaging", "Events"):
            el = find(caps, svc, "XAddr")
            if el is not None and el.text:
                self.xaddr[svc.lower()] = el.text.strip()
        self._ready = True

    # ------------------------------------------------------------------ media
    def profiles(self) -> list[Profile]:
        r = self.call("media", "<trt:GetProfiles/>")
        out = []
        for p in find_all(r, "Profiles"):
            vsc, res, ptz = find(p, "VideoSourceConfiguration"), find(p, "Resolution"), find(p, "PTZConfiguration")
            name = find(p, "Name")
            out.append(Profile(
                token=p.get("token", ""), name=(name.text or "") if name is not None else "",
                video_source=(find(vsc, "SourceToken").text if find(vsc, "SourceToken") is not None else None),
                width=int(find(res, "Width").text) if find(res, "Width") is not None else None,
                height=int(find(res, "Height").text) if find(res, "Height") is not None else None,
                ptz_config=ptz.get("token") if ptz is not None else None))
        return out

    def stream_uri(self, profile: str) -> str:
        r = self.call("media", f'<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>'
                               f'<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
                               f'<trt:ProfileToken>{escape(profile)}</trt:ProfileToken></trt:GetStreamUri>')
        uri = find(r, "Uri")
        return (uri.text or "").strip() if uri is not None else ""

    def ptz_profile(self) -> Profile:
        for p in self.profiles():
            if p.ptz_config:
                return p
        raise OnvifError("no media profile has a PTZ configuration")

    # ------------------------------------------------------------------ PTZ
    def ptz_spaces(self, ptz_config: str) -> PTZSpaces:
        r = self.call("ptz", f"<tptz:GetConfigurationOptions><tptz:ConfigurationToken>{escape(ptz_config)}"
                             f"</tptz:ConfigurationToken></tptz:GetConfigurationOptions>")
        sp = PTZSpaces()
        apt = find(r, "AbsolutePanTiltPositionSpace")
        if apt is not None:
            xr, yr = find(apt, "XRange"), find(apt, "YRange")
            sp.pan = (float(find(xr, "Min").text), float(find(xr, "Max").text))
            sp.tilt = (float(find(yr, "Min").text), float(find(yr, "Max").text))
        az = find(r, "AbsoluteZoomPositionSpace")
        if az is not None:
            xr = find(az, "XRange")
            sp.zoom = (float(find(xr, "Min").text), float(find(xr, "Max").text))
        sp.raw = {"absolute_pan_tilt": apt is not None, "absolute_zoom": az is not None,
                  "continuous": find(r, "ContinuousPanTiltVelocitySpace") is not None}
        return sp

    def absolute_move(self, profile: str, pan: float | None, tilt: float | None, zoom: float | None,
                      speed: float | None = None) -> None:
        pos = ""
        if pan is not None and tilt is not None:
            pos += f'<tt:PanTilt x="{pan:.6f}" y="{tilt:.6f}"/>'
        if zoom is not None:
            pos += f'<tt:Zoom x="{zoom:.6f}"/>'
        spd = ""
        if speed is not None:
            spd = f'<tptz:Speed><tt:PanTilt x="{speed:.3f}" y="{speed:.3f}"/><tt:Zoom x="{speed:.3f}"/></tptz:Speed>'
        self.call("ptz", f"<tptz:AbsoluteMove><tptz:ProfileToken>{escape(profile)}</tptz:ProfileToken>"
                         f"<tptz:Position>{pos}</tptz:Position>{spd}</tptz:AbsoluteMove>")

    def continuous_move(self, profile: str, pan_speed: float, tilt_speed: float, zoom_speed: float = 0.0,
                        timeout_s: float | None = None) -> None:
        to = f"<tptz:Timeout>PT{timeout_s:.1f}S</tptz:Timeout>" if timeout_s else ""
        self.call("ptz", f"<tptz:ContinuousMove><tptz:ProfileToken>{escape(profile)}</tptz:ProfileToken>"
                         f'<tptz:Velocity><tt:PanTilt x="{pan_speed:.3f}" y="{tilt_speed:.3f}"/>'
                         f'<tt:Zoom x="{zoom_speed:.3f}"/></tptz:Velocity>{to}</tptz:ContinuousMove>')

    def stop(self, profile: str) -> None:
        self.call("ptz", f"<tptz:Stop><tptz:ProfileToken>{escape(profile)}</tptz:ProfileToken>"
                         f"<tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>true</tptz:Zoom></tptz:Stop>")

    def status(self, profile: str) -> PTZStatus:
        r = self.call("ptz", f"<tptz:GetStatus><tptz:ProfileToken>{escape(profile)}</tptz:ProfileToken></tptz:GetStatus>")
        pos = find(r, "Position")
        pt, zm = find(pos, "PanTilt"), find(pos, "Zoom")
        ms = find(r, "MoveStatus")
        mpt, mz = find(ms, "PanTilt"), find(ms, "Zoom")
        utc = find(r, "UtcTime")

        def moving(el):
            return None if el is None or not el.text else el.text.strip().upper() == "MOVING"
        return PTZStatus(_float(pt, "x"), _float(pt, "y"), _float(zm, "x"), moving(mpt), moving(mz),
                         utc.text if utc is not None else None)

    def goto_preset(self, profile: str, preset_token: str) -> None:
        self.call("ptz", f"<tptz:GotoPreset><tptz:ProfileToken>{escape(profile)}</tptz:ProfileToken>"
                         f"<tptz:PresetToken>{escape(preset_token)}</tptz:PresetToken></tptz:GotoPreset>")

    def wait_idle(self, profile: str, timeout_s: float = 10.0, poll_s: float = 0.1) -> PTZStatus | None:
        """Poll GetStatus until both axes report IDLE. Returns None if the camera
        never reports a move status (then only the video can tell)."""
        end = time.monotonic() + timeout_s
        st = None
        while time.monotonic() < end:
            st = self.status(profile)
            if st.pan_tilt_moving is None and st.zoom_moving is None:
                return None
            if not st.pan_tilt_moving and not st.zoom_moving:
                return st
            time.sleep(poll_s)
        return st

    # ------------------------------------------------------------------ imaging
    def imaging_settings(self, video_source: str) -> ET.Element:
        return self.call("imaging", f"<timg:GetImagingSettings><timg:VideoSourceToken>{escape(video_source)}"
                                    f"</timg:VideoSourceToken></timg:GetImagingSettings>")

    def set_exposure(self, video_source: str, *, manual: bool, exposure_us: float | None = None,
                     gain: float | None = None) -> None:
        """Fixed fast shutter for plates at night (ExposureTime is in microseconds)."""
        inner = f"<tt:Mode>{'MANUAL' if manual else 'AUTO'}</tt:Mode>"
        if manual and exposure_us is not None:
            inner += f"<tt:ExposureTime>{float(exposure_us):.0f}</tt:ExposureTime>"
        if manual and gain is not None:
            inner += f"<tt:Gain>{float(gain):.1f}</tt:Gain>"
        self.call("imaging", f"<timg:SetImagingSettings><timg:VideoSourceToken>{escape(video_source)}</timg:VideoSourceToken>"
                             f"<timg:ImagingSettings><tt:Exposure>{inner}</tt:Exposure></timg:ImagingSettings>"
                             f"<timg:ForcePersistence>false</timg:ForcePersistence></timg:SetImagingSettings>")

    # ------------------------------------------------------------------ device
    def device_information(self) -> dict:
        r = self.call("device", "<tds:GetDeviceInformation/>")
        return {_local(c.tag): (c.text or "").strip() for c in r}
