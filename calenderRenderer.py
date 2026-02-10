import math

from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple, Mapping

import os
from zoneinfo import ZoneInfo
import re

import requests_cache
from icalevents.icalevents import events as icalevents_events
from icalevents.icalevents import parse_events
from icalendar import Calendar

from PIL import Image, ImageDraw, ImageFont
import heapq

from fontHelper import STMSFont

fontPath = "fonts"
FONTS = {
    8 : STMSFont.from_cpp(os.path.join(fontPath,"font8.cpp"), hgap=1),
    12 : STMSFont.from_cpp(os.path.join(fontPath,"font12.cpp"), hgap=1),
    16 : STMSFont.from_cpp(os.path.join(fontPath,"font16.cpp"), hgap=1),
    20 : STMSFont.from_cpp(os.path.join(fontPath,"font20.cpp"), hgap=1),
    24 : STMSFont.from_cpp(os.path.join(fontPath,"font24.cpp"), hgap=1)
}

HTTP = requests_cache.CachedSession(
    cache_name="cache/http_cache",
    backend="sqlite",
    expire_after=600,  # seconds
)



def stack_events(
    formatted: List[Dict[str, Any]],
    max_lanes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Stack calendar events into lanes (zindex) with a maximum lane limit.

    Each event must have:
      - event["start"]: datetime
      - event["end"]: datetime

    Returns:
      - stacked events: [{ "zindex": int, "event": event }]
      - lanes_used: actual number of lanes used (<= max_lanes)

    Policy when overlaps exceed max_lanes:
      - events beyond capacity are assigned zindex = max_lanes - 1
        (i.e. visually clipped into the last lane)
    """

    if not formatted or max_lanes <= 0:
        return [], 0

    # events = sorted(formatted, key=lambda e: e["start"])

    active: List[Tuple[datetime, int]] = []

    free: List[int] = []

    next_lane = 0
    lanes_used = 0

    stacked: List[Dict[str, Any]] = []

    for ev in formatted:
        start = ev["start"]
        end = ev["end"]

        while active and active[0][0] <= start:
            _, lane = heapq.heappop(active)
            heapq.heappush(free, lane)

        if free:
            lane = heapq.heappop(free)
        elif next_lane < max_lanes:
            lane = next_lane
            next_lane += 1
            lanes_used = max(lanes_used, next_lane)
        else:
            lane = max_lanes - 1

        if lane < max_lanes:
            heapq.heappush(active, (end, lane))

        stacked.append({
            "zindex": lane,
            "event": ev
        })

    return stacked, lanes_used

def cacluate_time_delta(start: datetime, end:datetime) -> int:
    return (end-start).seconds // 60

def download_ics(sources: list[dict], output_dir: str) -> list[tuple[str, str]]:
    """
    Downloads each URL source into output_dir/{name}.ics (binary-safe).
    Returns list of (name, filepath).
    """
    os.makedirs(output_dir, exist_ok=True)

    output: list[tuple[str, str]] = []
    for item in sources:
        url = item.get("url")
        name = item.get("name", "calendar")

        if not url:
            continue  # only downloads URL sources

        r = HTTP.get(url, stream=True, timeout=20, headers={"Accept": "text/calendar"})
        r.raise_for_status()

        path = os.path.join(output_dir, f"{name}.ics")
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        output.append({
            "name" : name,
            "file" : os.path.join(output_dir, f"{name}.ics")
        })

    return output


def merge_calendars(
    sources: list[dict],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    tz: str = "Europe/London",  # Europe/England is not a valid IANA tz
    dedupe: bool = True,
) -> list[Any]:
    """
    Uses icalevents to EXPAND + MERGE event instances (good for rendering/stacking).
    Returns a list of icalevents Event objects.

    sources items can include:
      {"name": "...", "url": "...", "fix_apple": bool}
      {"name": "...", "file": "...", "fix_apple": bool}
      {"name": "...", "string_content": "...", "fix_apple": bool}
    """
    tzinfo = ZoneInfo(tz)
    merged: list[Any] = []

    for src in sources:
        es = icalevents_events(
            url=src.get("url"),
            file=src.get("file"),
            string_content=src.get("string_content"),
            start=start,
            end=end,
            fix_apple=bool(src.get("fix_apple", False)),
            tzinfo=tzinfo,
            sort=False,
            strict=False,
        )

        for e in es:
            setattr(e, "_source", src.get("name", "unknown"))
            merged.append(e)

    merged.sort(key=lambda e: (e.start, e.end))

    if not dedupe:
        return merged

    seen = set()
    out: list[Any] = []
    for e in merged:
        uid = getattr(e, "uid", None) or getattr(e, "summary", None) or getattr(e, "title", None)
        key = (uid, e.start, e.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)

    return out

def _read_source_ics_bytes(src: dict) -> bytes:
    """
    Reads ICS content from a source dict and returns raw bytes.
    Supports: url, file, string_content (str or bytes).
    """
    if src.get("string_content") is not None:
        sc = src["string_content"]
        if isinstance(sc, bytes):
            return sc
        return str(sc).encode("utf-8", errors="replace")

    if src.get("file"):
        with open(src["file"], "rb") as f:
            return f.read()

    if src.get("url"):
        r = HTTP.get(src["url"], timeout=20, headers={"Accept": "text/calendar"})
        r.raise_for_status()
        return r.content

    raise ValueError(f"Source '{src.get('name', 'unknown')}' has no url/file/string_content.")


def merge_sources_to_ics(
    sources: list[dict],
    *,
    prodid: str = "-//Epaper Active Calendar//Merged//EN",
    method: str = "PUBLISH",
    dedupe_events: bool = True,
    keep_timezones: bool = True,
) -> str:
    """
    Merges multiple calendar SOURCES into a single VCALENDAR and returns ICS PLAIN TEXT.

    This uses `icalendar` for proper serialization:
      - Parses each ICS source into a Calendar object
      - Copies VEVENTs (and optionally VTIMEZONEs) into a new Calendar
      - Optionally de-duplicates VEVENTs (UID + DTSTART + DTEND)
      - Serializes with Calendar.to_ical() -> bytes -> UTF-8 string

    Returns:
      str (ICS plain text)
    """
    merged = Calendar()
    merged.add("prodid", prodid)
    merged.add("version", "2.0")
    merged.add("calscale", "GREGORIAN")
    merged.add("method", method)

    # Track timezones by TZID to avoid duplicates
    tz_seen: set[str] = set()

    # Dedupe key for events
    ev_seen: set[tuple[str, str, str]] = set()

    for src in sources:
        raw = _read_source_ics_bytes(src)
        cal = Calendar.from_ical(raw)

        # Optionally carry over VTIMEZONE blocks
        if keep_timezones:
            for comp in cal.walk("VTIMEZONE"):
                tzid = str(comp.get("TZID", "")).strip()
                if tzid and tzid in tz_seen:
                    continue
                merged.add_component(comp)
                if tzid:
                    tz_seen.add(tzid)

        # Copy VEVENTs
        for ev in cal.walk("VEVENT"):
            if not dedupe_events:
                merged.add_component(ev)
                continue

            uid = str(ev.get("UID", "")).strip()

            # DTSTART/DTEND can be date or datetime; normalize to string for stable key
            dtstart = ev.get("DTSTART")
            dtend = ev.get("DTEND")

            dtstart_s = str(dtstart.dt) if dtstart is not None else ""
            dtend_s = str(dtend.dt) if dtend is not None else ""

            key = (uid, dtstart_s, dtend_s)
            if key in ev_seen:
                continue
            ev_seen.add(key)
            merged.add_component(ev)

    # Serialize to ICS plain text
    return merged.to_ical().decode("utf-8", errors="replace")

def GetDateRange(mergedText, dateStart, length = timedelta(1)):
    data = [x.get_data() for x in parse_events(mergedText,dateStart,dateStart + length)]
    print("got")
    formatted = []

    for event in data:
        formatted.append({
            "start" : event["start"],
            "end" : event["end"],
            "summary" : event["summary"],
            "location" : event["location"]
        })

    formatted.sort(key=lambda x: x["start"])

    return formatted

# Bar Calendar

def compute_event_max_concurrency(stack: list[dict]) -> dict[int, int]:
    """
    Given stacked events (output of stack_events: [{"zindex": lane, "event": {...}}, ...]),
    compute, for each event index, the maximum number of concurrent events during its lifetime.

    Returns:
        max_conc: {event_index: max_concurrency_int}
    """
    # Index events so we can annotate them
    indexed = [(i, item["event"]["start"], item["event"]["end"]) for i, item in enumerate(stack)]

    # Sweep points: (time, kind, event_index)
    # end before start at same timestamp so touching events aren't counted as overlap
    points = []
    for i, s, e in indexed:
        points.append((e, 0, i))  # 0=end
        points.append((s, 1, i))  # 1=start
    points.sort()

    active = set()
    max_conc = {i: 1 for i, _, _ in indexed}

    for t, kind, i in points:
        if kind == 0:
            # end
            active.discard(i)
        else:
            # start
            active.add(i)

        # Current concurrency applies to all currently active events
        conc = len(active)
        if conc > 1:
            for j in active:
                if conc > max_conc[j]:
                    max_conc[j] = conc

    return max_conc

def track_widths_int(W: int, N: int, g: int) -> list[int]:
    usable = W - (N - 1) * g
    base = usable // N
    remainder = usable % N
    return [
        base + (1 if i < remainder else 0)
        for i in range(N)
    ]

def CreateBarCalender(formatted, width, height, barColor, NowLineColor, fontSize: int | None = 8):
    barWidth = 8

    # barColor = 8, (0,255,255)
    # NowLineColor = (255,0,0)

    padding = 4

    max_lanes = (width + padding) // (barWidth + padding)
    maxZIndex = max_lanes - 1


    image = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(image)

    font = FONTS[fontSize]

    # Formatting

    # Caculate min/pixel
    if len(formatted) != 0:
        earliest, latest = formatted[0]["start"], formatted[-1]["end"]
    else:
        earliest, latest = None, None

    timeline_top_y = 0   # set if your timeline isn't at y=0
    timeline_x0 = 0      # set to timeline left x
    timeline_w = width   # width of the timeline area (not the whole screen if different)

    # Minutes per pixel (float)
    total_minutes = cacluate_time_delta(earliest, latest)  # assume minutes
    mpp = total_minutes / float(height)

    # Stack (lane assignment)
    stack, used = stack_events(formatted, maxZIndex)

    # Compute max concurrency per event (this enables "expand to full width when alone")
    max_conc = compute_event_max_concurrency(stack)

    # Build bars with time-based Y and dynamic lane widths
    bars = []
    for idx, item in enumerate(stack):
        ev = item["event"]
        lane = item["zindex"]

        start_min = (ev["start"] - earliest).total_seconds() / 60.0
        end_min   = (ev["end"]   - earliest).total_seconds() / 60.0

        # Clamp to visible window
        start_min = max(0.0, min(total_minutes, start_min))
        end_min   = max(0.0, min(total_minutes, end_min))
        if end_min < start_min:
            end_min = start_min

        y0 = timeline_top_y + (start_min / mpp)
        y1 = timeline_top_y + (end_min   / mpp)

        y0 = int(math.floor(y0))
        y1 = int(math.ceil(y1))
        if y1 <= y0:
            y1 = y0 + 1

        k = max_conc[idx]
        lane_w = (timeline_w - (k - 1) * padding) / float(k)
        lane_pos = min(lane, k - 1)

        x0 = timeline_x0 + lane_pos * (lane_w + padding)
        x1 = x0 + lane_w

        x0 = int(round(x0))
        x1 = int(round(x1))

        bars.append({
            "zindex": lane,
            "k": k,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "label": ev["start"].strftime("%H:%M"),  # <-- HH:MM label
        })

    for bar in bars:
        w = max(1, bar["x1"] - bar["x0"])
        h = max(1, bar["y1"] - bar["y0"])
        radius = max(2, min(10, w // 4, h // 3))

        # Rectangle
        draw.rounded_rectangle(
            (bar["x0"], bar["y0"], bar["x1"], bar["y1"]),
            radius=radius,
            fill=barColor,
            outline=(255, 255, 255)
        )

        # Centered label (HH:MM) — draw only if it will fit
        text = bar["label"]

        tw, th = font.getsize(text)   # <-- use font metrics, not draw.textbbox

        # Small padding so text doesn't touch edges
        if tw <= w - 2 and th <= h - 2:
            cx = bar["x0"] + (w - tw) // 2
            cy = bar["y0"] + (h - th) // 2 + 2
            draw.text((cx, cy), text, font=font, fill=(0, 0, 0))

    now = datetime.now(earliest.tzinfo)
    offsettedNow = now - earliest

    # Only draw "now" line if now is within [earliest, latest]
    if offsettedNow >= timedelta(0):
        minutes_since_start = offsettedNow.total_seconds() / 60.0
        y_now = timeline_top_y + (minutes_since_start / mpp)

        y_now = int(round(y_now))

        # Optional clamp to image bounds
        if 0 <= y_now < height:
            draw.line((0, y_now, width, y_now), fill=NowLineColor)

    return image

# CreateBarCalender(formatted, 75, 255, (0,255,255), (255,0,0))

# Day Summary
def wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    """
    Wrap a single logical line into multiple lines that fit max_width.
    Returns a list of lines (at least 1).
    """
    words = text.split(" ")
    if not words:
        return []

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        w, _ = font.getsize(trial)
        if w <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines

def render_upcoming_event_list(
    formatted: List[Dict[str, Any]],
    width: int,
    height: int,
    fontSize: int,
    *,
    now: Optional[datetime] = None,
    padding: int = 2,
    line_gap: int = 1,
    section_gap: int = 5,
    bg: Tuple[int, int, int] = (255, 255, 255),
    fg: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    if fontSize not in FONTS:
        raise KeyError(f"fontSize={fontSize} not found in fonts keys={sorted(FONTS.keys())}")
    font = FONTS[fontSize]

    if not formatted:
        return image

    if now is None:
        sample = formatted[0]["start"]
        now = datetime.now(sample.tzinfo) if sample.tzinfo else datetime.now()

    y = padding
    lh = font.getsize("00:00")[1]
    usable_w = width - 2 * padding

    for item in formatted:
        if item["end"] <= now:
            continue

        start = item["start"].strftime("%H:%M")
        end = item["end"].strftime("%H:%M")

        logical_lines = [
            f"{start}-{end}",
            str(item["summary"]).strip(),
        ]
        if item.get("location"):
            logical_lines.append(str(item["location"]).strip())

        # ---- NEW: wrap logical lines ----
        wrapped_lines: list[str] = []
        for ln in logical_lines:
            wrapped_lines.extend(wrap_text(draw, ln, font, usable_w))

        if not wrapped_lines:
            continue

        # ---- Measure wrapped chunk height ----
        chunk_h = (
            len(wrapped_lines) * lh
            + (len(wrapped_lines) - 1) * line_gap
            + section_gap
        )

        if y + chunk_h > height - padding:
            break  # nothing further will fit

        # ---- Draw wrapped chunk ----
        x = padding
        for ln in wrapped_lines:
            draw.text((x, y), ln, font=font, fill=fg)
            y += lh + line_gap

        y += section_gap

    return image

# now = datetime.now(formatted[0]["start"].tzinfo)
# render_upcoming_event_list(formatted,150,225,8,padding=5)

#Title
def fit_font_to_box(
    draw,
    text: str,
    font_path: str,
    box_w: int,
    box_h: int,
    *,
    max_size: int = 200,
    min_size: int = 6,
) -> ImageFont.FreeTypeFont:
    """
    Find the largest font size that fits inside (box_w, box_h).
    """
    low, high = min_size, max_size
    best_font = None

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= box_w and th <= box_h:
            best_font = font
            low = mid + 1   # try larger
        else:
            high = mid - 1  # too big

    if best_font is None:
        return ImageFont.truetype(font_path, min_size)

    return best_font

def format_date_cn(dt: datetime) -> str:
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{dt.year}年{dt.month}月{dt.day}日 {weekdays[dt.weekday()]}"

def Date_Time_Chinese(width,height, time):
    text = format_date_cn(time)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font = fit_font_to_box(
        draw,
        text,
        "fonts/NotoSerifCJKsc-Regular.otf",
        box_w=width,
        box_h=height,
        max_size=80
    )

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    cx = (width - tw) // 2 - bbox[0]
    cy = (height - th) // 2 - bbox[1]

    draw.text((cx, cy), text, font=font, fill=(0, 0, 0))
    
    return image

# Date_Time_Chinese(350, 50)

def render_event_summary_image(
    template: str,
    formatted: List[Dict[str, Any]],
    width: int,
    height: int,
    fontSize: int,
    *,
    padding: int = 5,
    line_gap: int = 1,
    bg: Tuple[int, int, int] = (255, 255, 255),
    fg: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    - Uses pixel-perfect bitmap font from global FONTS[fontSize] (NO font-size fitting).
    - Replaces ONLY:
        <totalTimeInEvents>, <totalTimeNotInEvents>, <TotalEvents>
    - Respects '\\n'
    - Wraps ONLY on spaces (never breaks a word).
      If any single word is wider than the available width, that logical line is NOT drawable and is skipped.
    - Draws as much as fits vertically.
    """

    # -------- helpers (self-contained) --------
    def format_timedelta_hm(td: timedelta) -> str:
        total_minutes = int(td.total_seconds() // 60)
        h, m = divmod(total_minutes, 60)
        return f"{h}h {m}m"

    def wrap_spaces_only(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str] | None:
        """
        Wrap text on spaces only.
        Never breaks a word.
        If any single word exceeds max_width, returns None (not drawable).
        """
        text = text.strip()
        if text == "":
            return [""]

        words = text.split()
        lines: List[str] = []
        current = ""

        for w in words:
            ww, _ = font.getsize(w)
            if ww > max_width:
                return None  # cannot draw without breaking word

            if current == "":
                current = w
            else:
                trial = current + " " + w
                tw, _ = font.getsize(trial)
                if tw <= max_width:
                    current = trial
                else:
                    lines.append(current)
                    current = w

        if current:
            lines.append(current)

        return lines

    # -------- compute values (your existing approach) --------
    total = timedelta(0)
    for item in formatted:
        total += item["end"] - item["start"]

    if formatted:
        empty_time = formatted[-1]["end"] - formatted[0]["start"] - total
    else:
        empty_time = timedelta(0)

    values: Mapping[str, Any] = {
        "totalTimeInEvents": format_timedelta_hm(total),
        "totalTimeNotInEvents": format_timedelta_hm(empty_time),
        "TotalEvents": len(formatted),
    }

    pattern = re.compile(r"<(totalTimeInEvents|totalTimeNotInEvents|TotalEvents)>")

    def repl(m: re.Match) -> str:
        key = m.group(1)
        v = values.get(key, None)
        return m.group(0) if v is None else str(v)

    rendered_text = pattern.sub(repl, template)

    # -------- render --------
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    if fontSize not in FONTS:
        raise KeyError(f"fontSize={fontSize} not found in FONTS keys={sorted(FONTS.keys())}")
    font = FONTS[fontSize]

    usable_w = max(1, width - 2 * padding)
    y = padding
    lh = font.getsize("00:00")[1]

    # Respect explicit newlines; wrap each logical line separately
    for logical in rendered_text.split("\n"):
        wrapped = wrap_spaces_only(draw, logical, font, usable_w)
        if wrapped is None:
            # This logical line contains a word too wide to fit; skip it entirely
            continue

        for ln in wrapped:
            if y + lh > height - padding:
                return image
            draw.text((padding, y), ln, font=font, fill=fg)
            y += lh + line_gap

    return image

template = """Today
<TotalEvents> Events\n
<totalTimeInEvents> in Events\n
<totalTimeNotInEvents> Free"""
# render_event_summary_image(template,formatted,150,225, 20)

#Countdown

def DrawCountdown(template, size, progressbarHeight, padding, startTime, targetTime, now, fg, bg):
    width, height = size

    image = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(image)

    progressbarsize  = (width - 2 * padding, progressbarHeight)

    progressbarlocation = ((width - progressbarsize[0]) // 2, height - progressbarHeight - padding)

    timeFromStart = (now - startTime).total_seconds()
    Total = (targetTime - startTime).total_seconds()

    subbed =  re.sub(r"<days>", str(int((Total-timeFromStart)/86400)), template)
    spp = Total / progressbarsize[0]
    pixelsFromStart = timeFromStart // spp

    font = fit_font_to_box(
        draw,
        subbed,
        "fonts/NotoSerifCJKsc-Regular.otf",
        box_w=width-2*padding,
        box_h=progressbarlocation[1]-10,
        max_size=80
    )

    bbox = draw.textbbox((0, 0), subbed, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    cx = width - padding - bbox[0] - tw
    cy = (progressbarlocation[1] - th) // 2 - bbox[1]

    draw.text((cx, cy), subbed, font=font, fill=(0, 0, 0))

    # Render Bar
    draw.rounded_rectangle(
        (
            progressbarlocation[0], 
            progressbarlocation[1], 
            progressbarlocation[0] + progressbarsize[0], 
            progressbarlocation[1] + progressbarsize[1]
        ), 
        progressbarsize[1]//2, bg
    )

    draw.rounded_rectangle(
        (
            progressbarlocation[0], 
            progressbarlocation[1], 
            progressbarlocation[0] + pixelsFromStart, 
            progressbarlocation[1] + progressbarsize[1]
        ), 
        progressbarsize[1]//2, fg
    )

    return image


startTime = datetime(2026,1,10)
targetTime = datetime(2026,2,13)
now = datetime.now()
fg = (0,255,255)
bg = (0,0,0)

template = "还有<days>天回家"
# DrawCountdown(template, (250, 60), 20, 5, startTime, targetTime, now, fg, bg)