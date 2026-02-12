# Python Script to generate a modifiable Dashboard image.
# Copyright (C) 2026  Sun Zheng @ admin@suncoolserver.net

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from weatherRenderer import *
from calenderRenderer import *

from PIL import Image, ImageDraw, ImageFont

import json
import os

import requests_cache
import openmeteo_requests
import argparse

def read_layout(path):
    with open(path) as f:
        return json.load(f)
    
CWD = os.getcwd()

LAYOUT_DIR = "layout"
ICS_DIR = "ics"

LAYOUT_DIR = os.path.join(CWD, LAYOUT_DIR)
if "layout.json" not in list(os.walk(LAYOUT_DIR))[0][2]:
    raise ImportError(f"layout.json Not found at {LAYOUT_DIR}")

ICS_DIR = os.path.join(CWD, ICS_DIR)

layout = read_layout(os.path.join(LAYOUT_DIR, "layout.json"))

cache_session = requests_cache.CachedSession("cache/http_cache", expire_after=600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = layout["weatherParams"]

def TimeFormatter(input: dict | str) -> datetime:
    if input == "Now":
        return datetime.now()
    else:
        year = input.get("year", 0)
        month = input.get("month", 0)
        day = input.get("day", 0)
        hour = input.get("hour", 0)
        minute = input.get("minute", 0)
        seconds = input.get("seconds", 0)

        return datetime(year,month,day,hour,minute,seconds)

def ColorFormatter(input: list) -> tuple:
    return (input[0], input[1], input[2])
    
def updateData(layout):
    responses = openmeteo.weather_api(url, params=params)
    WeatherResponse = responses[0]

    CalendarSource = []
    for source in layout["CalendarSource"]:
        temp =     {
            "name" : source["name"],
            "url" : source["url"]
        }
        CalendarSource.append(temp)

    if len(CalendarSource) == 0:
        raise ImportError(f"No Calendar Data Found in layout.json")

    downloaded = download_ics(CalendarSource, output_dir=ICS_DIR)
    merged_ics_text = merge_sources_to_ics(downloaded)

    current_frame = build_current_frame(WeatherResponse)
    hourly_frames = build_hourly_frames(WeatherResponse)

    all_frames = [current_frame] + hourly_frames

    days_frames = build_daily_frames(WeatherResponse)

    return merged_ics_text, [current_frame, hourly_frames, all_frames, days_frames]

def renderWidgets(layout, eventsData, weatherData, nextScheduled, savePath = None):
    widgets = layout["widgets"]
    [current_frame, hourly_frames, all_frames, days_frames] = weatherData
    baseImage = Image.new("RGB", (layout["canvas"]["width"], layout["canvas"]["height"]), ColorFormatter(layout["global"]["background"]))

    for widget in widgets:
        widgetType = widget["type"]
        if widget["enabled"]:
            parameters = widget["parameters"]
            size = widget["size"]
            position = widget["location"]

            if widgetType == "DateTimeFrame":
                
                temp = Date_Time_Chinese(
                    width=size["width"],
                    height=size["height"],
                    time = TimeFormatter(parameters["time"])
                )

                baseImage.paste(temp,position)
                continue
                
            if widgetType == "BarCalender":

                temp = CreateBarCalender(
                    formatted = eventsData,
                    width = size["width"],
                    height = size["height"],
                    barColor = ColorFormatter(parameters["barColor"]),
                    NowLineColor = ColorFormatter(parameters["NowLineColor"]),
                    fontSize = parameters["fontSize"]
                )

                baseImage.paste(temp,position)
                continue
            
            if widgetType == "UpcomingEventsFrame":
                temp = render_upcoming_event_list(
                    formatted=eventsData,
                    width=size["width"],
                    height=size["height"],
                    fontSize=parameters["fontSize"],
                    padding=parameters["padding"],
                    line_gap=parameters["line_gap"],
                    section_gap=parameters["section_gap"],
                    fg = ColorFormatter(parameters["fg"]),
                    bg = ColorFormatter(parameters["bg"])
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "EventSummaryFrame":
                temp = render_event_summary_image(
                    template=parameters["template"],
                    formatted=eventsData,
                    width=size["width"],
                    height=size["height"],
                    fontSize=parameters["fontSize"],
                    padding=parameters["padding"],
                    line_gap=parameters["line_gap"],
                    fg = ColorFormatter(parameters["fg"]),
                    bg = ColorFormatter(parameters["bg"])
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "CountdownFrame":
                temp = DrawCountdown(
                    template=parameters["template"],
                    size=(size["width"],size["height"]),
                    progressbarHeight=parameters["progressbarHeight"],
                    padding=parameters["padding"],
                    startTime=TimeFormatter(parameters["startTime"]),
                    targetTime=TimeFormatter(parameters["targetTime"]),
                    now=TimeFormatter(parameters["now"]),
                    fg = ColorFormatter(parameters["fg"]),
                    bg = ColorFormatter(parameters["bg"])
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "HourlyRecommendation":
                temp = render_hourly_recommendation_image(
                    hourly=hourly_frames,
                    width=size["width"],
                    height=size["height"],
                    fontSize=parameters["fontSize"],
                    fun=parameters["fun"]
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "WeatherFrame":
                temp = Render_weather(
                    currentFrame=current_frame,
                    all_frames=all_frames,
                    width=size["width"],
                    height=size["height"],
                    hours=parameters["hours_ahead"],
                    fontSize=parameters["fontSize"]
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "DaySummaryFrame":
                temp = render_daylight_frame(
                    data=days_frames[0],
                    template=parameters["template"],
                    width=size["width"],
                    height=size["height"],
                    fontSize=parameters["fontSize"]
                )

                baseImage.paste(temp, position)
                continue

            if widgetType == "UpdateStatusFrame":
                temp = render_update_status_frame(
                    dateNow=datetime.now(),
                    dateNext=nextScheduled,
                    template=parameters["template"],
                    fontSize=parameters["fontSize"],
                    width=size["width"],
                    height=size["height"]
                )

                baseImage.paste(temp, position)
                continue


    if savePath:
        baseImage.save(savePath)
    else:
        return baseImage

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--next-update", default=None)
    args = p.parse_args()


    next_update = datetime.fromisoformat(args.next_update) if args.next_update else datetime.now() + timedelta(minutes=15)

    ics_text, weather_frames = updateData(layout)
    start = datetime.now().date()
    length = timedelta(1)

    eventsData = GetDateRange(ics_text, start, length)

    renderWidgets(layout, eventsData, weather_frames, next_update, "output.png")

    