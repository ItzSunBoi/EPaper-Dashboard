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

from waveshare_epd import epd4in0e

epd = epd4in0e.EPD()   
epd.init()
epd.Clear()

epd4in0e.epdconfig.module_exit(cleanup=True)