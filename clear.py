from waveshare_epd import epd4in0e

epd = epd4in0e.EPD()   
epd.init()
epd.Clear()

epd4in0e.epdconfig.module_exit(cleanup=True)