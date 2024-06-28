class Weather:
    def __init__(self, description=None, temperature=None, humidity=None, wind_speed=None, sunset=None, sunrise=None):
        self.description = description
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.sunset = sunset
        self.sunrise = sunrise


    def update(self, description=None, temperature=None, wind_speed=None, humidity=None, sunset=None, sunrise=None):
        if description is not None:
            self.description = description
        if temperature is not None:
            self.temperature = temperature
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if humidity is not None:
            self.humidity = humidity
        if sunset is not None:
            self.sunset = sunset
        if sunrise is not None:
            self.sunrise = sunrise