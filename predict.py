#!/usr/bin/python
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import datetime
import requests
import argparse
import pygrib
import time
import math
import os

EARTH_RADIUS = 6378137 # in meters

parser = argparse.ArgumentParser(description='Predict the motion of a weather balloon based on GFS model data.')
parser.add_argument('-a', '--altitude', help='The altitude of the balloon at the start of the simulation in meters', type=float, default=100)
parser.add_argument('-l', '--latitude', help='The latitude of the balloon at the start of the simulation in decimal degrees', type=float, default=38.9928954)
parser.add_argument('-L', '--longitude', help='The longitude of the balloon at the start of the simulation in decimal degrees', type=float, default=-76.939704)
parser.add_argument('-v', '--verbose', help='Show more output while running the simulation', action='store_true')
parser.add_argument('-d', '--burst-diameter', help='The estimated diameter of the balloon when it bursts in meters', type=float, default=5)
parser.add_argument('-g', '--gas-moles', help='The amount of gas inside the balloon in moles', type=float, default=1500)
parser.add_argument('-M', '--mass', help='The total mass of the balloon including gas and payload in grams', type=float, default=15000)
parser.add_argument('-t', '--time', help='The unix timestamp at the start of the simulation', type=float, default=time.time())
parser.add_argument('-b', '--burst', help='This flag indicates that the balloon has burst prior to the start of the simulation.', action='store_true')
parser.add_argument('-s', '--step', help='Time in seconds between simulation steps', type=float, default=1)
args = parser.parse_args()

gribs = {}
interpolators = {}

def get_grib(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    if dt.date() > datetime.datetime.now().date():
        calc_day = datetime.datetime.now().strftime("%Y%m%d")
        calc_hour = (dt.time().hour // 6) * 6
        model_hour = (dt - datetime.datetime.now()).seconds // 3600
    else:
        calc_day = dt.strftime("%Y%m%d")
        calc_hour = (dt.time().hour // 6) * 6
        model_hour = dt.time().hour % 6
    
    path = "gfs.{}/{}/gfs.t{}z.pgrb2.0p25.f{}".format(calc_day, str(calc_hour).zfill(2), str(calc_hour).zfill(2), str(model_hour).zfill(3))
    local_path = os.path.join('data', path)
    if not os.path.isfile(os.path.join('data', path)):
        http_path = "https://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/" + path
        if args.verbose:
            print("Fetching grib file {}".format(http_path))
        if not os.path.isdir(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        with requests.get(http_path, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as GRIB:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk:
                        GRIB.write(chunk)
    if local_path in gribs.keys():
        return gribs[local_path]
    gribs[local_path] = pygrib.index(local_path, 'name')
    return gribs[local_path]

def get_weather(timestamp, latitude, longitude, altitude):
    longitude %= 360
    startlatidx = int((90 - latitude) // 0.25)
    startlonidx = int(longitude // 0.25)
    for i in interpolators:
        if i[0] == startlatidx and i[1] == startlonidx:
            if i[2] <= altitude <= i[3]:
                if i[4] <= timestamp <= i[5]:
                    return interpolators[i](np.array([latitude, longitude, altitude]))[0]
    coords = [
        (0,0,0),
        (0,0,1),
        (0,1,0),
        (0,1,1),
        (1,0,0),
        (1,0,1),
        (1,1,0),
        (1,1,1)
    ]
    points = []
    data = np.ndarray(shape=(2,2,2,3), dtype=float)
    grib = get_grib(timestamp)
    layers = grib.select(name='Geopotential Height')
    heights = []
    for layer in layers:
        heights.append((layer.values[startlatidx][startlonidx], layer['scaledValueOfFirstFixedSurface']))
    heights.sort(key=lambda x: x[0])
    for coord in coords:
        latidx = (startlatidx + 1) - coord[0]
        lonidx = (startlonidx + coord[1]) % 1440
        altidx = 0
        for i in heights:
            if i[0] > altitude:
                break
            altidx += 1
        if not coord[2]:
            if altidx > 0:
                altidx -= 1
        height, pressure = heights[altidx]
        for layer in grib.select(name='U component of wind'):
            if layer['scaledValueOfFirstFixedSurface'] == pressure:
                windu = layer.values[latidx][lonidx]
        for layer in grib.select(name='V component of wind'):
            if layer['scaledValueOfFirstFixedSurface'] == pressure:
                windv = layer.values[latidx][lonidx]
        points.append(np.array([90 - (latidx * 0.25), lonidx * 0.25, height]))
        data[coord[0]][coord[1]][coord[2]] = np.array([windu, windv, pressure])
    cache_key = (startlatidx, startlonidx, points[0][2], points[7][2], (timestamp // 3600) * 3600, (timestamp // 3600 + 1) * 3600)
    interpolators[cache_key] = RegularGridInterpolator(((points[0][0], points[7][0]), (points[0][1], points[7][1]), (points[0][2], points[7][2])), data)
    return interpolators[cache_key](np.array([latitude, longitude, altitude]))[0]

def simulate(sim_time, latitude, longitude, altitude, weather):
    time_step = args.step
    dlat = (weather[0] * time_step) / EARTH_RADIUS
    dlon = (weather[1] * time_step) / (EARTH_RADIUS*math.cos(math.pi*latitude/180))
    latitude += dlat * 180 / math.pi
    longitude += dlon * 180 / math.pi
    altitude += 5 * time_step
    sim_time += time_step
    return sim_time, latitude, longitude, altitude

lat = float(args.latitude)
lon = float(args.longitude)
alt = float(args.altitude)
sim_time = float(args.time)

for i in range(100):
    weather = get_weather(sim_time, lat, lon, alt)
    sim_time, lat, lon, alt = simulate(sim_time, lat, lon, alt, weather)
    print(lat, lon)
