#!/usr/bin/python
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool, Manager
import numpy as np
import datetime
import requests
import argparse
import tempfile
import pygrib
import gmplot
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
parser.add_argument('-S', '--sigma', help='Variance to add to random variables for monte carlo analysis', type=float, default=0.2)
args = parser.parse_args()

gribs = {}
grib_indicies = {}
interpolators = {}

def get_grib(timestamp, lock=None):
    dt = datetime.datetime.fromtimestamp(timestamp)
    if dt.date() > datetime.datetime.now().date():
        calc_day = datetime.datetime.now().strftime("%Y%m%d")
        calc_hour = (datetime.datetime.now().time().hour // 6) * 6
        model_hour = (dt - datetime.datetime.now()).seconds // 3600
    else:
        calc_day = dt.strftime("%Y%m%d")
        calc_hour = (dt.time().hour // 6) * 6
        model_hour = dt.time().hour % 6
    
    path = "gfs.{}/{}/gfs.t{}z.pgrb2.0p25.f{}".format(calc_day, str(calc_hour).zfill(2), str(calc_hour).zfill(2), str(model_hour).zfill(3))
    local_path = os.path.join('data', path)
    if lock:
        lock.acquire()
    if not os.path.isfile(os.path.join('data', path)):
        http_path = "https://www.ftp.ncep.noaa.gov/data/nccf/com/gfs/prod/" + path
        if args.verbose:
            print("Fetching grib file {}".format(http_path))
        if not os.path.isdir(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        with requests.get(http_path, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            fd, temp_path = tempfile.mkstemp()
            with os.fdopen(fd, "wb") as GRIB:
                for chunk in r.iter_content(chunk_size=8192): 
                    if chunk:
                        GRIB.write(chunk)
            os.rename(temp_path, local_path)
    if lock:
        lock.release()
    if local_path in gribs.keys():
        return gribs[local_path], grib_indicies[local_path]
    gribs[local_path] = pygrib.open(local_path)
    index = {
        "heights": {},
        "u-wind": {},
        "v-wind": {}
    }
    for i in gribs[local_path]:
        if i['name'] == "Geopotential Height":
            index['heights'][i['scaledValueOfFirstFixedSurface']] = i.values
        elif i['name'] == "U component of wind":
            index['u-wind'][i['scaledValueOfFirstFixedSurface']] = i.values
        elif i['name'] == "V component of wind":
            index['v-wind'][i['scaledValueOfFirstFixedSurface']] = i.values
    grib_indicies[local_path] = index
    return gribs[local_path], grib_indicies[local_path]

def get_weather(timestamp, latitude, longitude, altitude, lock=None):
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
    grib, grib_index = get_grib(timestamp, lock)
    heights = []
    for layer in grib_index['heights']:
        heights.append((grib_index['heights'][layer][startlatidx][startlonidx], layer))
    heights.sort(key=lambda x: x[0])
    minheight, minpressure, maxheight, maxpressure = 0, 0, 0, 0
    for height in heights:
        if height[0] < altitude:
            minheight, minpressure = height
        if height[0] > altitude:
            maxheight, maxpressure = height
            break
    if altitude > maxheight:
        print("Balloon is above valid wind data. Extrapolating to {}.".format(altitude))
        maxheight = altitude + 1000
        maxpressure = 0
    if maxheight <= minheight:
        print("Something went wrong in setting the altitude window.", heights, altitude, maxheight, minheight)
    for coord in coords:
        latidx = (startlatidx + 1) - coord[0]
        lonidx = (startlonidx + coord[1]) % 1440
        altidx = 0
        if coord[2]:
            height, pressure = maxheight, maxpressure
        else:
            height, pressure = minheight, minpressure
        windv, windu = 0, 0

        for layer in grib_index['u-wind']:
            if layer == pressure:
                windu = grib_index['u-wind'][layer][latidx][lonidx]
                break
        for layer in grib_index['v-wind']:
            if layer == pressure:
                windv = grib_index['v-wind'][layer][latidx][lonidx]
                break
        points.append(np.array([90 - (latidx * 0.25), lonidx * 0.25, height]))
        data[coord[0]][coord[1]][coord[2]] = np.array([windu, windv, pressure])
    cache_key = (startlatidx, startlonidx, points[0][2], points[7][2], (timestamp // 3600) * 3600, (timestamp // 3600 + 1) * 3600)
    try:
        interpolators[cache_key] = RegularGridInterpolator(((points[0][0], points[7][0]), (points[0][1], points[7][1]), (points[0][2], points[7][2])), data)
    except ValueError:
        print(points, altidx, heights)
    return interpolators[cache_key](np.array([latitude, longitude, altitude]))[0]

def random_step():
    return np.random.normal(args.step, args.sigma)

def simulate(sim_time, latitude, longitude, altitude, burst, weather):
    u, v, pressure = weather
    dlat = (v * random_step()) / EARTH_RADIUS
    dlon = (u * random_step()) / (EARTH_RADIUS * math.cos(math.pi * latitude / 180))
    latitude += dlat * 180 / math.pi
    longitude += dlon * 180 / math.pi
    if altitude > 30510:
        burst = True
    if burst:
        altitude += -17 * random_step()
    else:
        altitude += 4 * random_step()
    sim_time += args.step
    return sim_time, latitude, longitude, altitude, burst

def run_simulation(lock):
    sim_time = args.time
    lat = args.latitude
    lon = args.longitude
    alt = args.altitude
    burst = args.burst
    for i in range(10000):
        weather = get_weather(sim_time, lat, lon, alt, lock)
        sim_time, lat, lon, alt, burst = simulate(sim_time, lat, lon, alt, burst, weather)
        if alt < args.altitude:
            break

def main():
    latitudes = []
    longitudes = []
    pool = Pool(4)
    m = Manager()
    lock = m.Lock()
    pool.map(run_simulation, [lock for x in range(100)])
    gmap = gmplot.GoogleMapPlotter(args.latitude, args.longitude, 12)
    gmap.heatmap(latitudes, longitudes)
    gmap.apikey = ""
    gmap.draw("heatmap.html")
    print(len(interpolators))

main()