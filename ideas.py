from geopy import distance
from pyproj import Proj

# (lat, lon)
seattle = (47.608013, -122.335167)
boise = (43.616616, -116.200886)
everett = (47.967306, -122.201399)
pendleton = (45.672075, -118.788597)
biggs = (45.669846, -120.832841)
portland = (45.520247, -122.674194)
twin_falls = (42.570446, -114.460255)
bend = (44.058173, -121.315310)
spokane = (47.657193, -117.423510)
grant_pass = (42.441561, -123.339336)
burns = (43.586126, -119.054413)
eugene = (44.050505, -123.095051)
lakeview = (42.188772, -120.345792)
missoula = (46.870105, -113.995267)

p = Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)

x1, y1 = p(-114.460255, 42.570446)  # twin_falls
x2, y2 = p(-122.335167, 47.608013)  # seattle

xCoords = [x1, x2]
yCoords = [y1, y2]

# print(np.sqrt(np.diff(xCoords) ** 2 + np.diff(yCoords) ** 2) / 1000)

city_mapping = {
    "orig-dest": ["seattle", (47.608013, -122.335167)],
    1: ["boise", (43.616616, -116.200886)],
    2: ["everett", (47.967306, -122.201399)],
    3: ["pendleton", (45.672075, -118.788597)],
    4: ["biggs", (45.669846, -120.832841)],
    5: ["portland", (45.520247, -122.674194)],
    6: ["twin_falls", (42.570446, -114.460255)],
    7: ["bend", (44.058173, -121.315310)],
    8: ["spokane", (47.657193, -117.423510)],
    9: ["grant_pass", (42.441561, -123.339336)],
    10: ["burns", (43.586126, -119.054413)],
    11: ["eugene", (44.050505, -123.095051)],
    12: ["lakeview", (42.188772, -120.345792)],
    0: ["missoula", (46.870105, -113.995267)]
}

