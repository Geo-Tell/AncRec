#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import open3d as o3d

class Spherical(object):
  '''球坐标系'''
  def __init__(self, radial = 1.0, polar = 0.0, azimuthal = 0.0):
    self.radial = radial
    self.polar = polar
    self.azimuthal = azimuthal
  def toCartesian(self):
    '''转直角坐标系'''
    r = math.sin(self.azimuthal) * self.radial
    x = math.cos(self.polar) * r
    y = math.sin(self.polar) * r
    z = math.cos(self.azimuthal) * self.radial
    return x, y, z
def splot(limit):
  s = Spherical()
  points = []
  n = int(math.ceil(math.sqrt((limit - 2) / 4))) #((102-2)/4)= 5^2
  azimuthal = 0.5 * math.pi / n
  for a in range(-n, n + 1):
    s.polar = 0
    size = (n - abs(a)) * 4 or 1
    polar = 2 * math.pi / size
    for i in range(size):
      points.append(s.toCartesian())
      s.polar += polar
    s.azimuthal += azimuthal
  return np.array(points).T

v = splot(902) #225*4+2
print(v.shape)

np.save('sphere%d' % v.shape[1], v)
p = o3d.utility.Vector3dVector(v)
pc = o3d.geometry.PointCloud(p)
o3d.visualization.draw_geometries([pc])


# for point in splot(100):
#   print("%f %f %f" % point)