import numpy as np

import math

def calculate_vector(point1, point2):
  """Calculates the vector from point1 to point2."""
  x = point2[0] - point1[0]
  y = point2[1] - point1[1]
  return (x, y)


def calculate_angle_degrees(vector):
  """Calculates the angle of the vector in degrees (0-360)."""
  x = vector[0]
  y = vector[1]
  angle_radians = math.atan2(y, x)
  angle_degrees = math.degrees(angle_radians)
  # Adjust to be in the range 0-360
  if angle_degrees < 0:
    angle_degrees += 360
  return angle_degrees

def get_cardinal_direction(degrees):
  """Converts an angle in degrees to a cardinal direction."""
  if 337.5 <= degrees < 360 or 0 <= degrees < 22.5:
    return "E"
  elif 22.5 <= degrees < 67.5:
    return "SE"
  elif 67.5 <= degrees < 112.5:
    return "S"
  elif 112.5 <= degrees < 157.5:
    return "SW"
  elif 157.5 <= degrees < 202.5:
    return "W"
  elif 202.5 <= degrees < 247.5:
    return "NW"
  elif 247.5 <= degrees < 292.5:
    return "N"
  elif 292.5 <= degrees < 337.5:
    return "NE"
  else:
    return "Invalid Angle"



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


class OrientationCalculator():
    """
    Calculate the orientation of an input.
    """
    def calc(world_landmarks):
        w1 = world_landmarks[0]
        w2 = world_landmarks[5]
        point1 = (w1.x, w1.y)
        point2 = (w2.x, w2.y)

        # 1. Calculate the vector
        vector = calculate_vector(point1, point2)
        #print(f"Vector between {point1} and {point2}: {vector}")

        # 2. Calculate the angle in degrees
        angle_degrees = calculate_angle_degrees(vector)
        #print(f"Angle of the vector: {angle_degrees:.2f} degrees")

        # 3. Convert the degree to a cardinal direction
        cardinal_direction = get_cardinal_direction(angle_degrees)
        #print(f"Cardinal direction: {cardinal_direction}")
        return cardinal_direction


 
if __name__ == "__main__":
  # Example usage:
  point1 = (-0.01210838, 0.06373119)
  point2 = (0.01960363, 0.00124735)

  # 1. Calculate the vector
  vector = calculate_vector(point1, point2)
  print(f"Vector between {point1} and {point2}: {vector}")

  # 2. Calculate the angle in degrees
  angle_degrees = calculate_angle_degrees(vector)
  print(f"Angle of the vector: {angle_degrees:.2f} degrees")

  # 3. Convert the degree to a cardinal direction
  cardinal_direction = get_cardinal_direction(angle_degrees)
  print(f"Cardinal direction: {cardinal_direction}")
