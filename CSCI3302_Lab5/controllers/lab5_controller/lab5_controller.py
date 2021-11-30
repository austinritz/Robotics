"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' # Part 1.1: manual mode
mode = 'planner'
#mode = 'autonomous'
def pixelsToMeters(pos):
  x = 12/360 * pos[0]
  y = 12/360 * pos[1]
  return (x,y)

def meterToPixels(m):
  pixels = 360/12 * m
  if pixels < 0:
      pixels = 0
  elif pixels > 359:
      pixels = 359
  return int(pixels)
###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (4.47, 8.054) # (Pose_X, Pose_Z) in meters
    end_w = (5.69, 5.94) # (Pose_X, Pose_Z) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = (meterToPixels(start_w[0]), meterToPixels(start_w[1])) # (x, y) in 360x360 map
    end = (meterToPixels(end_w[0]), meterToPixels(end_w[1])) # (x, y) in 360x360 map
    

      
    robotRadiusPixels = meterToPixels(AXLE_LENGTH*0.5)
    
    def map_approx(map, robotRadiusPixels):
      map2 = np.copy(map)
      
      for i in range(360):
        for j in range(360):
          if map[i][j] == 1:
            for k in range(robotRadiusPixels+1):
              for m in range(robotRadiusPixels+1):
                if i+k <= 359:
                  if j+m <= 359:
                    map2[i+k][j+m] = 1
                  if j-m <= 359:
                    map2[i+k][j-m] = 1
                if i-k <= 359:
                  if j+m <= 359:
                    map2[i-k][j+m] = 1
                  if j+m <= 359: 
                    map2[i-k][j-m] = 1          
      return map2
    


    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        '''
        # :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        # :param start: A tuple of indices representing the start cell in the map
        # :param end: A tuple of indices representing the end cell in the map
        # :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        # start = (meterToPixels(start[0]), meterToPixels(start[1]))
        # end = (meterToPixels(end[0]), meterToPixels(end[1]))
        frontier = {start : 0}
        explored = []
        previous = {start : None}

        while frontier:
          smallest = (9999,9999)
          smallestTotalCost = 9999999
          smallestGCost = 999999
          for f in frontier:
            gCost = frontier[f]
            v1 = np.array(f)
            v2 = np.array(end)

            hCost = np.linalg.norm(v1-v2)

            totalCost = gCost + hCost

            if totalCost < smallestTotalCost:
              smallest = f
              smallestTotalCost = totalCost
              smallestGCost = frontier[smallest]

          frontier.pop(smallest, None)
          explored.append(smallest)
          
          if smallest == end: #Found shortest path
            print("were here")
            ret = [start]
            prev = previous[smallest]
            while prev is not None:
              ret.append(prev)
              prev = previous[prev]
              
            ret.append(end)
            return ret
              
          x, y = smallest
          neighbors = [(x+1,y),
                       (x+1,y+1),
                       (x+1,y-1),
                       (x,y+1),
                       (x,y-1),
                       (x-1,y),
                       (x-1,y+1),
                       (x-1, y-1)]
          
          for neighbor in neighbors:
            if (neighbor[0] >= 0 and neighbor[0] < 360) and (neighbor[1] >= 0 and neighbor[1] < 360): #checking if in bounds
              print(map[neighbor[0]][neighbor[1]])
              if (map[neighbor[0]][neighbor[1]] == 0): #checking if freespace
                
                v1 = np.array(neighbor)
                v2 = np.array(end)
                hCost = np.linalg.norm(v1-v2)

                newGCost = smallestGCost + 1
                newTotalCost = newGCost + hCost

                if (neighbor not in explored) and (neighbor not in frontier):
                  frontier[neighbor] = newGCost
                  previous[neighbor] = smallest
                elif (neighbor in frontier) and (frontier[neighbor] + hCost > newTotalCost):
                  print('wtf')
                  del frontier[neighbor]
                  frontier[neighbor] = newGCost
                  previous[neighbor] = smallest

    # Part 2.1: Load map (map.npy) from disk and visualize it
    Loadmap= np.load("map.npy")
    plt.imshow(Loadmap)
    plt.show()
    # Part 2.2: Compute an approximation of the “configuration space”
    map=map_approx(Loadmap, robotRadiusPixels)#robot pixels
    plt.imshow(map)
    plt.show()
    # Part 2.3 continuation: Call path_planner
    print("value: ", map[start[0]][start[1]])
    path=path_planner(map, start, end)
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoint = []
    pathmap = np.zeros(shape=(360,360))
    for p in path:
        waypoint.append(pixelsToMeters(p))
        map[360-p[1]][p[0]] = 5
    plt.imshow(map)
    plt.show()
    np.save("path.npy", waypoint) 

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=(360,360)) # Replace None by a numpy 2D floating point array
waypoint = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoint = np.load("path.npy")
    

state = 0 # use this to iterate through your path




while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_y = gps.getValues()[2]
    pose_x = gps.getValues()[0]

    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  -(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y
    
        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 2 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            map_x = 359-int(wy*30)
            map_y = int(wx*30) -1
            if map_x > 359:
                map_x = 359
            elif map_x < 0:
                map_x = 0
            if map_y > 359:
                map_y = 359
            elif map_y < 0:
                map_y = 0
            if map[map_x][map_y] < 0.99:
                map[map_x][map_y] += 0.01
            g = map[map_x][map_y]
            if(g > 0.4):
                color = int((g*256**2+g*256+g)*255)
                display.setColor(color)
                display.drawPixel(360-int(wy*30),int(wx*30))

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-int(pose_y*30),int(pose_x*30))



    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            adjusted_map = map > 0.4
            final_map = adjusted_map *1
            np.save("map.npy", final_map) 
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        rho = math.sqrt((pose_x-waypoint[state][0])**2 + (pose_y-waypoint[state][1])**2)
        alpha = -(math.atan2(waypoint[state][1]-pose_y,waypoint[state][0]-pose_x) + pose_theta)


        #STEP 2: Controller
        xDot = 0
        thetaDot = 0
        if abs(alpha) < 0.01:
            xDot = rho
            thetaDot = alpha
            if rho < 0.05 and state < (len(waypoint)-1):
                state+=1 
                    
        else:
            thetaDot = alpha
            xDot = 0

        #STEP 3: Compute wheelspeeds
        leftSpeed = (2*xDot-thetaDot*AXLE_LENGTH)/2 #m/s
        rightSpeed = (2*xDot+thetaDot*AXLE_LENGTH)/2 #m/s
        
        vL = (leftSpeed/MAX_SPEED_MS) * MAX_SPEED *0.8
        vR = (rightSpeed/MAX_SPEED_MS) * MAX_SPEED * 0.8
        if vL > MAX_SPEED:
          vL = MAX_SPEED * 0.9
        if vR > MAX_SPEED:
          vR = MAX_SPEED * 0.9
        if vL < -MAX_SPEED:
          vL = -MAX_SPEED * 0.9
        if vR < -MAX_SPEED:
          vR = -MAX_SPEED * 0.9
        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)