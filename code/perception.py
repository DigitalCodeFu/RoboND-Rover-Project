import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


# My color band threshold function, with options for >, <, or <> RGB filters.
def color_band(img, rgb_min=(0, 0, 0), rgb_max=(255, 255, 255)):
    binary_image = np.zeros_like(img[:, :, 0])
    thresh_mask = (img[:, :, 0] >= rgb_min[0]) \
                & (img[:, :, 1] >= rgb_min[1]) \
                & (img[:, :, 2] >= rgb_min[2]) \
                & (img[:, :, 0] <= rgb_max[0]) \
                & (img[:, :, 1] <= rgb_max[1]) \
                & (img[:, :, 2] <= rgb_max[2])
    binary_image[thresh_mask] = 1
    return binary_image


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    
    # 1) Define source and destination points for perspective transform
        #img = mpimg.imread(Rover.img)
        #img = np.copy(image)
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([      [14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([ [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                               [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                               [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                               [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset] ])
    
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navi_min = (171, 153, 141) 
    obst_max = (170, 152, 140)
    rock_min, rock_max = (130, 50, 0), (198, 172, 79) 
    navi_img = color_band(warped, rgb_min=navi_min)
    obst_img = color_band(warped, rgb_max=obst_max)
    rock_img = color_band(warped, rgb_min=rock_min, rgb_max=rock_max)
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = obst_img * 255
    Rover.vision_image[:, :, 1] = rock_img * 255
    Rover.vision_image[:, :, 2] = navi_img * 255
    
    # 5) Convert map image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(navi_img)
    obst_xpix, obst_ypix = rover_coords(obst_img)
    rock_xpix, rock_ypix = rover_coords(rock_img)
    
    # 6) Convert rover-centric pixel values to world coordinates
        # Rotate rover-centric pixel values to world x,y plot
    navi_xpix_rotated, navi_ypix_rotated = rotate_pix(navi_xpix, navi_ypix, Rover.yaw)
    obst_xpix_rotated, obst_ypix_rotated = rotate_pix(obst_xpix, obst_ypix, Rover.yaw)
    rock_xpix_rotated, rock_ypix_rotated = rotate_pix(rock_xpix, rock_ypix, Rover.yaw)
        # Shift angled map coords to rover's map location, distance from 0,0
    scale = 10
    navi_xpix_translated, navi_ypix_translated = translate_pix(
        navi_xpix_rotated, navi_ypix_rotated, Rover.pos[0], Rover.pos[1], scale)
    obst_xpix_translated, obst_ypix_translated = translate_pix(
        obst_xpix_rotated, obst_ypix_rotated, Rover.pos[0], Rover.pos[1], scale)
    rock_xpix_translated, rock_ypix_translated = translate_pix(
        rock_xpix_rotated, rock_ypix_rotated, Rover.pos[0], Rover.pos[1], scale)
        # Clip pixel values that fall outside of worldmap coords
    world_size = Rover.worldmap.shape[0]
    navi_xpix_world = np.clip(np.int_(navi_xpix_translated), 0, world_size - 1)
    navi_ypix_world = np.clip(np.int_(navi_ypix_translated), 0, world_size - 1)
    obst_xpix_world = np.clip(np.int_(obst_xpix_translated), 0, world_size - 1)
    obst_ypix_world = np.clip(np.int_(obst_ypix_translated), 0, world_size - 1)
    rock_xpix_world = np.clip(np.int_(rock_xpix_translated), 0, world_size - 1)
    rock_ypix_world = np.clip(np.int_(rock_ypix_translated), 0, world_size - 1)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    if ( (Rover.roll < 0.4) or (Rover.roll > 359.6) and
         (Rover.pitch < 0.3) or (Rover.pitch > 359.7) and
         (Rover.vel >= 0) ):
        Rover.worldmap[obst_ypix_world, obst_xpix_world, 0] += 1
        Rover.worldmap[rock_ypix_world, rock_xpix_world, 1] += 1
        Rover.worldmap[navi_ypix_world, navi_xpix_world, 2] += 1
        # Zero out the rock coords in the obst red and navi blue color layers
        Rover.worldmap[rock_ypix_world, rock_xpix_world, 0] = 0
        Rover.worldmap[rock_ypix_world, rock_xpix_world, 2] = 0
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    navi_dist, navi_angles = to_polar_coords(navi_xpix, navi_ypix)
    obst_dist, obst_angles = to_polar_coords(obst_xpix, obst_ypix)
    rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)
        # Update Rover pixel distances and angles
    Rover.nav_dists = navi_dist
    Rover.nav_angles = navi_angles
    Rover.rock_dists = rock_dist
    Rover.rock_angles = rock_angles
    
    # Update navigable terrain data for navigation.
    Rover.nav_angles_avg = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
    
    # Update rock sample data for navigation.
    Rover.rock_angles_avg = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15)
    Rover.rock_dists_avg = np.mean(Rover.rock_dists)
    if len(Rover.rock_dists) > 1:
        Rover.rock_dists_min = np.nanmin(Rover.rock_dists)
    # Update if rock samples are seen close or sample_nearby.
    Rover.rock_nearby = (len(Rover.rock_dists) > 1 and 
                         Rover.rock_dists_avg < Rover.rock_min_dist)
    
    # Update hard_turn bearing to avoid collisions only if not already turning.
    if Rover.turningmode == 'off':
        if Rover.nav_angles_avg > 0.0:
            Rover.hard_turn = 15
        elif Rover.nav_angles_avg <= 0.0:
            Rover.hard_turn = -15
    
    # Update navigable terrain distance in front of Rover and detect for collisions and impacts.
    nav_dist_front = []
    for i in range(len(Rover.nav_angles)):
        if  (Rover.nav_angles[i] * 180/np.pi) > -3.0 \
        and (Rover.nav_angles[i] * 180/np.pi) < 3.0:
            nav_dist_front.append(Rover.nav_dists[i])
    if len(nav_dist_front) > 0:    # Avoid a zero division error.
        # Average distance in front of navigable terrain.
        Rover.nav_dist_front = sum(nav_dist_front) / float(len(nav_dist_front))
        # Update if a collision risk is detected.
        if Rover.nav_dist_front <= Rover.fwd_obstacle_dist and Rover.vel == Rover.max_vel:
            Rover.collision_detected = True
    
    # Update the average angle of navigable terrain that has not been mapped.
    nav_angles_uncharted = []
    for i in range(len(Rover.nav_angles)):
        if Rover.worldmap[navi_ypix_world[i], navi_xpix_world[i], 2] < 30:
            nav_angles_uncharted.append(Rover.nav_angles[i])
    if len(nav_angles_uncharted) > 0:    # Avoid a zero division error.
        # Convert angle to degrees and clip +/- 15
        nav_angles_uncharted = [i * 180/np.pi for i in nav_angles_uncharted]
        nav_angles_uncharted = [nav_angles_uncharted[i] for i in range(len(nav_angles_uncharted)) \
                                if (nav_angles_uncharted[i] >= -15) and (nav_angles_uncharted[i] <= 15)]
    if len(nav_angles_uncharted) > 0:    # Avoid a zero division error.
        # Average distance in front of navigable terrain.
        Rover.nav_angles_uncharted_avg = sum(nav_angles_uncharted) / float(len(nav_angles_uncharted))
        Rover.nav_angles_uncharted_count = len(nav_angles_uncharted)
    
    # Update if an impact is detected.
    Rover.impact = ( Rover.throttle > 0 and Rover.vel < -0.2 )
    
    # Update last position after Rover.stuck_time seconds and if stuck update Rover.stuck.
    if Rover.time_updated == None:
        Rover.pos_old = tuple(Rover.pos)
        Rover.time_updated = Rover.total_time
    elif (Rover.total_time - Rover.time_updated) > Rover.stuck_time:
        if  round(Rover.pos_old[0], 0) == round(Rover.pos[0], 0) \
        and round(Rover.pos_old[1], 0) == round(Rover.pos[1], 0) \
        and Rover.turningmode is not 'on' and not Rover.picking_up:
            if Rover.stuck == True:
                Rover.stuckagain = True
            else:
                Rover.stuck = True
            if Rover.turningmode == 'off':
                Rover.opposite_direction = (Rover.yaw + 180) % 360
                Rover.turningmode == 'on'
        else:
            Rover.stuck = False
            Rover.turningmode == 'off'
        # Update last position
        Rover.pos_old = tuple(Rover.pos)
        Rover.time_updated = Rover.total_time
    
    return Rover