import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # Implement conditionals to decide what to do given perception data
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        
        # - Forward Mode -
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward: # 125                
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                Rover.brake = 0
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0                       
                # - Steering -
                # Check for nearby rocks, steer toward it.
                if len(Rover.rock_angles) > 1:
                    if Rover.rock_nearby: # Head directly toward it
                        Rover.steer = Rover.rock_angles_avg 
                    else: # Head between the rock and average navigable terrain directions.
                        Rover.steer = (Rover.rock_angles_avg + Rover.nav_angles_avg) / 2.0
                # Else If there is an incoming obstacle, steer away harder to avoid.
                elif Rover.collision_detected:
                    Rover.steer = Rover.hard_turn
                # Otherwise head toward navigable terrain pixels searching for uncharted territory.
                else:
                    if Rover.nav_angles_uncharted_count > 30:
                        Rover.steer = (Rover.nav_angles_avg + Rover.nav_angles_uncharted_avg*2) / 3.0 * Rover.steer_dampener
                    else:    # Head toward average navigable terrain pixels (clipped for +/- 15 degrees).
                        Rover.steer = Rover.nav_angles_avg * Rover.steer_dampener
                # If sufficient nav_angles, stop the Rover if it hits an obstacle or can pickup rock sample.
                if Rover.rock_dists_min > 0 and Rover.rock_dists_min <= Rover.rock_pickup_range:
                    Rover.throttle = 0
                    Rover.brake = 0
                    Rover.steer = 0
                    Rover.mode = 'stop'
                if Rover.impact:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
                    Rover.stuck = True

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward: # 125
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                Rover.brake = Rover.brake_set # Set brake to stored brake value
                Rover.steer = 0
                Rover.mode = 'stop'
        
        # - Stop Mode -
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.1:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            
            # If we're not moving (vel < 0.1) then do something else
            elif Rover.vel <= 0.1:                    
                Rover.brake = 0 # Release the brake to allow turning
                Rover.steer = 0
                Rover.throttle = 0
                # Only execute the turning and forward mode if Rover is not picking up a rock.
                if not Rover.send_pickup and not Rover.picking_up:
                    # First if the Rover stopped for a visible rock sample, turn to it
                    if Rover.rock_nearby:
                        Rover.steer = Rover.rock_angles_avg
                        # Get closer if the rock sample is not in pickup range.
                        if Rover.rock_dists_min > Rover.rock_pickup_range:
                            Rover.throttle = Rover.throttle_set
                            Rover.mode = 'forward'
                            Rover.turningmode = 'off'
                        else:    # Brake for pickup.
                            Rover.brake = Rover.brake_set
                    # Now we're stopped and we have vision data to see if there's a path forward
                    elif len(Rover.nav_angles) < Rover.go_forward: # 900
                        Rover.throttle = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        # Turn to the average navigable terrain pixels, but if already turning don't update bearing.
                        if Rover.steer != -15 and Rover.steer != 15: # if Rover.turningmode is not 'on':
                            Rover.steer = Rover.hard_turn
                        Rover.turningmode = 'on'
                    # If we're stopped but see sufficient navigable terrain in front then go!
                    elif len(Rover.nav_angles) >= Rover.go_forward: # 900
                        Rover.throttle = Rover.throttle_set # Set throttle back to stored value
                        # Set steer to mean angle
                        Rover.steer = Rover.nav_angles_avg
                        Rover.mode = 'forward'
                        Rover.turningmode = 'off'
    
        # If the Rover has not moved and is not in turning mode, turn around 180 degrees.
            #and Rover.turningmode == 'off'
        if Rover.stuck and not Rover.picking_up and not Rover.send_pickup:
            # If in the same location, then Rover is stuck and needs to get out.
            Rover.throttle = 0.0
            Rover.brake = 0 # Release the brake to allow turning
            # Go in reverse if turning during the last stuck flag did not work.
            if Rover.stuckagain:
                Rover.throttle = -2.0
            Rover.steer = 15
            if round(Rover.yaw, -1) == round(Rover.opposite_direction, -1):
                Rover.stuck = False
    
    # Just to make the rover do something 
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.stuck = False
    
    # If Rover is stuck, then stop going forward and go to stop mode.
    #if Rover.stuck and not Rover.picking_up and not Rover.send_pickup:
    #    Rover.mode = 'stop'
    
    return Rover

