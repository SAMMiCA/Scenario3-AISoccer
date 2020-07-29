import math

def go_to(dx, dy, th, maxvel):
    scale, mult_lin, mult_ang = 1.4, 3.5, 0.4
    damping = 0.35
    ka = 0
    sign = 1
    # calculate how far the target position is from the robot
    d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
    # calculate how much the direction is off
    desired_th = (math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)
    d_th = desired_th - th
    while (d_th > math.pi):
        d_th -= 2 * math.pi
    while (d_th < -math.pi):
        d_th += 2 * math.pi

    # based on how far the target position is, set a parameter that
    # decides how much importance should be put into changing directions
    # farther the target is, less need to change directions fastly
    if (d_e > 1):
        ka = 17 / 90
    elif (d_e > 0.5):
        ka = 19 / 90
    elif (d_e > 0.3):
        ka = 21 / 90
    elif (d_e > 0.2):
        ka = 23 / 90
    else:
        ka = 25 / 90

    # if the target position is at rear of the robot, drive backward instead
    if (d_th > math.radians(95)):
        d_th -= math.pi
        sign = -1
    elif (d_th < math.radians(-95)):
        d_th += math.pi
        sign = -1

    # if the direction is off by more than 85 degrees,
    # make a turn first instead of start moving toward the target
    if (abs(d_th) > math.radians(85)):
        left_wheel, right_wheel = trim_velocity(-mult_ang * d_th, mult_ang * d_th, maxvel)
    # otherwise
    else:
        # scale the angular velocity further down if the direction is off by less than 40 degrees
        if (d_e < 5 and abs(d_th) < math.radians(40)):
            ka = 0.1
        ka *= 4
        # set the wheel velocity
        # 'sign' determines the direction [forward, backward]
        # 'scale' scales the overall velocity at which the robot is driving
        # 'mult_lin' scales the linear velocity at which the robot is driving
        # larger distance 'd_e' scales the base linear velocity higher
        # 'damping' slows the linear velocity down
        # 'mult_ang' and 'ka' scales the angular velocity at which the robot is driving
        # larger angular difference 'd_th' scales the base angular velocity higher
        left_wheel, right_wheel = trim_velocity(sign * scale * (mult_lin * (
                                            1 / (1 + math.exp(-3 * d_e)) - damping) - mult_ang * ka * d_th),
                                sign * scale * (mult_lin * (
                                            1 / (1 + math.exp(-3 * d_e)) - damping) + mult_ang * ka * d_th), maxvel)
    return [left_wheel, right_wheel]

def trim_velocity(left_wheel, right_wheel, maxvel):
    multiplier = 1

    # wheel velocities need to be scaled so that none of wheels exceed the maximum velocity available
    # otherwise, the velocity above the limit will be set to the max velocity by the simulation program
    # if that happens, the velocity ratio between left and right wheels will be changed that the robot may not execute
    # turning actions correctly.
    if (abs(left_wheel) > maxvel or abs(right_wheel) > maxvel):
        if (abs(left_wheel) > abs(right_wheel)):
            multiplier = maxvel / abs(left_wheel)
        else:
            multiplier = maxvel / abs(right_wheel)

    return left_wheel*multiplier, right_wheel*multiplier
