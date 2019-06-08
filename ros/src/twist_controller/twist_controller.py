from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import  rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
            accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, 
        max_steer_angle):

        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1,max_lat_accel, max_steer_angle)

        kp = 0.4
        ki = 0.01
        kd = 0.2
        mn = 0.
        mx = 0.2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5
        ts = .02
        self.vel_lpf = LowPassFilter(tau, ts)


        kp_steer = 0.5
        ki_steer = 0.0
        kd_steer = 0.2
        self.steer_controller = PID(kp_steer, ki_steer, kd_steer, -max_steer_angle, max_steer_angle)


        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.max_steer_angle = max_steer_angle
        self.total_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY 

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel,cte):
        # TODO: Change the arg, kwarg list to suit your needs
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.steer_controller.reset()            
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel


        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        steering_base = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering_cte  = self.steer_controller.step(cte, sample_time)
        steering_total = steering_base + steering_cte
        steering = max(min(self.max_steer_angle, steering_total), -self.max_steer_angle)        


        throttle = self.throttle_controller.step(vel_error, sample_time)
        #throttle = ((self.max_steer_angle - abs(steering))/self.max_steer_angle)*throttle 


        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700

        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)  * self.total_mass * self.wheel_radius


        #rospy.logwarn("throttle={: .4f}".format(float(throttle)) + "  brake={0:4d}".format(int(brake ))  +"  steering={: .4f}".format(float(steering)) )
        return throttle, brake, steering