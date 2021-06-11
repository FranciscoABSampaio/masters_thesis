# LIBRARY AND IMPORT STATEMENTS

# Essentials
import numpy as np
import pandas as pd
import math
import time
import sys  # For handling errors
import signal as sig_signal  # For handling events like KeyboardInterrupt
import ast  # For converting string representations of lists into lists

# For EMG pre-processing
from scipy import signal
# Feature extraction
# from itertools import repeat  # To automatically create a list of N lists
from pyentrp import entropy as ent  # Sample Entropy
from statsmodels.tsa.ar_model import AutoReg  # Autoregressive Model

# For Train-Test-Validation splits
import random

# Random Forest
import sklearn.ensemble as ml  # For Random Forest algorithm
from sklearn import metrics  # For evaluating the model's performance

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate  # For k-fold cross-validation. cross_validate differs from
# cross_val_score in that it also returns fit_time, score_time, and other optional data

# Sensors
from threading import Thread  # For continuous sensor data acquisition
from multiprocessing import Process, Value, Manager, Queue, Array, \
    Lock  # For splitting the control workload into different processes
# Value allows certain variables and flags to be shared between processes
# Manager allows a list to be shared between processes

import board  # For I2C communication
import busio  # For I2C communication

import qwiic_icm20948  # IMU's library
from ahrs.filters import Madgwick  # Import Madgwick algorithm for attitude determination from MARG data
from scipy.spatial.transform import Rotation  # For conversion from quaternions to Euler angles

import adafruit_ads1x15.ads1115 as ADS  # ADC library
from adafruit_ads1x15.analog_in import AnalogIn  # Provides behaviour similar to the core AnalogIn library,
# but is specific to the ADS1x15 ADCs

# DC Motor Control

import Encoder
import RPi.GPIO as GPIO


# ==================================================  FUNCTIONS  =======================================================


# Callback function for SIGINT event, to cleanly and safely quit the program
def signal_handler(arg_signal, arg_frame):
    global flag_KeyboardInterrupt
    flag_KeyboardInterrupt.value = True


# Change the Encoder reference state
def encoder_ref_state(channel):
    global enc_R, device_theta, enc_correction
    enc_correction = 31 - device_theta


# Use the dot product as a measure of how dissimilar two identity quaternions are
def quat_error(arg_q1, arg_q2):
    _dot = arg_q1[0] * arg_q2[0] + arg_q1[1] * arg_q2[1] + arg_q1[2] * arg_q2[2] + arg_q1[3] * arg_q2[3]

    # When the quaternions are very similar, their dot product yields approximately 1 and the difference between 1 and
    # that result should be close to 0 -> error is close to 0. If they are very different, their dot product is close to
    # 0 and thus the error is close to 1.
    return np.abs(1 - np.abs(_dot))


# Invert quaternion
def quat_inversion(arg_quat):
    return arg_quat[0], -arg_quat[1], -arg_quat[2], -arg_quat[3]


# Multiply two quaternions
def quat_mult(arg_q1, arg_q2):
    _w1, _x1, _y1, _z1 = arg_q1
    _w2, _x2, _y2, _z2 = arg_q2

    _w = _w1 * _w2 - _x1 * _x2 - _y1 * _y2 - _z1 * _z2
    _x = _w1 * _x2 + _x1 * _w2 + _y1 * _z2 - _z1 * _y2
    _y = _w1 * _y2 + _y1 * _w2 + _z1 * _x2 - _x1 * _z2
    _z = _w1 * _z2 + _z1 * _w2 + _x1 * _y2 - _y1 * _x2

    return np.array([_w, _x, _y, _z]).T


# Rotate a vector through quaternion product
def rotate_vector(arg_quat, arg_vector, arg_inv_quat):
    _aux_quat = np.array([0., arg_vector[0], arg_vector[1], arg_vector[2]])

    return quat_mult(quat_mult(arg_quat, _aux_quat), arg_inv_quat)[1:]


# Compute the unit-vector direction of a 3D vector
def normalize(arg_vector):
    return arg_vector / np.linalg.norm(arg_vector)


# Compute the relative angular velocity between two bodies based on their angular velocity relative to a third,
# common body
def compute_wn(arg_w1, arg_w2, arg_quat1, arg_quat2, arg_inv_quat1, arg_inv_quat2):
    _w_rel = rotate_vector(arg_quat1, arg_w1, arg_inv_quat1) - rotate_vector(arg_quat2, arg_w2, arg_inv_quat2)

    return normalize(_w_rel)


# Compute the joint axis i
def j_i(arg_theta, arg_rho):
    if abs(math.sin(arg_theta)) >= 0.5:  # To avoid the singularity at dj_i/dphi_i = 0
        _vector = np.array(
            [math.sin(arg_theta) * math.cos(arg_rho), math.sin(arg_theta) * math.sin(arg_rho), math.cos(arg_theta)]).T
    else:
        _vector = np.array(
            [math.cos(arg_theta), math.sin(arg_theta) * math.sin(arg_rho), math.sin(arg_theta) * math.cos(arg_rho)]).T

    return _vector


# Compute the derivative of the joint axis i relative to theta i
def dj_dtheta_i(arg_theta, arg_rho):
    if abs(math.sin(arg_theta)) >= 0.5:  # To avoid the singularity at dj_i/dphi_i = 0
        _vector = np.array(
            [math.cos(arg_theta) * math.cos(arg_rho), math.cos(arg_theta) * math.sin(arg_rho), -math.sin(arg_theta)]).T
    else:
        _vector = np.array(
            [-math.sin(arg_theta), math.cos(arg_theta) * math.sin(arg_rho), math.cos(arg_theta) * math.cos(arg_rho)]).T

    return _vector


# Compute the derivative of the joint axis i relative to rho i
def dj_drho_i(arg_theta, arg_rho):
    if abs(math.sin(arg_theta)) >= 0.5:  # To avoid the singularity at dj_i/dphi_i = 0
        _vector = np.array(
            [-math.sin(arg_theta) * math.sin(arg_rho), math.sin(arg_theta) * math.cos(arg_rho), 0]).T
    else:
        _vector = np.array(
            [0, math.sin(arg_theta) * math.cos(arg_rho), -math.sin(arg_theta) * math.sin(arg_rho)]).T

    return _vector


# Compute the partial derivative of j_n,k relative to one of the angles of vector phi
def bracket_dji(arg_phi, arg_i, arg_quat, arg_inv_quat):
    # Compute the respective partial derivative, depending on which component of vector phi it is
    if arg_i % 2 == 0:  # If its index is an even number, it must be theta_i
        _dji = dj_dtheta_i(arg_phi[arg_i], arg_phi[arg_i + 1])
    else:  # If its index is an odd number, it must be rho_i
        _dji = dj_drho_i(arg_phi[arg_i - 1], arg_phi[arg_i])

    return rotate_vector(arg_quat, _dji, arg_inv_quat)


# Compute the derivative of the error for iteration k
def E_k(arg_wn, arg_quat1, arg_quat2, arg_inv_quat1, arg_inv_quat2, arg_phi, arg_i):
    # Compute the rotated joint axes which will be used for both _error_n_k and _derror_n_k_dphi_i
    _bracket_j1 = rotate_vector(arg_quat1, j_i(arg_phi[0], arg_phi[1]), arg_inv_quat1)
    _bracket_j2 = rotate_vector(arg_quat2, j_i(arg_phi[2], arg_phi[3]), arg_inv_quat2)

    # The first term is the error_n_k and is given by the inner product between the normalized relative angular velocity
    # and the cross product of the joint axes
    _error_n_k = arg_wn @ np.cross(_bracket_j1, _bracket_j2)
    # The second term is the derivative of error_n_k relative to phi_i and is given by the inner product between the
    # normalized relative angular velocity and the cross product of a joint axis and the partial derivative relative to
    # phi_i of the other joint axis
    if arg_i < 2:
        _derror_n_k_dphi_i = arg_wn @ np.cross(bracket_dji(arg_phi, arg_i, arg_quat1, arg_inv_quat1), _bracket_j2)
    else:
        _derror_n_k_dphi_i = arg_wn @ np.cross(_bracket_j1, bracket_dji(arg_phi, arg_i, arg_quat2, arg_inv_quat2))

    return _error_n_k * _derror_n_k_dphi_i


# Compute the derivative of the Jacobian relative to phi_i
def dJacobian_dphi_i(arg_n, arg_wn_list, arg_quat1_list, arg_quat2_list, arg_inv_quat1_list, arg_inv_quat2_list,
                     arg_phi, arg_i):
    _sum = 0.

    # Sum all errors for all samples
    for k in range(arg_n):
        _sum += E_k(arg_wn_list[k], arg_quat1_list[k], arg_quat2_list[k], arg_inv_quat1_list[k], arg_inv_quat2_list[k],
                    arg_phi, arg_i)

    # Divide by the number of samples
    return (2. / arg_n) * _sum


# Take a step in the sensor-to-segment orientation estimation algorithm
def next_step_phi(arg_phi, arg_step_size, arg_wn_list, arg_quat1_list, arg_quat2_list, arg_inv_quat1_list,
                  arg_inv_quat2_list):
    # Keep track of the largest cost
    _largest_cost = 0

    # Update each angle phi_i in vector phi
    for i in range(4):
        # Save the current cost (partial derivative of the Jacobian)
        _current_cost = dJacobian_dphi_i(len(arg_wn_list), arg_wn_list, arg_quat1_list,
                                         arg_quat2_list, arg_inv_quat1_list,
                                         arg_inv_quat2_list, arg_phi, i)
        # Update the angle
        arg_phi[i] = arg_phi[i] - arg_step_size.value * _current_cost
        # Save only the largest cost
        _largest_cost = max(abs(_largest_cost), abs(_current_cost))

    # Shorten the step size if near an optimum
    if _largest_cost < arg_step_size.value / 2:
        arg_step_size.value /= 2

    return arg_phi[:]


# Compute the Euler angles from sensor orientations and sensor-to-segment orientations
def joint_eulers(arg_quat1, arg_quat2, arg_sts1, arg_sts2):
    _first_term = quat_mult(arg_quat1, arg_sts1)
    _second_term = quat_mult(arg_quat2, arg_sts2)

    _body_to_body = quat_mult(quat_inversion(_first_term), _second_term)

    # The Rotation class takes in quaternions in scalar-last (x, y, z, w) format
    _rotation = Rotation.from_quat([_body_to_body[1], _body_to_body[2], _body_to_body[3], _body_to_body[0]])

    return _rotation.as_euler('zxy', degrees=True)


# Compute the sensor to segment orientation quaternions from the vector phi of spherical coordinate angles
def sensors_to_segments(arg_phi, arg_alfa, arg_gamma):
    # Compute the DoF axes
    _j1 = j_i(arg_phi[0], arg_phi[1])
    _j2 = j_i(arg_phi[2], arg_phi[3])

    # For the arm sensor, compute the sensor to segment orientation quaternion
    _cross = np.cross([0., 0., 1.], _j1)
    _sts1 = np.array([math.acos(np.dot([0., 0., 1.], _j1)), _cross[0], _cross[1], _cross[2]])

    # For the forearm sensor, compute the sensor to segment orientation quaternion
    _cross = np.cross([0., 1., 0.], _j2)
    _sts2 = np.array([math.acos(np.dot([0., 1., 0.], _j2)), _cross[0], _cross[1], _cross[2]])

    # Adjust sensor-to-segment angles based on starting values
    _sts1 = quat_mult(_sts1, np.array([arg_alfa, 0., 0., 1.]))
    _sts1 = quat_mult(_sts2, np.array([arg_gamma, 0., 1., 0.]))

    return _sts1, _sts2


# Function for the concurrent thread of orientation data acquisition and Madgwick attitude estimation
def motion_tracking_function(arg_imu1, arg_imu2, arg_calibration_step, arg_flag_calibration_complete, arg_list_quat1,
                             arg_list_quat2, arg_wn, arg_window_increment):
    # (0/3)CALIBRATE INERTIAL MAGNETIC SENSORS
    # Initiate all gyroscope offsets at zero
    gyr_1_offset = np.array([0, 0, 0])
    gyr_2_offset = np.array([0, 0, 0])

    # Initiate the range of the magnetometer at the opposite limits of its range
    mag_1_min = np.tile(4.900, 3)
    mag_1_max = np.tile(-4.900, 3)
    mag_2_min = np.tile(4.900, 3)
    mag_2_max = np.tile(-4.900, 3)

    calibration_end = time.time() + 20  # Acquire IMU data for a total of 20 seconds
    number_of_calibration_iters = 0
    while time.time() < calibration_end:
        # Get the IMU1 readings
        arg_imu1.getAgmt()  # If the sensor was able to read and update the readings
        acc_1 = np.array([arg_imu1.axRaw, arg_imu1.ayRaw, arg_imu1.azRaw])  # Get the acceleration readings on all axes
        gyr_1 = np.array([arg_imu1.gxRaw, arg_imu1.gyRaw, arg_imu1.gzRaw])  # Get the gyroscope readings on all axes
        mag_1 = np.array([arg_imu1.mxRaw, arg_imu1.myRaw, arg_imu1.mzRaw])  # Get the magnetometer readings on all axes

        # Get the IMU2 readings
        arg_imu2.getAgmt()  # If the sensor was able to read and update the readings
        acc_2 = np.array([arg_imu2.axRaw, arg_imu2.ayRaw, arg_imu2.azRaw])  # Get the acceleration readings on all axes
        gyr_2 = np.array([arg_imu2.gxRaw, arg_imu2.gyRaw, arg_imu2.gzRaw])  # Get the gyroscope readings on all axes
        mag_2 = np.array([arg_imu2.mxRaw, arg_imu2.myRaw, arg_imu2.mzRaw])  # Get the magnetometer readings on all axes

        # Sum all offsets
        number_of_calibration_iters += 1
        gyr_1_offset += gyr_1
        gyr_2_offset += gyr_2

        # Compute the ranges of the measurements for the magnetometers
        for i in range(3):
            mag_1_max[i] = max(mag_1_max[i], mag_1[i])
            mag_1_min[i] = max(mag_1_min[i], mag_1[i])
            mag_2_max[i] = max(mag_2_max[i], mag_2[i])
            mag_2_min[i] = max(mag_2_min[i], mag_2[i])

    # Compute the average of the gyroscope offsets
    gyr_1_offset = gyr_1_offset / number_of_calibration_iters
    gyr_2_offset = gyr_2_offset / number_of_calibration_iters
    # gyr_y1_offset = max(gyr_y1_offset, gyr_1[1], key=abs)

    # Compute the spatial offset of the magnetometer measurements
    magbias_1 = tuple(map(lambda _a, _b: (_a + _b) / 2, mag_1_min, mag_1_max))
    magbias_2 = tuple(map(lambda _a, _b: (_a + _b) / 2, mag_2_min, mag_2_max))

    # Create variables for changing the IMU's units
    twos_to_acceleration = 2. / 2 ** (
            16 - 1) * 9.80665  # Convert from 16-bit two's complement to meters per second squared
    twos_to_ang_velocity = 250. * (math.pi / 180) / 2 ** (
            16 - 1)  # Convert from 16-bit two's complement to radians per second
    twos_to_magnetic = 4.900 / 2 ** (16 - 1)  # Convert from 16-bit two's complement to milliteslas

    # Create AHRS for each IMU, using the Madgwick algorithm
    orientation_1 = Madgwick()
    orientation_2 = Madgwick()
    madgwick_prior_time = time.time()  # Save the last iteration time

    # List of quaternions, where the first has to be initiated as a unit/identity quaternion
    quat1 = np.array([1., 0., 0., 0.])
    quat2 = np.array([1., 0., 0., 0.])

    # CONVERGENCE OF MADGWICK FILTER
    _error = 1.
    arg_convergence_error_tolerance = 0.005
    c_timer = 0
    c_counter = 0
    sampling_time = 0
    # Create temporary variables
    _old_quat1 = quat1
    _old_quat2 = quat2

    while True:
        # Get the IMU1 readings and convert them to the correct units
        arg_imu1.getAgmt()  # If the sensor was able to read and update the readings
        acc_1 = twos_to_acceleration * np.array(
            [arg_imu1.axRaw, arg_imu1.ayRaw, arg_imu1.azRaw])  # Get the acceleration readings on all axes
        gyr_1 = twos_to_ang_velocity * np.array([arg_imu1.gxRaw - gyr_1_offset[0], arg_imu1.gyRaw - gyr_1_offset[1],
                                                 arg_imu1.gzRaw - gyr_1_offset[
                                                     2]])  # Get the gyroscope readings on all axes
        mag_1 = twos_to_magnetic * np.array([arg_imu1.mxRaw - magbias_1[0], arg_imu1.myRaw - magbias_1[1],
                                             arg_imu1.mzRaw - magbias_1[
                                                 2]])  # Get the magnetometer readings on all axes

        # Get the IMU2 readings and convert them to the correct units
        arg_imu2.getAgmt()  # If the sensor was able to read and update the readings
        acc_2 = twos_to_acceleration * np.array(
            [arg_imu2.axRaw, arg_imu2.ayRaw, arg_imu2.azRaw])  # Get the acceleration readings on all axes
        gyr_2 = twos_to_ang_velocity * np.array([arg_imu2.gxRaw - gyr_2_offset[0], arg_imu2.gyRaw - gyr_2_offset[1],
                                                 arg_imu2.gzRaw - gyr_2_offset[
                                                     2]])  # Get the gyroscope readings on all axes
        mag_2 = twos_to_magnetic * np.array([arg_imu2.mxRaw - magbias_2[0], arg_imu2.myRaw - magbias_2[1],
                                             arg_imu2.mzRaw - magbias_2[
                                                 2]])  # Get the magnetometer readings on all axes

        # Take into account the sampling rate
        step_time = time.time() - madgwick_prior_time
        orientation_1.Dt = step_time
        orientation_2.Dt = step_time
        madgwick_prior_time = time.time()

        # Update the Madgwick AHRS
        quat1 = orientation_1.updateMARG(quat1, acc=acc_1, gyr=gyr_1, mag=mag_1)
        quat2 = orientation_2.updateMARG(quat2, acc=acc_2, gyr=gyr_2, mag=mag_2)

        # (1/3)CONVERGE ON INITIAL POSITION FOR THE MADGWICK FILTER
        if _error > arg_convergence_error_tolerance:  # The error doesn't have to be scaled by the quaternion's magnitude
            # because they are unit quaternions
            # Wait at least a second before computing the error - this helps ensure that the evaluation of the error is based on steady state performance
            if c_timer < 1:
                # Increment time passed since convergence loop started
                c_timer += step_time
            else:
                # Compute the iteration's error as a sum of the errors for all IMUs
                _error = quat_error(_old_quat1, quat1) + quat_error(_old_quat2, quat2)
                # Store the latest quaternions for measuring error
                _old_quat1 = quat1
                _old_quat2 = quat2
                # Wait another second before computing the error again
                c_timer = 0
        else:
            if arg_calibration_step.value == 0:
                # Flag the main process that IMU calibration is complete and the Madgwick algorithm has converged
                arg_flag_calibration_complete.value = True
                print('Madgwick filter converged successfully with an error of {:04.03e}.'.format(_error))
                # Update calibration step to signal the parent process
                arg_calibration_step.value += 1
                # Reset timer for convergence
                c_timer = 0

        # (2/3)CONVERGE ON SENSOR-TO-SEGMENT ORIENTATIONS THROUGH GRADIENT DESCENT
        # Wait for the user to ready up for the joint angle algorithm calibration
        if arg_calibration_step.value == 1 and not arg_flag_calibration_complete.value:
            # Start counting elapsed time since convergence started
            if c_timer < 45:
                # Outside of the loop so as to not interfere with logic
                c_timer += step_time
                # Append to list of quaternions
                arg_list_quat1.append(quat1)
                arg_list_quat2.append(quat2)
                # Compute relative angular velocity
                arg_wn.append(compute_wn(gyr_1, gyr_2, quat1, quat2, quat_inversion(quat1), quat_inversion(quat2)))
            elif c_timer > 45:  # Once 45 seconds of data has been acquired, pass orientation data to parent process
                # Signal the parent process that the data is ready
                arg_calibration_step.value += 1
                # Compute the average milliseconds it took to obtain a single sample.
                sampling_time = float(math.ceil(1000 * 45 / len(arg_wn)))
                # Without the quaternion inversion and the relative angular velocity computations, the imu_sampling_time should be
                # faster. However, this difference provides some room for error and abnormally long sampling times
                print('Quaternion orientations computed at a mean rate of {:04.02f} [Hz].'.format(
                    1000. / sampling_time))
                # Reset timer
                c_timer = 0

        # (3/3) BUILD THE FIRST WINDOW IN A KNOWN INITIAL POSITION
        # REGULARIZE DATA ACQUISITION
        if arg_calibration_step.value == 3:
            # Append latest data
            arg_list_quat1.append(quat1)
            arg_list_quat2.append(quat2)

            c_timer += step_time
            c_counter += 1
            if c_timer > 1:
                sampling_time = math.ceil(1000 * c_timer / c_counter)
                c_timer = 0
                c_counter = 0


# Concurrent thread for acquiring EMG data as fast as possible
def emg_concurrent_function(arg_emg_channel, arg_emg_voltage, arg_emg_lock, arg_flag_emg_ready):
    while True:
        # Acquire EMG data from the ADC
        arg_emg_voltage.value = arg_emg_channel.voltage
        time.sleep(0.0001)


# Function for the parallel process of sensor data acquisition and motion tracking
def acquisition_function(arg_window_length, arg_window_increment,
                         arg_flag_calibration_complete, arg_data_lock, arg_sampling_freq_list,
                         arg_emg_list, arg_step_size, arg_joint_angles_0r, arg_real_joint_angles,
                         arg_channel_bis, arg_channel_tris, arg_imu1, arg_imu2):
    # To keep track of the calibration step, create shared variable
    calibration_step = Value('i', 0)
    # Create empty quaternion shared lists and shared list of relative angular velocities for passing data
    motion_manager = Manager()
    list_quat1 = motion_manager.list()
    list_quat2 = motion_manager.list()
    wn = motion_manager.list()
    # (0/3)CALIBRATE INERTIAL MAGNETIC SENSORS
    # Start a concurrent thread for accurate and reliable motion tracking
    motion_tracking_thread = Thread(target=motion_tracking_function,
                                    args=(
                                        arg_imu1, arg_imu2, calibration_step, arg_flag_calibration_complete, list_quat1,
                                        list_quat2, wn, arg_window_increment))
    motion_tracking_thread.start()

    # (1/3)CONVERGE ON INITIAL POSITION FOR THE MADGWICK FILTER
    # Wait for the concurrent thread to converge
    while calibration_step.value == 0:
        time.sleep(0.05)

    # (2/3)CONVERGE ON SENSOR-TO-SEGMENT ORIENTATIONS THROUGH GRADIENT DESCENT
    # Wait for new data to be acquired
    while calibration_step.value == 1:
        time.sleep(0.05)

    print('Data acquisition successful! You can rest now.')
    print('Converging on a pair of joint DoF axes...')

    # Initialize phi vector of spherical coordinate angles
    phi = [math.radians(45), math.radians(45), math.radians(90), math.radians(45)]

    # Compute the inverted quaternions
    inv_quat1 = [quat_inversion(q) for q in list_quat1]
    inv_quat2 = [quat_inversion(q) for q in list_quat2]

    # Converge on a pair of joint DoF axes
    for i in range(0, len(wn), 100):  # Take a step every 10 data points
        if i + 100 <= len(wn):  # To prevent out of bounds errors
            # Take another step in the joint angle estimation algorithm
            phi = next_step_phi(phi, arg_step_size, wn[i:i + 100], list_quat1[i:i + 100],
                                list_quat2[i:i + 100], inv_quat1[i:i + 100], inv_quat2[i:i + 100])

    # For determining angle offset type
    _list_type_quat1 = list_quat1[:]
    _list_type_quat2 = list_quat2[:]
    # Clear the quaternion lists
    list_quat1[:] = []
    list_quat2[:] = []

    # Signal the main process and the child thread that the motion tracking algorithm has converged on a pair of joint DoF axes
    arg_flag_calibration_complete.value = True
    # Signal the motion tracking thread to start regular data acquisition
    calibration_step.value += 1

    # (3/3) BUILD THE FIRST WINDOW IN A KNOWN INITIAL POSITION
    # Wait for the user to be in position
    while arg_flag_calibration_complete.value:
        pass

    # Compute the sensor to segment orientations from the DoF axes j1 and j2 - the starting values are considered to be zero
    sts1, sts2 = sensors_to_segments(phi, 0., 0.)
    # Compute the NOT ADJUSTED segment to sensor orientations
    badsts1, badsts2 = sensors_to_segments(phi, 0., 0.)

    # Collect sEMG data for 15 seconds
    _t = time.time()
    _l_bis = []
    _l_tris = []
    while time.time() - _t < 15:
        _l_bis.append(arg_channel_bis.voltage)
        _l_tris.append(arg_channel_tris.voltage)
    # Compute signal offset
    emg_offset_bis = np.mean(_l_bis)
    emg_offset_tris = np.mean(_l_tris)

    print('sEMG offset is {:02.04f} and {:02.04f}'.format(emg_offset_bis, emg_offset_tris))

    # Save quaternions in throwaway list
    _list_q1 = list_quat1[-500:]
    _list_q2 = list_quat2[-500:]
    # Create the variables for estimated joint angles
    alfa0e, beta0e, gamma0e = 0., 0., 0.

    # Estimate for all angles in the last period
    for i in range(len(_list_q1)):
        # Initialize starting joint angle variables - "e" stands for "estimated" and "r" stands for "reference" (already known)
        alfa0e, beta0e, gamma0e = np.array([alfa0e, beta0e, gamma0e]) + np.array(
            joint_eulers(_list_q1[i], _list_q2[i], sts1, sts2))
    # Compute the mean estimate
    alfa0e /= len(_list_q1)
    beta0e /= len(_list_q1)
    gamma0e /= len(_list_q1)

    # Compute the adjusted segment to sensor orientations
    sts1, sts2 = sensors_to_segments(phi, math.radians(alfa0e - arg_joint_angles_0r[0]),
                                     math.radians(arg_joint_angles_0r[2] - gamma0e))
    for i in range(len(_list_q1)):
        # Compute the real joint angles and append to the end of the list
        arg_real_joint_angles.append(joint_eulers(_list_q1[i], _list_q2[i], sts1, sts2)[0])
    # Compute the offset AFTER adjusting the sensor-to-segment orientations
    arg_real_joint_angles.sort()
    mean_at_zero = np.mean(arg_real_joint_angles[:10])

    arg_real_joint_angles[:] = []
    for i in range(len(_list_type_quat1)):
        # Compute the real joint angles and append to the end of the list
        arg_real_joint_angles.append(joint_eulers(_list_type_quat1[i], _list_type_quat2[i], sts1, sts2)[0])
    mean_type = np.mean(arg_real_joint_angles[:])
    type_angle = ''
    joint_angle_offset = 0
    final_offset = 0
    if mean_at_zero > -90 and mean_at_zero < 90:
        # Type is A or B
        if mean_type > 0:
            type_angle = 'A'
            joint_angle_offset = mean_at_zero
        else:
            type_angle = 'B'
            joint_angle_offset = mean_at_zero
    else:
        # Type is C or D
        if mean_type > 0:
            type_angle = 'C'
            arg_real_joint_angles[:] = np.abs(arg_real_joint_angles[:])
            arg_real_joint_angles.sort()
            joint_angle_offset = np.mean(arg_real_joint_angles[:10])
            arg_real_joint_angles[:] = [180 - _i + joint_angle_offset for _i in arg_real_joint_angles[:]]
            arg_real_joint_angles.sort()
            final_offset = np.mean(arg_real_joint_angles[:10])
        else:
            type_angle = 'D'
            arg_real_joint_angles[:] = np.abs(np.negative(arg_real_joint_angles[:]))
            arg_real_joint_angles.sort()
            joint_angle_offset = np.mean(arg_real_joint_angles[:10])
            arg_real_joint_angles[:] = [180 - _i + joint_angle_offset for _i in arg_real_joint_angles[:]]
            arg_real_joint_angles.sort()
            final_offset = np.mean(arg_real_joint_angles[:10])
    print('Joint angle is type {} and offset is {:03.02f}.'.format(type_angle, joint_angle_offset))

    # Signal the main process that calibration is complete
    arg_flag_calibration_complete.value = True

    # Begin endless loop
    while True:
        _time0 = time.time()
        arg_data_lock.acquire()  # Lock for process safety
        for _ in range(arg_window_increment.value):
            # Get voltage readings from the ADC (which is connected to both EMG sensors)
            arg_emg_list.append((arg_channel_bis.voltage - emg_offset_bis, arg_channel_tris.voltage - emg_offset_tris))

        # Compute the real joint angles
        _angle = joint_eulers(list_quat1[-1], list_quat2[-1], sts1, sts2)[0]
        if type_angle == 'A':
            _angle = np.abs(_angle) - joint_angle_offset
        elif type_angle == 'B':
            _angle = np.abs(-_angle + joint_angle_offset)
        elif type_angle == 'C':
            _angle = 180 - np.abs(_angle - joint_angle_offset) - final_offset
        elif type_angle == 'D':
            _angle = 180 - np.abs(-_angle + joint_angle_offset) - final_offset

        # Saturate
        if _angle < 0:
            arg_real_joint_angles.append(0)
        #elif _angle > 150:
        #    arg_real_joint_angles.append(150)
        else:
            arg_real_joint_angles.append(_angle)

        # Compute the sampling frequency
        arg_sampling_freq_list.append(arg_window_increment.value / (time.time() - _time0))
        arg_data_lock.release()  # Release lock
        #badangles = joint_eulers(list_quat1[-1], list_quat2[-1], badsts1, badsts2)
        #print('\navg_quat1: {}\navg_quat2: {}\angles: {}\nadjusted angles: {}\nfreq: {}'.format(list_quat1[-1],
         #                                                                                       list_quat2[-1],
          #                                                                                      badangles,
           #                                                                                     arg_real_joint_angles[
            #                                                                                        -1],
             #                                                                                   arg_sampling_freq_list[
              #                                                                                      -1]))


# Return the filtered and rectified window of EMG data
def preprocessing(arg_window, arg_fs):
    # Bandpass filtering
    fc_low = 450  # Low-pass cut-off frequency
    fc_high = 20  # High-pass cut-off frequency
    # Normalize the frequencies (Nyquist)
    wc_low = fc_low / (arg_fs / 2)
    wc_high = fc_high / (arg_fs / 2)
    filtered_window = arg_window
    if arg_fs > 2 * fc_low:  # Only apply the low-pass filter if the sampling frequency is high enough
        # Compute design parameters for the low-pass filter
        _b, _a = signal.butter(4, wc_low, 'low')
        # Low-pass filter the EMG signals
        filtered_window = np.array(signal.filtfilt(_b, _a, filtered_window))
    if arg_fs > 2 * fc_high:  # Only apply the high-pass filter if the sampling frequency is high enough
        # Compute design parameters for the high-pass filter
        _b, _a = signal.butter(2, wc_high, 'high')
        # High-pass filter the EMG signals
        filtered_window = signal.filtfilt(_b, _a, filtered_window)

    # Notch filtering
    f_notch = 50  # Notch frequency
    f_upper = 50.5  # Specify the -3dB bandwidth
    f_lower = 49.5
    if arg_fs > 2 * f_notch:  # Only apply the notch filter if the sampling frequency is high enough
        # Obtain normalized frequency
        w_notch = f_notch / (arg_fs / 2)
        # Compute the quality factor
        Q_factor = math.tan(math.pi * f_notch / arg_fs) / (
                    math.tan(math.pi * f_upper / arg_fs) - math.tan(math.pi * f_lower / arg_fs))
        # Obtain design parameters for the notch filter
        _b, _a = signal.iirnotch(w_notch, Q_factor)
        # Notch filter the EMG signals
        filtered_window = signal.filtfilt(_b, _a, filtered_window)

    return filtered_window, np.absolute(filtered_window)  # Rectify the signal and output both


# Compute the Fourier transform of a sample based on sampling frequency
def fourier_transform(arg_data, arg_fs):
    # Fourier transform of the sample
    _fourier = np.abs(np.fft.fft(np.array(arg_data)))
    _n = len(arg_data)
    # Frequency series for the transformed data
    _freq = np.fft.fftfreq(_n, _n / arg_fs) * _n

    return _fourier, _freq


# Compute the cepstrum coefficients for a given window
def cepstrum_coefs(arg_df, arg_order):
    func_coefs = []  # Initiate coefficients as an empty list

    aux_model = AutoReg(arg_df, arg_order - 1, old_names=False)  # Create AutoRegressive model
    model_result = aux_model.fit()  # Fit model

    model_params = model_result.params  # Get Autoregressive model coefficients

    func_coefs.append(-model_params[0])  # c_1 = -a_1

    for i in range(1, 4):  # c_i = -a_i - sum((1 - n / (i + 1)) * a_n * c_{n-i} ,  n = 1, ... , i-1
        func_coefs.append(
            -model_params[i] - sum([(1 - n / (i + 1)) * model_params[n] * func_coefs[i - n] for n in range(1, i - 1)]))

    return func_coefs


# Extracts features for a single window
def feature_extraction(arg_window):
    _features = []  # Create empty list for all features

    # SAMPEN (Sample Entropy)
    _m = 2  # Dimensions
    _r = 0.2  # Tolerance

    _window = np.array(arg_window)
    _window[_window < 0.000001] = 1e-6
    _window[_window > 1] = 1e-3

    _global_tolerance = _r * np.std(_window)  # Global tolerance

    _features.append(ent.sample_entropy(_window, _m, _global_tolerance)[0])

    # CC (Cepstrum Coefficients)
    _order = 4  # Order of the CC regression

    aux_cepstrum = cepstrum_coefs(_window, _order)  # Save the function's output to avoid calling it multiple times
    for j in range(0, _order):  # Append all coefficients
        _features.append(aux_cepstrum[j])

    # RMS (Root Mean Square)
    _features.append(np.sqrt(np.mean(_window ** 2)))

    # WL (Waveform Length)
    # WL is the sum of all absolute differences between every two consecutive elements
    _features.append(np.sum(np.abs(np.diff(_window))))

    return _features


# Build a dataframe from all features
def build_feature_df(arg_bis, arg_tris, arg_angle):
    return pd.DataFrame({"SampEn_bis": [row[0] for row in arg_bis], "SampEn_tris": [row[0] for row in arg_tris],
                         "CC1_bis": [row[1] for row in arg_bis], "CC2_bis": [row[2] for row in arg_bis],
                         "CC3_bis": [row[3] for row in arg_bis], "CC4_bis": [row[4] for row in arg_bis],
                         "CC1_tris": [row[1] for row in arg_tris], "CC2_tris": [row[2] for row in arg_tris],
                         "CC3_tris": [row[3] for row in arg_tris], "CC4_tris": [row[4] for row in arg_tris],
                         "RMS_bis": [row[5] for row in arg_bis], "RMS_tris": [row[5] for row in arg_tris],
                         "WL_bis": [row[6] for row in arg_bis], "WL_tris": [row[6] for row in arg_tris],
                         "angle": arg_angle})


# Random Forest cross-validation with penalization over computational cost of prediction
def rf_cv_with_score_cost(n_estimators, max_features, max_samples, min_samples_split, x_train, y_train):
    global k_folds
    # Grow Random Forest
    rf = ml.RandomForestRegressor(criterion='mse', n_estimators=n_estimators, max_features=max_features, bootstrap=True,
                                  max_samples=max_samples, min_samples_split=min_samples_split, n_jobs=-1)  # n_jobs=-1 uses all possible processes

    cv_results = cross_validate(estimator=rf, X=x_train, y=y_train, scoring=None, n_jobs=-1)
    # scoring=None reverts to default (the estimator's score method), n_jobs=-1 uses all possible processes
    # return_estimator=True returns the model. Not useful here since we want to use training and validation data for the
    # final model

    return cv_results['test_score'].mean()
    # return cv_results.test_score.mean() / math.sqrt(cv_results.score_time.mean())


# Function for optimizing the Random Forest with cross-validation
def optimize_rf(data, targets, hyperbounds, probe_point):
    def rf_cross_val(n_estimators, max_features, max_samples, min_samples_split):  # Wrapper of Random Forest cross-validation
        return rf_cv_with_score_cost(
            n_estimators=round(n_estimators),  # Ensure parameters are integers
            max_features=max(min(max_features, 0.999), 1e-3),  # Ensure max_features is in the (0, 1) range
            max_samples=max(min(max_samples, 0.999), 1e-3),  # Ensure max_samples is in the (0, 1) range
            min_samples_split=round(min_samples_split),
            x_train=data,
            y_train=targets,
        )

    optimizer = BayesianOptimization(
        f=rf_cross_val,  # Black-box function being optimized
        pbounds=hyperbounds,  # Region of hyperparameter space to explore
        verbose=2,  # verbose=1 prints only when a maximum is observed, verbose=0 never prints
        random_state=1234,  # Establish a constant random_state to ensure data is reproducible
    )

    optimizer.probe(  # This forces the optimizer to search in a specific point of the hyperparameter space
        # These points are independent of the 'iter_points' and 'n_iter', that is, they don't count towards these limits
        params=probe_point,  # The order has to be alphabetical
        lazy=True,  # lazy=True means the point will only be probed at the next 'maximize' call
        # prior to the Gaussian Processes taking over
    )

    optimizer.maximize(  # This method naturally prints every iteration given verbose=2
        init_points=2,  # How many steps of random exploration before Bayesian Optimization, helps spreading the search
        n_iter=7,  # How many steps of Bayesian Optimization
        acq='ei',  # Use Expected Improvement as the acquisition function
        xi=0.1  # Tune the acquisition function's parameter
    )

    # Save the values
    _list_obs = [[res["params"]["n_estimators"] for res in optimizer.res],
                 [res["params"]["max_features"] for res in optimizer.res],
                 [res["params"]["max_samples"] for res in optimizer.res],
                 [res["params"]["min_samples_split"] for res in optimizer.res],
                 [res["target"] for res in optimizer.res]]

    return _list_obs, optimizer.max['params']  # Return the maximum point's hyperparameters


# Compute the arm's current angle based on the encoder's latest readings
def get_encoder_angle(arg_encoder, arg_encoder_bits, arg_speed_reduction):
    # 13 bits per revolution, thus 2pi / 2^bits radians per count
    # radians_per_counts = 2. * math.pi / (2 ** arg_encoder_bits)
    # return radians_per_counts * arg_encoder.read() / arg_speed_reduction  # Read the encoder's output and convert
    degrees_per_counts = 360 / (2 ** arg_encoder_bits) * 2
    return - degrees_per_counts * arg_encoder.read() / arg_speed_reduction  # Read the encoder's output and
    # invert its direction, convert to degrees and consider speed reduction


# Compute the next step of the PID's control action
def pid(arg_theta_ref, arg_theta_enc, arg_delta_t, arg_error_p, arg_error_i):
    # Controller gains
    kp = 1.4
    ti = 0.28
    td = 0.3

    # Save previous error for derivative error
    previous_error = arg_error_p

    # Compute the proportional, integral and derivative errors
    arg_error_p = arg_theta_ref - arg_theta_enc
    arg_error_i += arg_error_p * arg_delta_t
    if arg_delta_t > 0:
        error_d = (arg_error_p - previous_error) / arg_delta_t
    else:
        error_d = 0

    # Compute the proportional, integral and derivative control action
    u_p = arg_error_p * kp
    u_i = arg_error_i * kp / ti
    u_d = error_d * kp * td

    # Return the total control action
    u_pid = u_p + u_i + u_d

    return u_pid, arg_error_p, arg_error_i


# Translate the PID control action to a duty cycle command
def pid_to_255pwm(arg_pid_action, arg_theta, arg_pin1, arg_pin2):
    # Stop the motor if the elbow is near its range of motion and the control action is pushing it further
    if arg_pid_action > 0 and arg_theta > 115:
        arg_pid_action = 0
    else:
        # Update duty cycle
        if arg_pid_action < 0:  # If commanded to lower
            arg_pid_action = -arg_pid_action  # Take the symmetric

            if arg_theta > 20:  # Weight does the job
                arg_pid_action /= 10

            # Change the motor direction
            GPIO.output(arg_pin1, GPIO.LOW)
            GPIO.output(arg_pin2, GPIO.HIGH)

            # Maximum duty cycle is 100% or 255 bits
            if arg_pid_action > 100:
                arg_pid_action = 100
        else:  # If commanded to rise
            # Change the motor direction
            GPIO.output(arg_pin1, GPIO.HIGH)
            GPIO.output(arg_pin2, GPIO.LOW)

            # Maximum duty cycle is 100% or 255 bits
            if arg_pid_action > 100:
                arg_pid_action = 100

    return arg_pid_action


# Function for the DC motor control process
def dc_control(arg_real_joint_angle, arg_ref_angle, arg_fs, arg_flag_start, arg_flag_KeyboardInterrupt):
    # Set up GPIO.BCM numbering
    GPIO.setmode(GPIO.BCM)

    # Encoder setup
    enc_A = 5  # Pins for the quadrature encoder, GPIO 5 and 5 are board pins 29 and 31, respectively
    enc_B = 6
    encod = Encoder.Encoder(enc_A, enc_B)  # Use the input pins GPIO 5 and GPIO 6 for the quadrature encoder
    enc_R = 26  # Reference pin of the encoder, GPIO 26 is board pin 37
    GPIO.setup(enc_R, GPIO.IN)  # Set the encoder reference pin as an output
    GPIO.add_event_detect(enc_R, GPIO.RISING, callback=encoder_ref_state)
    enc_correction = 0  # Correction based on the encoder reference to prevent drift

    # PWM Setup
    pwm_pin = 25  # BOARD is 22
    pwm_freq = 10000  # For a 10 [kHz] PWM output
    GPIO.setup(pwm_pin, GPIO.OUT)  # Set the pwm_pin as an output
    pwm_channel = GPIO.PWM(pwm_pin, pwm_freq)  # Create a PWM instance

    # Set up motor driver's direction pins
    Hbridge_IN1 = 23  # BOARD is 16
    Hbridge_IN2 = 24  # BOARD is 18
    GPIO.setup(Hbridge_IN1, GPIO.OUT)
    GPIO.setup(Hbridge_IN2, GPIO.OUT)
    # Initiate the motor driver's direction at low
    GPIO.output(Hbridge_IN1, GPIO.LOW)
    GPIO.output(Hbridge_IN2, GPIO.LOW)

    duty_cycle = 0  # Initiate duty cycle at zero
    pwm_channel.start(duty_cycle)  # Start the PWM output

    # Initiate variables for the PID controller
    error_p = 0
    error_i = 0

    # First iteration
    # Read encoder data
    encoder_bits = 13
    speed_reduction = 3
    device_theta = get_encoder_angle(encod, encoder_bits, speed_reduction) + enc_correction

    elapsed_time = 0  # To store the latest time and start counting the elapsed time
    # Update PID
    pid_action, error_p, error_i = pid(arg_ref_angle.value, device_theta, 0, error_p, error_i)

    while not arg_flag_start.value:
        pass

    # Create new data file
    with open('encoder_data.csv', 'w') as fd:
        fd.write('time,joint_angle,ref_angle,device_angle,fs')
        fd.write('\n{},{},{},{},{}'.format(elapsed_time, arg_real_joint_angle[-1], arg_ref_angle.value, device_theta, arg_fs[-1]))

    # Store the first iteration's variables
    previous_theta = device_theta
    previous_time = time.time()

    # Issue command to DC motor
    duty_cycle = pid_to_255pwm(pid_action, device_theta, Hbridge_IN1, Hbridge_IN2)

    # Update PWM output
    pwm_channel.ChangeDutyCycle(duty_cycle)

    # Change the motor direction
    GPIO.output(Hbridge_IN1, GPIO.LOW)
    GPIO.output(Hbridge_IN2, GPIO.HIGH)

    while True:
        # Read encoder data
        device_theta = get_encoder_angle(encod, encoder_bits,
                                         speed_reduction) + enc_correction  # Read the encoder's output
        delta_t = time.time() - previous_time
        elapsed_time += delta_t

        # Update the output .csv file
        with open('encoder_data.csv', 'a') as fd:
            fd.write('\n{},{},{},{},{}'.format(elapsed_time, arg_real_joint_angle[-1], arg_ref_angle.value, device_theta, arg_fs[-1]))

        delta_t = time.time() - previous_time
        previous_theta = device_theta
        previous_time = time.time()

        # Update PID
        pid_action, error_p, error_i = pid(arg_ref_angle.value, device_theta, delta_t, error_p, error_i)

        # Issue command to DC motor
        duty_cycle = pid_to_255pwm(pid_action, device_theta, Hbridge_IN1, Hbridge_IN2)

        # Update PWM output
        pwm_channel.ChangeDutyCycle(duty_cycle)

        # Catch any interruptions and exit cleanly
        if arg_flag_KeyboardInterrupt.value:
            # Clean up any previous GPIO use
            GPIO.cleanup()
            sys.exit()
            # STOP button sends a SIGINT exception, waits one second, then sends SIGKILL

        time.sleep(0.0005)


# ==============================================  MAIN  ==============================================

# ==============================================  SETUP  ==============================================
# This step is for turning on sensors, connecting to them, setting up data storage and other variables

# Initiate I2C bus

i2c = busio.I2C(board.SCL, board.SDA)

# Setup ADC for EMG sensor data acquisition and conversion

print("Setting up EMG sensors and ADC connection...")
ADC_bis = ADS.ADS1115(i2c, gain=2)  # Create ADC object, with default address and PGA gain of 2
ADC_tris = ADS.ADS1115(i2c, gain=2, address=0x49)  # Create ADC object, with address 0x49 and PGS gain of 2

# Setup a differential mode channel for the first EMG sensor, on pins A0 and A1 of the first ADC
channel_bis = AnalogIn(ADC_bis, 0, 1)
# Setup a differential mode channel for the second EMG sensor, on pins A0 and A1 of the second ADC
channel_tris = AnalogIn(ADC_tris, 0, 1)
print("Connection to ADC successfully established.")

# Setup IMU

print("\nSetting up inertial sensors...")
# Create IMU objects
IMU1 = qwiic_icm20948.QwiicIcm20948(address=0x68)
IMU2 = qwiic_icm20948.QwiicIcm20948(address=0x69)

if not IMU1.connected:
    sys.exit("IMU1 is not connected. Setup aborted.")
elif not IMU2.connected:
    sys.exit("IMU2 is not connected. Setup aborted.")

print("Connection to IMUs successfully established.")
IMU1.begin()  # Begin communication with IMU1
IMU2.begin()  # Begin communication with IMU2

# ==============================================  BASELINE PROCESS  ==============================================
# Baseline process for continuous sensor data acquisition and IMU calibration.

print("\nInitiating baseline process for continuous sensor data acquisition and IMU calibration.")

# Define shared variables
window_length = Value('i', int(input("Define the target window size:")))  # Define target window length
window_increment = Value('i', int(input("Define the starting window increment:")))  # Define window increment length
# A Value is a parallel computing safe class that is defined by its type (in this case, 'i' for 'integer') and its value

m = Manager()  # Create manager for shared lists
emg_list = m.list()  # Shared sEMG data list
sampling_freq_list = m.list()  # List of sEMG sampling frequencies
real_joint_angles = m.list()  # Real joint angles computed by the algorithm at each iteration

step_size = Value('f', float(
    input("Define the step size for the gradient descent algorithm:")))  # Define step size for angle computation
joint_angles_0r = [0., 15., 0.]  # Reference joint angles of the initial position

flag_calibration_complete = Value('b', False)  # Ready check to coordinate calibrations

data_lock = Lock()  # Create lock for process safety

# Create the baseline process and start it
acquisition_process = Process(target=acquisition_function, args=(
    window_length, window_increment, flag_calibration_complete, data_lock, sampling_freq_list, emg_list, step_size,
    joint_angles_0r,
    real_joint_angles, channel_bis, channel_tris, IMU1, IMU2))

flag_confirm = input("Perform calibration? Y/N")
flag_skip = False
if (flag_confirm == "Y") or (flag_confirm == "y"):
    # Initial calibration
    print("\nThree-step initial calibration required.")
    print('\n(1/3) Calibration of inertial magnetic sensors and convergence of Madgwick filters required.')
    print("Keep the arm in a vertical, relaxed and fully extended position for about 30 seconds.")
    input('Ready?')
    print("Commencing in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Recording...")

    acquisition_process.start()

    # Wait for initial calibration to be complete
    while not flag_calibration_complete.value:
        pass

    print('Inertial magnetic sensors calibrated successfully.')

    time.sleep(0.5)

    print('\n(2/3) Calibration for sensor-to-segment orientation required.')
    print(
        "Repeat consecutive movements of elbow flexion into wrist supination into wrist pronation into elbow extension for 45 seconds.")
    input('Ready?')
    print("Commencing in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Recording...")

    # Signal the baseline process to start calibrating
    flag_calibration_complete.value = False

    # Wait for data acquisition and sensor-to-segment calibration and convergence to be complete
    while not flag_calibration_complete.value:
        time.sleep(5)
        print('...')

    print("Sensor-to-segment orientations converged successfully!")
    print('\n(3/3) Finally, sensor quaternions have to be computed for the initial position, to adjust for offset.')
    print('Calibration required.')
    print("Keep the arm in a vertical, relaxed and fully extended position for about 15 seconds.")
    input('Ready?')
    print("Commencing in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Recording...")
    time.sleep(1)

    # Signal the baseline process to start recording data
    flag_calibration_complete.value = False

    # Wait for initial position measurements to be complete
    while not flag_calibration_complete.value:
        pass

    print('Joint angle successfully adjusted for offset.')
    time.sleep(0.3)
    print('\nALL CALIBRATION STEPS COMPLETED.')
    time.sleep(1)
else:
    flag_skip = True

# ==============================================  TRAINING PHASE  ==============================================
# This step is for collecting data and training the Random Forest algorithm

# Create lists of EMG data, sampling frequencies and angles
list_emg = []
list_fs = []
list_angles = []
list_time = []

print("\n\n== TRAINING PHASE ==")

# Create KeyboardInterrupt event handler for stopping trials and exit script safely
sig_signal.signal(sig_signal.SIGINT, signal_handler)
flag_KeyboardInterrupt = Value('b', False)  # Flag for detecting SIGINT events, initiate it as False

flag_confirm = input('\n(A) Train Random Forest with new data or (B) load existing experimental data? A/B')
if ((flag_confirm == "A") or (flag_confirm == "a")) and not flag_skip:
    remaining_trials = int(input("\nHow many trials do you wish to conduct?"))
    print(str(remaining_trials) + " trials will be performed.")
    flag_confirm = input("Confirm? Y/N")
    if (flag_confirm == "Y") or (flag_confirm == "y"):
        print("Trials will commence shortly.")
    else:
        sys.exit("Training phase aborted. System shutdown.")

    # Trials
    while remaining_trials > 0:
        input("\nNext trial will begin. Ready?")
        print("Commencing in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("Recording...")

        # Flush the shared lists to ready for new data
        data_lock.acquire()  # Lock for process safety
        # Clear all three shared lists
        emg_list[:] = []
        sampling_freq_list[:] = []
        real_joint_angles[:] = []
        data_lock.release()  # Release lock

        # Acquire data for 60 seconds
        time.sleep(60)

        # Copy all data
        data_lock.acquire()  # Lock for process safety
        # Copy all emg data
        _l_emg = emg_list[:]
        # Copy all sampling frequency data
        _l_fs = sampling_freq_list[:]
        # Copy all angle data
        _l_angle = real_joint_angles[:]
        data_lock.release()  # Release lock

        print('Data recorded.')
        print('Mean sampling frequency for this trial was {:03.01f} [Hz]'.format(np.mean(_l_fs)))
        flag_confirm = input("Keep this trial? Y/N")
        if (flag_confirm == "Y") or (flag_confirm == "y"):
            # Append this trial's EMG data list to the complete list
            list_emg.append(_l_emg)
            # Append this trial's sampling frequency data to the complete list
            list_fs.append(_l_fs)
            # Append this trial's elbow flexion/extension angles to the complete list
            list_angles.append(_l_angle)

            # Obtain the corresponding time for each data point
            list_time = []

            for _trial in range(len(list_emg)):
                time_0 = 0
                # Append an empty list for each trial
                list_time.append([])
                for _measurement in range(len(list_emg[_trial])):
                    list_time[_trial].append(1 / list_fs[_trial][_measurement // window_increment.value] + time_0)
                    time_0 = list_time[_trial][-1]
            # Build DataFrame with all data and save everything to .csv file
            _df = pd.DataFrame([list_emg, list_fs, list_angles, list_time])
            _df.to_csv('trial_data.csv', index=False)

            print("Trial kept.\nTrials saved to .csv file.")
            remaining_trials -= 1  # Move on to the next trial
        else:
            print('Trial trashed.')

    print('\nTRIALS COMPLETE.')
    flag_confirm = input('Save trial data in .csv file? Y/N')
    if (flag_confirm == "Y") or (flag_confirm == "y"):
        # Obtain the corresponding time for each data point
        list_time = []
        for _trial in range(len(list_emg)):
            time_0 = 0
            # Append an empty list for each trial
            list_time.append([])
            for _measurement in range(len(list_emg[_trial])):
                list_time[_trial].append(1 / list_fs[_trial][_measurement // window_increment.value] + time_0)
                time_0 = list_time[_trial][-1]
        # Build DataFrame with all data and save everything to .csv file
        _df = pd.DataFrame([list_emg, list_fs, list_angles, list_time])
        _df.to_csv('trial_data.csv', index=False)
        print('Saved data looks like this:')
        print(_df.head())
else:
    # Load data from .csv file selected by user via dialog
    _df = pd.read_csv('trial_data.csv')
    # Split the lists of lists
    list_emg, list_fs, list_angles, list_time = _df.values
    # Convert the lists of string representations of lists into lists of lists
    list_emg = [ast.literal_eval(_list_string) for _list_string in list_emg[:6]]
    list_fs = [ast.literal_eval(_list_string) for _list_string in list_fs[:6]]
    list_angles = [ast.literal_eval(_list_string) for _list_string in list_angles[:6]]
    list_time = [ast.literal_eval(_list_string) for _list_string in list_time[:6]]

    print('Loaded data looks like this:')
    print(_df.head())

# For auxiliary calculations
incs_per_window = window_length.value // window_increment.value  # Number of increments in a window
n_trials = len(list_emg)  # Number of trials in the data set

# Split the EMG data into overlapped windows, starting at data point 0 (1st) to data point 249 (250th)
# All windows will be considered, except for the last, to avoid uneven window lengths
windows_emg = [
    [_l[i:i + window_length.value] for i in range(0, len(_l) - window_length.value + 1, window_increment.value)] for _l
    in list_emg]

# The number of windows is the dataset size divided by the increments (the step), minus the last number of windows that
# would be incomplete
n_windows_per_trial = [len(_l) for _l in windows_emg]

# Split the sampling frequency into overlapped windows by taking the hopping window average
windows_fs = [[np.mean(list_fs[_trial][i:i + incs_per_window]) for i in range(n_windows_per_trial[_trial])] for _trial
              in range(n_trials)]

# Split the angles into a list of the last angles of each window
windows_angles = [
    [list_angles[_trial][i] for i in range(incs_per_window - 1, n_windows_per_trial[_trial] + incs_per_window - 1)] for
    _trial in range(n_trials)]

print("\nData set is composed of {} trials for a total of {} windows.".format(n_trials, np.sum(n_windows_per_trial)))

# Create list of trial lists of DataFrames for each EMG window
list_df_windows = []
_n = 0
for _trial in windows_emg:
    # Append a list of DataFrames (one for each window), per trial, to the list of trial lists
    list_df_windows.append([pd.DataFrame(
        {"emg_bis": [_tuple[0] for _tuple in _window],  # Get the list of all first items in all tuples
         "emg_tris": [_tuple[1] for _tuple in _window]  # Get the list of all second items in all tuples
         }
    ) for _window in _trial])

    print(list_df_windows[_n][0].head())
    _n += 1

flag_confirm = input('(A) Perform feature extraction or (B) load pre-existing train and test data? A/B')
# Record the completion time of each step
start_time = time.time()
if (flag_confirm == 'a' or flag_confirm == 'A'):
    # PRE-PROCESSING

    print("\n\nPre-processing under way...")

    # Store filtered data
    windows_filtered = []

    # Apply pre-processing to every window of every trial
    for _trial in range(n_trials):
        # Append empty list to the filtered data list
        windows_filtered.append([])
        for _window in range(n_windows_per_trial[_trial]):
            # Append empty window to each trial list of filtered data
            windows_filtered[_trial].append([])
            # Do pre-processing on a per window basis, that is, to each DataFrame
            windows_filtered[_trial][_window], list_df_windows[_trial][_window]["preprocessed_bis"] = preprocessing(
                list_df_windows[_trial][_window]["emg_bis"].values, windows_fs[_trial][_window])
            windows_filtered[_trial][_window], list_df_windows[_trial][_window]["preprocessed_tris"] = preprocessing(
                list_df_windows[_trial][_window]["emg_tris"].values, windows_fs[_trial][_window])

    # Record the pre-processing time
    preprocessing_time = time.time() - start_time

    print("Pre-processing complete in approximately {:02.02f} seconds.\nComputing Fourier transforms of EMG data...".format(
        preprocessing_time))

    if False:
        # Compute the Fourier transforms of the raw and pre-processed sEMG signals
        fourier_raw = []
        fourier_filtered = []
        fourier_freq = []
        # Compute for each window of each trial
        for _trial in range(n_trials):
            # Append empty list for each trial
            fourier_raw.append([])
            fourier_filtered.append([])
            fourier_freq.append([])
            for _window in range(n_windows_per_trial[_trial]):
                # Append empty window to each trial list
                fourier_raw[_trial].append([])
                fourier_filtered[_trial].append([])
                fourier_freq[_trial].append([])
                # Compute the Fourier transforms
                fourier_raw[_trial][_window], fourier_freq[_trial][_window] = fourier_transform(windows_emg[_trial][_window],
                                                                         windows_fs[_trial][_window])
                fourier_filtered[_trial][_window], _ = fourier_transform(windows_filtered[_trial][_window], windows_fs[_trial][_window])
        # Write to file
        _df = pd.DataFrame([fourier_raw, fourier_filtered, fourier_freq])
        _df.to_csv('fourier_data.csv', index=False)

    # Record the Fourier transform time
    fourier_time = time.time() - preprocessing_time - start_time

    # FEATURE EXTRACTION
    print("Fourier transforms completed in approximately {:02.02f} seconds.\nFeature extraction under way...".format(
        fourier_time))

    features_bis = []  # Create empty list of lists (one per trial) of feature vectors for the biceps EMG
    features_tris = []  # Create empty list of lists (one per trial) of feature vectors) for the triceps EMG

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     #   for _trial in list_df_windows:
      #      print([_window["preprocessed_bis"] for _window in _trial])
       #     print([_window["preprocessed_tris"] for _window in _trial])

    for _trial in list_df_windows:
        # Append a list of DataFrames (one for each window), per trial, to the list of trial lists
        features_bis.append([feature_extraction(_window["preprocessed_bis"]) for _window in _trial])
        features_tris.append([feature_extraction(_window["preprocessed_tris"]) for _window in _trial])

    # Write biceps' processed data and features extracted to file
    preprocessed_data = [[list(_window['preprocessed_bis'].values) for _window in _trial] for _trial in list_df_windows]
    _df = pd.DataFrame([preprocessed_data, features_bis])
    _df.to_csv('processed_data.csv', index=False)

    # BUILD FINAL DATAFRAME
    df_ready = [build_feature_df(features_bis[_trial], features_tris[_trial], windows_angles[_trial]) for _trial in
                range(n_trials)]
    for _trial in df_ready:
        _trial.replace([np.inf, -np.inf], np.nan)
        _trial.dropna()
        _trial.reset_index(drop=True)

    # Record the total feature extraction time
    feature_extraction_time = time.time() - (fourier_time + preprocessing_time + start_time)

    # SPLIT DATA INTO TRAIN/VALIDATION/TEST SAMPLES
    print(
        "Feature extraction complete in approximately {:02.02f} seconds. Splitting trials into train-validation-test split...".format(
            feature_extraction_time))

    # Remove a number of trials for testing
    test_indexes = list(random.sample(range(0, n_trials), 1))
    test_indexes.sort()
    print('Test trials are trials {}.'.format([_trial + 1 for _trial in test_indexes]))
    # Concatenate them
    test_df = pd.concat([df_ready[_test_index] for _test_index in test_indexes])
    # Reset the indexes
    test_df.reset_index(drop=True, inplace=True)
    # Save test data to .csv file
    test_df.to_csv('test_data.csv', index=False)

    # Remove test sets from the train/validation set
    for _test_index in test_indexes:
        df_ready.pop(_test_index - (n_trials - len(df_ready)))
    # Concatenate data from all non-test trials
    data = pd.concat(df_ready)
    # Reset the indexes
    data.reset_index(drop=True, inplace=True)
    # Save training data to .csv file
    data.to_csv('training_data.csv', index=False)
else:
    data = pd.read_csv('training_data.csv')
    test_df = pd.read_csv('test_data.csv')
    preprocessing_time = 0
    fourier_time = 0
    feature_extraction_time = 0

# Split TRAIN data into dependent and independent variables
X_train = data.loc[:,
          data.columns != "angle"]  # The X_train set is all columns except the dependent variable
Y_train = data.loc[:, data.columns == "angle"]  # The Y_train set is only the dependent variable

# Split TEST data into dependent and independent variables
X_test = test_df.loc[:,
         test_df.columns != "angle"]  # The X_train set is all columns except the dependent variable
Y_test = test_df.loc[:, test_df.columns == "angle"]  # The Y_train set is only the dependent variable

# RANDOM FOREST ALGORITHM
print("Algorithm setup complete.")
# Number of CV folds
k_folds = 5

_ = input('\n\nReady to start training RF?')
setup_time = time.time() - (feature_extraction_time + fourier_time + preprocessing_time + start_time)
print('Training Random Forest with default hyperparameters...')

# Grow Random Forest
rf = ml.RandomForestRegressor(n_estimators=200, max_features=1 / 3,
                              max_samples=math.sqrt(len(X_train) * (k_folds - 1) / k_folds) / len(X_train),
                              min_samples_split=5, criterion='mse')

# Cross-validate
cv_results = cross_validate(estimator=rf, X=X_train, y=Y_train.values.ravel(), scoring=None, n_jobs=-1,
                            return_estimator=True)
# scoring=None reverts to default (the estimator's score method), n_jobs=-1 uses all possible processes
print(
    '\nRandom Forest validated using 5-fold cross-validation on training/validation data. Score computed using MSE: {}'.format(
        cv_results['test_score']))
print('The model will now be trained with all the training/validation data and validated using the test set.')

# Fit Random Forest
rf.fit(X_train, Y_train.values.ravel())  # Y_train.values.ravel() converts the column vector into a (n, ) array
# Record the training time
training_time = time.time() - (setup_time + feature_extraction_time + fourier_time + preprocessing_time + start_time)
print("Training complete in approximately {:02.02f} seconds. Validating with test data set...".format(training_time))

# Make predictions based on test set
prediction_rf = rf.predict(X_test)
# Compute R²
prediction_score = rf.score(X_test, Y_test)
# Record the testing time
test_time = time.time() - (training_time + setup_time + feature_extraction_time + fourier_time + preprocessing_time + start_time)
print('Testing complete in approximately {:02.02f} seconds with R² = {:02.04f}.'.format(test_time, prediction_score))

# Write reference angle and predicted angles to .csv file
_df = pd.DataFrame([list(prediction_rf), list(np.concatenate(Y_test.values))])
_df.to_csv('offline_rf_data.csv', index=False)
print('\nTest data recorded to .csv file.')

# RANDOM FOREST ALGORITHM WITH BAYESIAN OPTIMIZATION
flag_confirm = input('\n\nPerform Bayesian optimization of Random Forest Regressor? Y/N')
if (flag_confirm == 'Y' or flag_confirm == 'y'):
    print("Setting up Bayesian optimization of Random Forest...")

    # Create dictionary of bounded region of hyperparameter space
    hyperbounds = {'n_estimators': (100, 1000), 'max_features': (0.1, 1 / 3), 'min_samples_split': (2, 10),
                   'max_samples': (math.sqrt(len(X_train) * (k_folds - 1) / k_folds) / len(X_train), 1)}

    # Define hyperparameter settings based on empirical rules for tuning of Random Forests
    empirical_tuning = {'n_estimators': 200, 'max_features': 1 / 3, 'min_samples_split': 5,
                        'max_samples': math.sqrt(len(X_train) * (k_folds - 1) / k_folds) / len(X_train)}

    wait_time = time.time() - (test_time + training_time + setup_time + feature_extraction_time + fourier_time + preprocessing_time + start_time)
    print('\nBayesian optimization under way...')

    # Perform Bayesian Optimization
    list_obs, best_params = optimize_rf(X_train, Y_train.values.ravel(), hyperbounds, empirical_tuning)
    # Y_train.values.ravel() converts the column vector into a (n, ) array
    # Write observations and scores to .csv file
    _df = pd.DataFrame(list_obs)
    _df.to_csv('bayes_opt_data.csv', index=False)

    # Train the optimal Random Forest
    rf = ml.RandomForestRegressor(n_estimators=round(best_params['n_estimators']), max_features=best_params['max_features'],
                                  max_samples=best_params['max_samples'], min_samples_split=round(best_params['min_samples_split']),
                                  criterion='mse', bootstrap=True, n_jobs=-1)
    rf.fit(X_train, Y_train.values.ravel())  # Y_train.values.ravel() converts the column vector into a (n, ) array
    # Use all training data to train the optimal Random Forest

    # Record the Bayesian optimization time
    bo_time = time.time() - (wait_time + test_time + training_time + setup_time + feature_extraction_time + fourier_time + preprocessing_time + start_time)
    print("Bayesian optimization and training complete in approximately {:02.02f} seconds. Validating with test data set...".format(bo_time))

    # Make predictions based on test set
    prediction_bo = rf.predict(X_test)
    # Compute R²
    prediction_score = rf.score(X_test, Y_test)
    # Record the testing time
    bo_test_time = time.time() - (bo_time + wait_time + test_time + training_time + setup_time + feature_extraction_time + fourier_time + preprocessing_time + start_time)
    print('Testing complete in approximately {:02.02f} seconds with R² = {:02.04f}.'.format(bo_test_time, prediction_score))

    # Write reference angle and predicted angles to .csv file
    _df = pd.DataFrame([list(prediction_rf), list(prediction_bo), list(np.concatenate(Y_test.values))])
    _df.to_csv('offline_bo_data.csv', index=False)
    print('\nTest data recorded to .csv file.')

    #logger = JSONLogger(path='./logs.json')  # Create new logger
    #optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)  # Log data on every OPTIMIZATION_STEP event
    #load_logs(optimizer, logs=['./logs.json'])  # Load the optimizer with previously computed search space points

# VARIABLE IMPORTANCE
print("Assessing variable importance...")
print("Accuracy = ", rf.score(X_test, Y_test))

feature_list = list(X_train.columns)
feature_imp = pd.Series(rf.feature_importances_, index=feature_list).sort_values(ascending=False)

print(feature_imp)

# ==============================================  ONLINE TESTING PHASE  ==============================================

print("\n\n== ONLINE TESTING PHASE ==")

flag_confirm = input('\nDo you want to use (A) the sEMG data or (B) the IMU data (the measured elbow joint angle) for controlling the robotic arm? A/B')
while flag_confirm != 'A' and flag_confirm != 'B':
    flag_confirm = input(
        '\nInvalid answer.\nDo you want to use (A) the sEMG data or (B) the IMU data (the measured elbow joint angle) for controlling the robotic arm? A/B')

# Shared variable for DC motor control
target_angle = Value('i', 0)  # Initiate target angle at zero
flag_start = Value('b', False)  # Flag for signaling the DC motor control process to start

# Create DC motor online control process
dc_process = Process(target=dc_control, args=(real_joint_angles, target_angle, sampling_freq_list, flag_start, flag_KeyboardInterrupt))

# Start the parallel DC control process
dc_process.start()

input('Online testing lasts for 60 seconds. Ready?')
print("Commencing in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("Recording...")

# Flush the shared lists to ready for new data
data_lock.acquire()  # Lock for process safety
# Clear all three shared lists
emg_list[:] = []
sampling_freq_list[:] = []
real_joint_angles[:] = []
data_lock.release()  # Release lock

# Do a 120 second online testing trial
trial_start_time = time.time()

# Wait until at least one window of data is available
time.sleep(window_length.value / 1000)

# Signal the DC motor control process to start recording data
flag_start.value = True
# Record the time of each step in the intention estimation algorithm
list_t_collect = []
list_t_process = []
list_t_extract = []
list_t_predict = []

# Start trial
while time.time() - trial_start_time < 60:
    _t0 = time.time()

    if flag_confirm == 'A':  # Use the machine learning algorithm's estimate (based on sEMG data) to control the device
        _l = emg_list[-window_length.value:]

        # Get the latest window of data
        online_test_bis = list([_tuple[0] for _tuple in _l])  # Get the list of all first items in all tuples
        online_test_tris = list([_tuple[1] for _tuple in _l])  # Get the list of all second items in all tuples

        list_t_collect.append(time.time() - _t0)

        # Pre-Processing
        _, online_processed_bis = preprocessing(online_test_bis, sampling_freq_list[-1])
        _, online_processed_tris = preprocessing(online_test_tris, sampling_freq_list[-1])

        list_t_process.append(time.time() - _t0)

        # Feature Extraction
        online_features_bis = feature_extraction(online_processed_bis)
        online_features_tris = feature_extraction(online_processed_tris)

        feature_vector = np.array([online_features_bis[0], online_features_tris[0], online_features_bis[1], online_features_bis[2],
                          online_features_bis[3], online_features_bis[4], online_features_tris[1], online_features_tris[2],
                          online_features_tris[3], online_features_tris[4], online_features_bis[5], online_features_tris[5],
                          online_features_bis[6], online_features_bis[6]]).reshape(1, -1)

        list_t_extract.append(time.time() - _t0)

        # Estimate and pass target angle to DC motor
        _angle = rf.predict(feature_vector)
        if _angle > 120:
            target_angle.value = 120
        elif _angle < 0:
            target_angle.value = 0
        else:
            target_angle.value = _angle
    else:  # Use the measured elbow joint flexion/extension angle (based on IMU data) to control the device
        target_angle.value = real_joint_angles[-1]

    list_t_predict.append(time.time() - _t0)

# Copy all data
data_lock.acquire()  # Lock for process safety
# Copy all emg data
_l_emg = emg_list[:]
# Copy all sampling frequency data
_l_fs = sampling_freq_list[:]
# Copy all angle data
_l_angle = real_joint_angles[:]
data_lock.release()  # Release lock

flag_confirm = input("Save online testing data? Y/N")
if (flag_confirm == "Y") or (flag_confirm == "y"):
    _df = pd.DataFrame([list_emg, list_fs, list_angles])
    _df.to_csv('online_data.csv', index=False)

    # Concatenate and write all time data to .csv file
    _df = pd.DataFrame([list_t_collect, list_t_process, list_t_extract, list_t_predict])
    _df.to_csv('time_data.csv', index=False)

print('EXPERIMENT COMPLETE. SYSTEM SHUTDOWN.')
# Clean the RPi's GPIO
flag_KeyboardInterrupt.value = True
# Give it some time
time.sleep(1)
# Hoorah!
sys.exit()