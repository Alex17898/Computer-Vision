###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    Z=np.array([[0,1,0],
                [-1,0,0],
                [0,0,0]])
    W=np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])
    U,S,V_transpose = np.linalg.svd(E)
    M=U.dot(Z).dot(U.T)
    Q1=U.dot(W).dot(V_transpose)
    R1=np.linalg.det(Q1)*Q1
    T1=U[:3,2:3]
    RT1=np.around(np.hstack((R1,T1)),decimals=4)

    Q2=U.dot(W).dot(V_transpose)
    R2=np.linalg.det(Q2)*Q2
    T2=-U[:3,2:3]
    RT2=np.around(np.hstack((R2,T2)),decimals=4)

    Q3=U.dot(W.T).dot(V_transpose)
    R3=np.linalg.det(Q3)*Q3
    T3=U[:3,2:3]
    RT3=np.around(np.hstack((R3,T3)),decimals=4)

    Q4=U.dot(W.T).dot(V_transpose)
    R4=np.linalg.det(Q4)*Q4
    T4=-U[:3,2:3]
    RT4=np.around(np.hstack((R4,T4)),decimals=4)
    RT=np.stack((RT1,RT2,RT3,RT4))
    return RT
    raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    M=np.size(camera_matrices,0)
    A=np.zeros((2*M,4))
    for i in range(M):
        A[2*i,:]=image_points[i,1]*camera_matrices[i,2,:]-camera_matrices[i,1,:]
        A[2*i+1,:]=camera_matrices[i,0,:]-image_points[i,0]*camera_matrices[i,2,:]
    U,S,V_transpose=np.linalg.svd(A)
    point_3d=V_transpose[3,:3]/V_transpose[3,3]
    return point_3d
    raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    M=np.size(camera_matrices,0)
    y=np.zeros((M, 3))
    error=np.zeros(2*M)
    for i in range(M):
        y[i,:]=camera_matrices[i,:,:].dot(np.append(point_3d,[1]))
        error[2*i]=y[i,0]/y[i,2]-image_points[i,0]
        error[2*i+1]=y[i,1]/y[i,2]-image_points[i,1]
    return error
    raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    y=np.zeros((3,1))
    M=np.size(camera_matrices,0)
    jacobian=np.zeros((2*M,3))
    for i in range(M):
        y=camera_matrices[i,:,:].dot(np.append(point_3d,[1]))
        jacobian[2*i,:]=np.array([(y[2]*camera_matrices[i,0,0]-camera_matrices[i,2,0]*y[0]),(y[2]*camera_matrices[i,0,1]-camera_matrices[i,2,1]*y[0]),(y[2]*camera_matrices[i,0,2]-camera_matrices[i,2,2]*y[0])])/y[2]**2
        jacobian[2*i+1,:]=np.array([(y[2]*camera_matrices[i,1,0]-camera_matrices[i,2,0]*y[1]),(y[2]*camera_matrices[i,1,1]-camera_matrices[i,2,1]*y[1]),(y[2]*camera_matrices[i,1,2]-camera_matrices[i,2,2]*y[1])])/y[2]**2
    return jacobian
    raise Exception('Not Implemented Error')

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    point_3d=linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10):
        e=reprojection_error(point_3d, image_points, camera_matrices)
        J=jacobian(point_3d, camera_matrices)
        point_3d=point_3d-np.linalg.inv((J.T.dot(J))).dot(J.T).dot(e)
    return point_3d
    raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    RT0=estimate_initial_RT(E)
    N=np.size(image_points,0)
    M=K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    M1=K.dot(RT0[0,:,:])
    M2=K.dot(RT0[1,:,:])
    M3=K.dot(RT0[2,:,:])
    M4=K.dot(RT0[3,:,:])
    x1=np.zeros((2,3,4))
    x1[0,:,:]=M
    x1[1,:,:]=M1
    x2=np.zeros((2,3,4))
    x2[0,:,:]=M
    x2[1,:,:]=M2
    x3=np.zeros((2,3,4))
    x3[0,:,:]=M
    x3[1,:,:]=M3
    x4=np.zeros((2,3,4))
    x4[0,:,:]=M
    x4[1,:,:]=M4
    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0
    RT=np.zeros((3,4))
    for i in range(N):
        points_3d1=nonlinear_estimate_3d_point(image_points[i,:,:], x1)
        points_3d2=nonlinear_estimate_3d_point(image_points[i,:,:], x2)
        points_3d3=nonlinear_estimate_3d_point(image_points[i,:,:], x3)
        points_3d4=nonlinear_estimate_3d_point(image_points[i,:,:], x4)
        if(points_3d1[2]>0):
            cnt1=cnt1+1
        if(points_3d2[2]>0):
            cnt2=cnt2+1
        if(points_3d3[2]>0):
            cnt3=cnt3+1
        if(points_3d4[2]>0):
            cnt4=cnt4+1
    if((cnt1>=cnt2)&(cnt1>=cnt3)&(cnt1>=cnt4)):
        for j in range(3):
            for k in range(4):
                RT[j,k]=RT0[0,j,k]
    if((cnt2>=cnt1)&(cnt2>=cnt3)&(cnt2>=cnt4)):
        for j in range(3):
            for k in range(4):
                RT[j,k]=RT0[1,j,k]
    if((cnt3>=cnt1)&(cnt3>=cnt2)&(cnt3>=cnt4)):
        for j in range(3):
            for k in range(4):
                RT[j,k]=RT0[2,j,k]
    if((cnt4>=cnt1)&(cnt4>=cnt2)&(cnt4>=cnt3)):
        for j in range(3):
            for k in range(4):
                RT[j,k]=RT0[3,j,k]
    return RT
    raise Exception('Not Implemented Error')
    

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
