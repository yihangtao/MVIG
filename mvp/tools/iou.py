# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def iou3d(bbox1, bbox2):
    bbox1 = get_3d_box(box_size=(bbox1[3], bbox1[4], bbox1[5]), 
                       heading_angle=bbox1[6],
                       center=(bbox1[0], bbox1[1], bbox1[2] + 0.5 * bbox1[5]))
    bbox2 = get_3d_box(box_size=(bbox2[3], bbox2[4], bbox2[5]), 
                       heading_angle=bbox2[6],
                       center=(bbox2[0], bbox2[1], bbox2[2] + 0.5 * bbox2[5]))
    iou, _ = box3d_iou(bbox1, bbox2)
    return iou


def convert_3d_to_2d_bbox(bbox_3d):
    """
    将3D边界框转换为2D边界框格式 [x, y, width, height]
    支持多种输入格式
    """
    if isinstance(bbox_3d, dict):
        # 如果是字典格式，尝试直接提取相关字段
        if all(k in bbox_3d for k in ['x', 'y', 'length', 'width']):
            x, y = bbox_3d['x'], bbox_3d['y']
            length, width = bbox_3d['length'], bbox_3d['width']
            return [x - length/2, y - width/2, length, width]
        elif all(k in bbox_3d for k in ['x1', 'y1', 'x2', 'y2']):
            # 已经是2D格式，直接转换为[x,y,width,height]
            return [bbox_3d['x1'], 
                    bbox_3d['y1'], 
                    bbox_3d['x2'] - bbox_3d['x1'], 
                    bbox_3d['y2'] - bbox_3d['y1']]
    
    # 如果是数组/列表格式 [x, y, z, length, width, height, yaw]
    elif isinstance(bbox_3d, (list, tuple, np.ndarray)):
        if len(bbox_3d) >= 5:  # 至少包含x,y,z,length,width
            x, y = bbox_3d[0], bbox_3d[1]
            length, width = bbox_3d[3], bbox_3d[4]
            return [x - length/2, y - width/2, length, width]
        elif len(bbox_3d) == 4:
            # 可能已经是[x,y,width,height]格式
            return bbox_3d
    
    # 如果无法解析，返回原始输入
    print(f"Warning: Unable to convert bbox format: {bbox_3d}")
    return bbox_3d

# 修改原始iou2d函数以自动处理不同格式
def iou2d(bbox1, bbox2):
    """
    计算两个边界框的IoU，自动处理3D和2D格式
    """
    # 首先转换为2D格式
    bbox1_2d = convert_3d_to_2d_bbox(bbox1)
    bbox2_2d = convert_3d_to_2d_bbox(bbox2)
    
    # 然后转换为字典格式
    bb1 = {'x1': bbox1_2d[0], 'x2': bbox1_2d[0] + bbox1_2d[2], 
           'y1': bbox1_2d[1], 'y2': bbox1_2d[1] + bbox1_2d[3]}
    bb2 = {'x1': bbox2_2d[0], 'x2': bbox2_2d[0] + bbox2_2d[2], 
           'y1': bbox2_2d[1], 'y2': bbox2_2d[1] + bbox2_2d[3]}
           
    # 确保边界框有效
    if bb1['x1'] >= bb1['x2']:
        bb1['x1'], bb1['x2'] = bb1['x1'] - 0.01, bb1['x1'] + 0.01
    if bb1['y1'] >= bb1['y2']:
        bb1['y1'], bb1['y2'] = bb1['y1'] - 0.01, bb1['y1'] + 0.01
        
    if bb2['x1'] >= bb2['x2']:
        bb2['x1'], bb2['x2'] = bb2['x1'] - 0.01, bb2['x1'] + 0.01
    if bb2['y1'] >= bb2['y2']:
        bb2['y1'], bb2['y2'] = bb2['y1'] - 0.01, bb2['y1'] + 0.01
    
    # 计算IoU
    # 确定交集矩形的坐标
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的面积
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # 计算IoU
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    # 确保IoU在有效范围内
    iou = max(0.0, min(iou, 1.0))
    
    return iou