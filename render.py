import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

def render(coords, faces, deform, mode):

    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 640
    light_ambient = [1.0, 1.0, 1.0, 1.0]
    light_position = [0, -2.0, 0, 1.0]
 
    # Initialize the library
    if not glfw.init():
        return
    
    # Set window hint NOT visible
    glfw.window_hint(glfw.VISIBLE, False)
    
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return
        
    # Make the window's context current
    glfw.make_context_current(window)
    glOrtho(-0.5, 0.5, -0.5, 0.5, -1024, 1024)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glDepthFunc(GL_LEQUAL)
    
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)

    if mode == True: 
        glClearColor(0.4, 0.4, 0.4, 1.0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    gluLookAt(0.0, -512.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # Render mesh
    for i in range(len(faces)):

        glBegin(GL_TRIANGLES)

        for j in range(3):

            if mode == True:
                co = 20.0
                glColor3f(deform[faces[i][j+1]][0] * co + 0.4, deform[faces[i][j+1]][1] * co + 0.4, deform[faces[i][j+1]][2] * co + 0.4)
            else:
                glColor3f(1.0, 1.0, 1.0)

            glVertex3f(coords[faces[i][j+1]][0], coords[faces[i][j+1]][1], coords[faces[i][j+1]][2])

        glEnd()

    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)
    image = cv2.flip(image, 0)

    glfw.destroy_window(window)
    glfw.terminate()

    return image


def render_multi(coords, targets, faces, deform, mode):

    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 640
    light_ambient = [1.0, 1.0, 1.0, 1.0]
    light_position = [0, -2.0, 0, 1.0]
 
    # Initialize the library
    if not glfw.init():
        return
    
    # Set window hint NOT visible
    glfw.window_hint(glfw.VISIBLE, False)
    
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return
        
    # Make the window's context current
    glfw.make_context_current(window)
    glOrtho(-0.5, 0.5, -0.5, 0.5, -1024, 1024)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glDepthFunc(GL_LEQUAL)
    
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)

    if mode == True: 
        glClearColor(0.4, 0.4, 0.4, 1.0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    gluLookAt(0.0, -512.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    norm = np.zeros((len(coords), 3), dtype='float32')

    for i in range(len(faces)):
        vec1 = np.subtract(coords[faces[i][2]], coords[faces[i][1]])
        vec2 = np.subtract(coords[faces[i][3]], coords[faces[i][2]])
        vec3 = np.subtract(coords[faces[i][1]], coords[faces[i][3]])

        len1 = np.math.sqrt( np.dot(vec1, vec1) )
        len2 = np.math.sqrt( np.dot(vec2, vec2) )
        len3 = np.math.sqrt( np.dot(vec3, vec3) )

        norm[faces[i][1]] += (len1 * len1 + len3 * len3) * np.cross(vec1, vec2)
        norm[faces[i][2]] += (len2 * len2 + len1 * len1) * np.cross(vec1, vec2)
        norm[faces[i][3]] += (len3 * len3 + len2 * len2) * np.cross(vec1, vec2)

    for i in range(len(coords)) :
        norm[i] = norm[i] / np.math.sqrt( np.dot(norm[i], norm[i]) )


    co = 20.0

    # Render target mesh
    for i in range(len(faces)):

        glBegin(GL_TRIANGLES)

        for j in range(3):
            glColor3f(deform[faces[i][j+1]][0] * co + 0.4, deform[faces[i][j+1]][1] * co + 0.4, deform[faces[i][j+1]][2] * co + 0.4)
            glVertex3f(targets[faces[i][j+1]][0] + norm[faces[i][j+1]][0] * 0.004, targets[faces[i][j+1]][1] + norm[faces[i][j+1]][1] * 0.004, targets[faces[i][j+1]][2] + norm[faces[i][j+1]][2] * 0.004)

        glEnd()

    glClear(GL_DEPTH_BUFFER_BIT)

    # Render initial mesh
    for i in range(len(faces)):

        glBegin(GL_TRIANGLES)

        for j in range(3):

            if mode == True:
                glColor3f(deform[faces[i][j+1]][0] * co + 0.4, deform[faces[i][j+1]][1] * co + 0.4, deform[faces[i][j+1]][2] * co + 0.4)
            else:
                glColor3f(1.0, 1.0, 1.0)

            glVertex3f(coords[faces[i][j+1]][0] + norm[faces[i][j+1]][0] * 0.004 , coords[faces[i][j+1]][1] + norm[faces[i][j+1]][1] * 0.004, coords[faces[i][j+1]][2] + norm[faces[i][j+1]][2] * 0.004) 

        glEnd()

    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)
    image = cv2.flip(image, 0)

    glfw.destroy_window(window)
    glfw.terminate()

    return image
