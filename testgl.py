# See sobel filter example on pycuda wiki

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutInitWindowSize(400, 400)
glutInitWindowPosition(100, 100)
glutCreateWindow("Lattice Boltzmann (0 fps)")
import pycuda.gl.autoinit
import pycuda.driver as drv
import pycuda.gl as cugl
from pycuda.compiler import SourceModule

data = np.zeros((N*M,3),np.uint8)
pbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, pbo)
glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, 0)
pycuda_pbo = cugl.RegisteredImage(int(pbo), GL_TEXTURE_2D)
