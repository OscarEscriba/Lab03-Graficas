
from gl import *
import pygame
from pygame.locals import *
from Figuras import *
from Material import *
from Lights import *
from texture import Texture
width = 520
height = 520

screen = pygame.display.set_mode((width, height), pygame.SCALED )
clock = pygame.time.Clock()

rt = RendererRT(screen)
rt.enveriomentMap = Texture("parkingLot.bmp")
rt.glClearColor(0.5,0.0,0.0)
rt.glClear()
#Materiales
brick = Material(difuse=[1,0.2,0.2], spec= 32, ks= 0.1)
grass = Material(difuse=[0.2,1,0.2], spec= 64, ks= 0.2)
mirror = Material(difuse=[0.9,0.9,0.9], spec= 128, ks= 0.2, matType=REFLECTIVE)
bluemirror = Material(difuse=[0.5,0.5,1], spec= 128, ks= 0.2, matType=REFLECTIVE)
waterTransparent = Material(ior=1.33, spec=64, ks=0.1, matType=TRANSPARENT)
woodtexture = Material(texture=Texture('woodenBox.bmp'))
metal = Material(difuse=[0.7, 0.7, 0.7], spec=128, ks=0.3)


#Lights 
rt.lights.append(DirectionalLight(direction=[-1,-1,-1], intensity=0.8 ))
rt.lights.append(AmbientLight(intensity=0.1))

#objets
#rt.scene.append(Sphere([0,0,-5], radius=1.5, material=glass)) #la creo en todo
#rt.scene.append(Plane(position=[0,-5,-5], normal=[0,1,0], material=brick))
#rt.scene.append(Sphere([0,0,-5], radius=1, material=brick)) #la creo en todo
#rt.scene.append(Disk(position=[0,-1,-5], normal=[0,1,0], radius=1.5, material=mirror))
#rt.scene.append(AABB(position=[1.5,-1.5,-5], sizes=[1,1,1], material=woodtexture))

# Crear los cilindros (más abajo y pequeños)
cylinder_positions = [[-3, 1.5, -8], [0, 1.5, -8], [3, 1.5, -8]]
cylinder_materials = [bluemirror, mirror, metal]
cylinder_radius = 0.8
cylinder_height = 1.5

for pos, mat in zip(cylinder_positions, cylinder_materials):
    rt.scene.append(Cylinder(position=pos, radius=cylinder_radius, height=cylinder_height, material=mat))

# Crear las pirámides (más pequeñas y abajo)
pyramid_positions = [[-3, -1.5, -8], [0, -1.5, -8], [3, -1.5, -8]]
pyramid_materials = [waterTransparent, grass, metal]
pyramid_base_size = 1.2
pyramid_height = 1.5
pyramid_rotations = [np.radians(20), np.radians(45), np.radians(70)]

for pos, mat, rot in zip(pyramid_positions, pyramid_materials, pyramid_rotations):
    rt.scene.append(Pyramid(position=pos, base_size=pyramid_base_size, height=pyramid_height, material=mat, rotation_angle=rot))

rt.glRender()

isRunning = True

# Bucle principal, corregir esto... 
isRunning = True
while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                isRunning = False
    				
				
pygame.display.flip()
clock.tick(60)
	
pygame.quit()