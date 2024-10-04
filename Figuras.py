import numpy as np
from Intercept import *
from math import atan2, acos, cos, pi, isclose, sin
class Shape(object):
    def __init__(self, position, material):
        self.position = position
        self.type = "None"
        self.material = material
    
    def ray_intersect(self, origin, dir):
        return None

class Sphere(Shape):
    def __init__(self, position, radius, material):
        super().__init__(position , material)  # Corrige la llamada a super()
        self.radius = radius
        self.type = "Sphere"
    
    def ray_intersect(self, origin, dir):
        L = np.subtract(self.position, origin)
        tca = np.dot(L, dir)
        a = np.linalg.norm(L)
        f = a **2
        d = f - tca**2
        if d > self.radius:
            return None
        thc = (self.radius ** 2 - d ** 2) ** 0.5 
        t0 = tca - thc
        t1 = tca + thc
        #esto quiere decir que esta atras
        if t0 < 0:
            t0 = t1
        if t0< 0:
            return None
        P = np.add(origin, np.multiply(dir, t0))
        normal = np.subtract(P, self.position) 
        normal /= np.linalg.norm(normal)
        
        #normal como la direccion del rayo para calcular la direccion de la esfera texturizada
        u = (atan2(normal[2], normal[0])) / (2*pi) +0.5
        v = acos(-normal[1]) /pi

        return Intercept(point=P, normal=normal, distance=t0, obj=self, rayDirection=dir, texCords=[u,v])


class Plane(Shape):
    def __init__(self, position, material, normal):
        super().__init__(position, material)
        self.normal = normal
        self.type = "Plane"
    
    def ray_intersect(self, origin, dir):
        #distance = (plane Pos - rayOrigin o normal) / rayDir o nromal
        denom = np.dot(dir, self.normal)

        if isclose(0, denom):
            return None
        num = np.dot(np.subtract (self.position, origin), self.normal)

        t = num/denom

        if t < 0:
            return None

        # Puntodecontacto = origin  + dir * to
        P = np.add(origin, np.array(dir) * t)
        
        return Intercept(point=P, 
                         normal=self.normal,
                         distance=t,
                         texCords=None,
                         rayDirection=dir,
                         obj=self)
    

class Disk(Plane):
    def __init__(self, position, material, normal, radius):
        super().__init__(position, material, normal)
        self.radius  = radius
        self.type = "Disk"
    
    def ray_intersect(self, origin, dir):
        planeIntercept =  super().ray_intersect(origin, dir)
        if planeIntercept is None:
            return None
        contact = np.subtract(planeIntercept.point, self.position)
        contact = np.linalg.norm(contact) # vamos a sacar la magnitud de esto
        
        #para ignorar todo dentro del radio
        if contact > self.radius:
            return None

        return planeIntercept


#estas siglas significan que son access axis aligne bunding box
#es un cubo que no tiene rotacion, que esta alineado a x, y , z del mundo
""
class AABB(Shape):
    def __init__(self, position, material, sizes):
        super().__init__(position, material)
        self.sizes = sizes
        self.type = "AABB"
        #son planos con limites
        self.planes = []

        righPlane = Plane(
                    position=[position[0] + sizes[0]/2, position[1], position[2]], 
                    normal=[1,0,0], 
                    material=material)
        lefthPlane = Plane(
                    position=[position[0] - sizes[0]/2, position[1], position[2]], 
                    normal=[-1,0,0], 
                    material=material)
        
        UpPlane = Plane(
                    position=[position[0], position[1] + sizes[1]/2, position[2]], 
                    normal=[0,1,0], 
                    material=material)
        
        DownPlane = Plane(
                    position=[position[0], position[1] - sizes[1]/2, position[2]], 
                    normal=[0,-1,0], 
                    material=material)
        
        frontPlane = Plane(
                    position=[position[0], position[1] , position[2] + sizes[2]/2], 
                    normal=[0,0,1], 
                    material=material)
        
        backPlane = Plane(
                    position=[position[0], position[1] , position[2] + sizes[2]/2], 
                    normal=[0,0,-1], 
                    material=material)
        
        self.planes.append(righPlane)
        self.planes.append(lefthPlane)
        self.planes.append(UpPlane)
        self.planes.append(DownPlane)
        self.planes.append(frontPlane)
        self.planes.append(backPlane)

        #bounding box es una caja con limites

        #Bounds
        self.minBonds = [0,0,0]
        self.maxBounds = [0,0,0]

        #es un margen de error se podria decir
        epsilon = 0.001

        for i in range(3):
            self.minBonds[i] = position[i] - (epsilon+sizes[i]/2)
            self.maxBounds[i] = position[i] + (epsilon + sizes[i] / 2)


    def ray_intersect(self, origin, dir):
        intercept = None
        t = float("inf")
        for plane in self.planes:
            planeIntercept = plane.ray_intersect(origin, dir)

            if planeIntercept is not None:
                #revisar que esta dentro de los limites
                planePoint = planeIntercept.point
                #si cumplo con los limites en x
                if self.minBonds[0] <= planePoint[0] <= self.maxBounds[0]:
                    if self.minBonds[1] <= planePoint[1] <= self.maxBounds[1]:
                        if self.minBonds[2] <= planePoint[2] <= self.maxBounds[2]:
                            if planeIntercept.distance < t:
                                t = planeIntercept.distance
                                intercept = planeIntercept

        if intercept is None:
            return None
        
        u, v = 0,0

        if (abs(intercept.normal[0]> 0 )):
            #mapear las uvs para el eje x usando las coordenads de y y z
            u = (intercept.point[1]-self.minBonds[1])/self.sizes[1]
            v = (intercept.point[2]-self.minBonds[2])/self.sizes[2]

        elif (abs(intercept.normal[1]> 0 )):
            #mapear las uvs para el eje x usando las coordenads de y y z
            u = (intercept.point[0]-self.minBonds[0])/self.sizes[0]
            v = (intercept.point[2]-self.minBonds[2])/self.sizes[2]
        
        elif (abs(intercept.normal[2]> 0 )):
            #mapear las uvs para el eje x usando las coordenads de y y z
            u = (intercept.point[0]-self.minBonds[0])/self.sizes[0]
            v = (intercept.point[1]-self.minBonds[1])/self.sizes[1]
        u = min(0.999, max(0,u))
        v = min(0.999, max(0,v))
        return Intercept(point=intercept.point, 
                         normal=intercept.normal,
                         distance=t,
                         texCords=[u,v],
                         rayDirection=dir,
                         obj=self)
        
class Triangle(Shape):
    def __init__(self, p0, p1, p2, material):
        super().__init__(None, material)
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.type = "Triangle"

        # Calcular la normal del triángulo
        self.normal = np.cross(np.subtract(self.p1, self.p0), np.subtract(self.p2, self.p0))
        self.normal /= np.linalg.norm(self.normal)

    def ray_intersect(self, origin, dir):
        epsilon = 0.000001
        edge1 = np.subtract(self.p1, self.p0)
        edge2 = np.subtract(self.p2, self.p0)
        h = np.cross(dir, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None  # El rayo es paralelo al triángulo

        f = 1.0 / a
        s = np.subtract(origin, self.p0)
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(dir, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > epsilon:  # Intersección
            P = np.add(origin, np.multiply(dir, t))

            return Intercept(
                point=P,
                normal=self.normal,
                distance=t,
                texCords=None,
                rayDirection=dir,
                obj=self
            )

        return None

class Pyramid(Shape):
    def __init__(self, position, base_size, height, material, rotation_angle=0):
        super().__init__(position, material)
        self.base_size = base_size
        self.height = height
        self.rotation_angle = rotation_angle  # Ángulo de rotación en radianes
        
        half_base = base_size / 2
        # Vértices de la base
        self.base_vertices = [
            [-half_base, 0, -half_base],  # Vértice 1
            [half_base, 0, -half_base],   # Vértice 2
            [half_base, 0, half_base],    # Vértice 3
            [-half_base, 0, half_base]    # Vértice 4
        ]
        # Vértice superior (pico)
        self.top_vertex = [0, height, 0]

        # Aplica la rotación sobre el eje Y
        self.base_vertices = [self.rotate_y(v, self.rotation_angle) for v in self.base_vertices]
        
        # Ajusta los vértices con la posición de la pirámide en el mundo
        self.base_vertices = [np.add(v, position) for v in self.base_vertices]
        self.top_vertex = np.add(self.top_vertex, position)
        
    def rotate_y(self, vertex, angle):
        """Rota el vértice alrededor del eje Y."""
        x, y, z = vertex
        x_rot = x * np.cos(angle) - z * np.sin(angle)
        z_rot = x * np.sin(angle) + z * np.cos(angle)
        return [x_rot, y, z_rot]

    def ray_intersect(self, origin, dir):
        # Intersección con las 4 caras triangulares de la pirámide
        triangles = [
            (self.base_vertices[0], self.base_vertices[1], self.top_vertex),  # Cara 1
            (self.base_vertices[1], self.base_vertices[2], self.top_vertex),  # Cara 2
            (self.base_vertices[2], self.base_vertices[3], self.top_vertex),  # Cara 3
            (self.base_vertices[3], self.base_vertices[0], self.top_vertex)   # Cara 4
        ]
        
        t_min = float('inf')
        intercept = None
        
        # Comprobamos la intersección con cada cara triangular
        for triangle in triangles:
            result = self.ray_triangle_intersect(origin, dir, triangle)
            if result and result.distance < t_min:
                t_min = result.distance
                intercept = result
        
        # Si no hay intersección con las caras triangulares, chequeamos la base
        base_intercept = self.ray_plane_intersect(origin, dir, self.base_vertices)
        if base_intercept and base_intercept.distance < t_min:
            t_min = base_intercept.distance
            intercept = base_intercept
        
        return intercept

    def ray_plane_intersect(self, origin, dir, vertices):
        """Calcula la intersección de un rayo con la base de la pirámide (cuadrada)."""
        # Usa los tres primeros vértices de la base para definir el plano
        normal = np.cross(np.subtract(vertices[1], vertices[0]), np.subtract(vertices[2], vertices[0]))
        normal = normal / np.linalg.norm(normal)

        denom = np.dot(dir, normal)
        if isclose(denom, 0):  # El rayo es paralelo a la base
            return None

        d = np.dot(normal, vertices[0])
        t = (d - np.dot(normal, origin)) / denom

        if t < 0:
            return None

        P = np.add(origin, np.multiply(dir, t))
        
        # Verificamos si el punto de intersección está dentro de la base
        u, v = self.point_in_quad(P, vertices)
        if u is not None and v is not None:
            return Intercept(point=P, normal=normal, distance=t, texCords=[u, v], obj=self, rayDirection=dir)
        
        return None

    def ray_triangle_intersect(self, origin, dir, triangle):
        """Calcula la intersección de un rayo con un triángulo."""
        v0, v1, v2 = triangle
        # Calcula la normal del triángulo
        edge1 = np.subtract(v1, v0)
        edge2 = np.subtract(v2, v0)
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)

        denom = np.dot(normal, dir)
        if isclose(denom, 0):  # El rayo es paralelo al triángulo
            return None

        d = np.dot(normal, v0)
        t = (d - np.dot(normal, origin)) / denom
        if t < 0:
            return None

        P = np.add(origin, np.multiply(dir, t))

        # Verificamos si el punto de intersección está dentro del triángulo
        if self.point_in_triangle(P, v0, v1, v2):
            return Intercept(point=P, normal=normal, distance=t, texCords=None, obj=self, rayDirection=dir)

        return None

    def point_in_triangle(self, P, A, B, C):
        """Verifica si un punto P está dentro del triángulo ABC usando coordenadas baricéntricas."""
        v0 = np.subtract(C, A)
        v1 = np.subtract(B, A)
        v2 = np.subtract(P, A)

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) and (v >= 0) and (u + v < 1)

    def point_in_quad(self, P, vertices):
        """Verifica si un punto P está dentro de un cuadrado."""
        # Divide el cuadrado en dos triángulos
        triangle1 = (vertices[0], vertices[1], vertices[2])
        triangle2 = (vertices[2], vertices[3], vertices[0])

        # Usa la misma lógica de punto en triángulo
        if self.point_in_triangle(P, *triangle1) or self.point_in_triangle(P, *triangle2):
            return 0, 0  # Aquí puedes ajustar las coordenadas UV si es necesario

        return None, None
