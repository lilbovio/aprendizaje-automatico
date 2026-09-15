import random

PASSWORD = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1] 
LARGO = len(PASSWORD)
POBLACION = 40
GENERACIONES = 60

def aciertos(candidato):
    iguales = 0
    for i in range(LARGO):
        if candidato[i] == PASSWORD[i]:
            iguales += 1
    return iguales  

def nuevo_candidato():
    return [random.randint(0, 1) for _ in range(LARGO)]

def cruzar(padre, madre, punto):
    return padre[:punto] + madre[punto:]

def mutar(candidato, probabilidad):
    for i in range(LARGO):
        if random.random() < probabilidad:
            candidato[i] = 1 - candidato[i]   

grupo = [nuevo_candidato() for _ in range(POBLACION)]   
generacion = 0

while generacion < GENERACIONES:
    generacion += 1
    tabla = []  

    for c in grupo:
        tabla.append((aciertos(c), c))

    tabla.sort(reverse=True)         
    mejores = [c for puntaje, c in tabla[:10]]   

    if aciertos(mejores[0]) == LARGO:
        print(f"¡Descifrado! Contraseña: {mejores[0]} (generación {generacion})")
        break

    
    hijos = mejores[:]        
    while len(hijos) < POBLACION:
        padre = random.choice(mejores)
        madre = random.choice(mejores)
        punto = random.randint(1, LARGO - 1)
        hijo = cruzar(padre, madre, punto)
        mutar(hijo, 0.1)
        hijos.append(hijo)

    grupo = hijos
else:
    print("Se acabaron las generaciones...")