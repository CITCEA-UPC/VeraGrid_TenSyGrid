import time
from concurrent.futures import ProcessPoolExecutor

def potencia(base, exponente):
    print(base, exponente)
    return base ** exponente

if __name__ == "__main__":
    bases = [2, 3, 4, 5]
    exponentes = [2, 3, 2, 3] # 2^2, 3^3, 4^2, 5^3

    inicio = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        # Pasas la función y luego todas las secuencias de argumentos
        resultados = list(executor.map(potencia, bases, exponentes))
    inicio = time.perf_counter()
    print(f"Resultados: {resultados}")
