from pathlib import Path
import argparse
import json

import numpy as np


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODELO = BASE / "modelo_hola_adios.npz"

ETIQUETAS = ("hola", "adios")
PASOS = 24


def extraer_secuencia(archivo):
    documento = json.loads(
        archivo.read_text(encoding="utf-8")
    )

    frames = documento.get("frames", [])

    if len(frames) < 10:
        raise ValueError(
            f"{archivo.name}: menos de 10 imágenes."
        )

    tiempos = []
    vectores = []

    for frame in frames:
        tiempo = float(frame["t"])

        if not np.isfinite(tiempo):
            raise ValueError("Tiempo inválido.")

        manos = []

        for mano in frame.get("hands", []):
            puntos = np.array(
                [[p["x"], p["y"]] for p in mano],
                dtype=np.float32,
            )

            if puntos.shape != (21, 2):
                raise ValueError(
                    f"{archivo.name}: mano incompleta."
                )

            if not np.isfinite(puntos).all():
                raise ValueError("Coordenadas inválidas.")

            manos.append(puntos)

        # Orden espacial consistente para hasta dos manos.
        # No representa identidad izquierda/derecha.
        manos.sort(key=lambda puntos: puntos[0, 0])

        vector = np.zeros((2, 45), dtype=np.float32)

        for indice, puntos in enumerate(manos[:2]):
            muneca = puntos[0]

            # Escala aproximada de la palma.
            escala = max(
                float(np.linalg.norm(
                    puntos[9] - muneca
                )),
                0.02,
            )

            forma = (puntos - muneca) / escala

            vector[indice, :42] = forma.reshape(-1)
            vector[indice, 42:44] = muneca
            vector[indice, 44] = 1.0

        tiempos.append(tiempo)
        vectores.append(vector.reshape(-1))

    tiempos = np.asarray(tiempos)
    vectores = np.asarray(vectores)

    if np.any(np.diff(tiempos) < 0):
        raise ValueError("Los tiempos no están ordenados.")

    # Seleccionar 24 momentos distribuidos en el ejemplo.
    objetivos = np.linspace(
        tiempos[0],
        tiempos[-1],
        PASOS,
    )

    indices = np.abs(
        tiempos[:, None] - objetivos[None, :]
    ).argmin(axis=0)

    return vectores[indices]


def distancia_dtw(a, b):
    """Compara movimientos admitiendo diferencias de velocidad."""
    costos = np.full(
        (len(a) + 1, len(b) + 1),
        np.inf,
    )

    costos[0, 0] = 0.0

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            diferencia = a[i - 1] - b[j - 1]

            costo = float(
                np.sqrt(np.mean(diferencia ** 2))
            )

            costos[i, j] = costo + min(
                costos[i - 1, j],
                costos[i, j - 1],
                costos[i - 1, j - 1],
            )

    return costos[-1, -1] / (len(a) + len(b))


def entrenar():
    secuencias = []
    etiquetas = []
    archivos = []

    for etiqueta in ETIQUETAS:
        carpeta = DATASET / etiqueta
        ejemplos = sorted(carpeta.glob("*.json"))

        if len(ejemplos) < 2:
            raise ValueError(
                f"Necesitas al menos 2 ejemplos de "
                f"'{etiqueta}'. Recomendado: 20–30."
            )

        print(
            f"{etiqueta}: {len(ejemplos)} ejemplos"
        )

        for archivo in ejemplos:
            secuencias.append(
                extraer_secuencia(archivo)
            )
            etiquetas.append(etiqueta)
            archivos.append(
                archivo.relative_to(DATASET).as_posix()
            )

    # Guardamos ejemplos de referencia, sin entrenar
    # una red neuronal.
    np.savez_compressed(
        MODELO,
        secuencias=np.stack(secuencias),
        etiquetas=np.asarray(etiquetas),
        archivos=np.asarray(archivos),
    )

    print(f"\nModelo guardado en: {MODELO}")
    print(
        "Ahora graba ejemplos NUEVOS en la aplicación."
    )
    print(
        "No vuelvas a entrenar antes de probarlos."
    )


def probar():
    if not MODELO.exists():
        raise ValueError(
            "Primero ejecuta: python probar_senas.py entrenar"
        )

    with np.load(MODELO, allow_pickle=False) as modelo:
        referencias = modelo["secuencias"]
        etiquetas = modelo["etiquetas"]
        usados = set(modelo["archivos"].tolist())

    nuevos = []

    for etiqueta in ETIQUETAS:
        for archivo in (DATASET / etiqueta).glob("*.json"):
            ruta = archivo.relative_to(DATASET).as_posix()

            if ruta not in usados:
                nuevos.append(archivo)

    if not nuevos:
        print(
            "No hay ejemplos nuevos. Graba uno en la "
            "aplicación y vuelve a ejecutar la prueba."
        )
        return

    aciertos = 0

    for archivo in sorted(nuevos):
        consulta = extraer_secuencia(archivo)

        distancias = np.array([
            distancia_dtw(consulta, referencia)
            for referencia in referencias
        ])

        # Comparar los tres ejemplos más cercanos
        # de cada categoría, o menos si no hay tres.
        puntuaciones = {}

        for etiqueta in ETIQUETAS:
            valores = np.sort(
                distancias[etiquetas == etiqueta]
            )

            puntuaciones[etiqueta] = float(
                np.mean(valores[:3])
            )

        prediccion = min(
            puntuaciones,
            key=puntuaciones.get,
        )

        real = archivo.parent.name
        correcto = prediccion == real
        aciertos += int(correcto)

        print(f"\nArchivo: {archivo.name}")
        print(f"Etiqueta grabada: {real}")
        print(f"Predicción: {prediccion}")
        print(
            "Resultado:",
            "ACIERTO" if correcto else "FALLO",
        )

        for etiqueta, valor in puntuaciones.items():
            print(
                f"  Distancia a {etiqueta}: {valor:.4f}"
            )

    print(
        f"\nAciertos: {aciertos}/{len(nuevos)} "
        "en ejemplos nuevos."
    )
    print(
        "Las distancias no son porcentajes de confianza."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "accion",
        choices=["entrenar", "probar"],
    )

    argumentos = parser.parse_args()

    try:
        if argumentos.accion == "entrenar":
            entrenar()
        else:
            probar()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()