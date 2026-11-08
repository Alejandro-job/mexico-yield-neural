"""Generate exact, dependency-free SVG documentation for the fixed architecture."""
from pathlib import Path
from html import escape
import shutil

ROOT = Path(__file__).resolve().parents[1]


def canvas(title, subtitle, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img">',
            f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#46617e"/></marker></defs>',
            '<rect width="100%" height="100%" rx="20" fill="#f2f6fb"/>',
            '<g font-family="Arial, sans-serif" fill="#172e4c">',
            f'<text x="34" y="47" font-size="27" font-weight="700">{escape(title)}</text>',
            f'<text x="34" y="78" font-size="16" fill="#526984">{escape(subtitle)}</text>']


def text(x, y, value, size=17, color="#172e4c", weight="400", anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x, y, w, h, color="#fff"):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{color}" stroke="#ccdae8"/>'


def arrow(x1, y1, x2, y2):
    return (f'<path d="M{x1} {y1} L{x2-8} {y2}" stroke="#46617e" stroke-width="2.3" fill="none"/>'
            f'<polygon points="{x2},{y2} {x2-8},{y2-4} {x2-8},{y2+4}" fill="#46617e"/>')


def architecture():
    out = canvas("Red neuronal densa · 1,066 parámetros", "Tasas mensuales de México · Implementación con NumPy, retropropagación y Adam", 570)
    blocks = [(34, 238), (326, 204), (584, 204), (842, 324)]
    for x, width in blocks:
        out.append(box(x, 118, width, 350))
    out += [text(54, 151, "ENTRADA", 14, "#526984", "700"), text(54, 186, "47 variables", 25, weight="700")]
    for i, line in enumerate(["9 tasas actuales", "9 rezagos de un mes", "9 rezagos de tres meses", "9 cambios de un mes", "9 cambios de tres meses", "2 pendientes de la curva"]):
        out.append(text(54, 231 + i * 30, line, 16))
    out.append(text(54, 445, "Imputación y normalización", 14, "#526984"))
    for x, count, rows, matrix, params in [(326, 16, 4, "W1: 47 × 16 · b1: 16", "768 parámetros"),
                                          (584, 8, 2, "W2: 16 × 8 · b2: 8", "136 parámetros")]:
        out += [text(x+20, 151, "CAPA OCULTA", 14, "#526984", "700"),
                text(x+20, 186, f"{count} neuronas", 25, weight="700"),
                text(x+20, 218, "ReLU", 17, "#087b82", "700")]
        for i in range(count):
            cx, cy = x + 42 + (i % 4) * 40, 255 + (i // 4) * 37
            out.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#d9f1f0" stroke="#087b82" stroke-width="1.5"/>')
        out += [text(x+20, 416, matrix, 14), text(x+20, 445, params, 14, "#526984")]
    out += [text(862, 151, "SALIDA LINEAL", 14, "#526984", "700"),
            text(862, 186, "18 cambios futuros", 25, weight="700"),
            text(862, 223, "9 a tres meses", 17, "#087b82", "700"),
            text(862, 302, "9 a doce meses", 17, "#087b82", "700")]
    for y in (250, 329):
        for i in range(9):
            out.append(f'<circle cx="{873 + i * 31}" cy="{y}" r="9" fill="#dfebfd" stroke="#315f9e"/>')
    out += [text(862, 374, "W3: 8 × 18 · b3: 18", 16), text(862, 405, "162 parámetros", 16, "#526984"),
            text(862, 445, "Dos horizontes × nueve vencimientos", 14, "#526984")]
    for start, end in ((272, 326), (530, 584), (788, 842)):
        out.append(arrow(start+8, 285, end-10, 285))
    out += [text(34, 512, "Pronóstico de tasa = tasa del mes de referencia + cambio estimado sin estandarizar", 20, weight="700"),
            text(34, 544, "Orden de tasas: 28d, 91d, 182d, 364d, 3a, 5a, 10a, 20a y 30a. Tasas expresadas como porcentaje anualizado.", 15, "#526984"),
            '</g></svg>']
    return "\n".join(out)


def validation():
    out = canvas("Validación temporal en cada origen o", "El modelo usa datos conocidos en o; sus predicciones se comparan con observaciones posteriores", 670)
    for x, width, color in ((34, 450, "#dff1ed"), (500, 380, "#e3edfb"), (900, 266, "#fcebd5")):
        out.append(box(x, 124, width, 206, color))
    out += [text(54, 158, "1 · CALIBRACIÓN", 18, weight="700"),
            text(54, 192, "Etiquetas observadas hasta v − 1", 18),
            text(54, 228, "Medianas y escalas de esta partición", 17),
            text(54, 261, "Entrena la red que seleccionará épocas", 17),
            text(54, 303, "Nunca usa etiquetas de validación", 16, "#06734b", "700"),
            text(520, 158, "2 · VALIDACIÓN INTERNA", 18, weight="700"),
            text(520, 192, "12 orígenes: de v a o − 12", 18),
            text(520, 228, "Sus objetivos a 12 meses llegan a o", 17),
            text(520, 261, "Selecciona el número de épocas", 17),
            text(520, 303, "Etiquetas ya conocidas en o", 16, "#315f9e", "700"),
            text(920, 158, "FUTURO", 18, weight="700"),
            text(920, 192, "o + 3 y o + 12", 19),
            text(920, 228, "Reservado para medir", 17),
            text(920, 261, "el error del pronóstico", 17),
            text(920, 303, "Fuera del entrenamiento", 16, "#8b5914", "700")]
    out += [arrow(54, 365, 1146, 365), text(54, 393, "Tiempo →", 15, "#526984"),
            text(500, 393, "v = o − 23", 15, "#526984"), text(860, 393, "o", 16, "#172e4c", "700"),
            box(34, 431, 846, 138), box(900, 431, 266, 138),
            text(54, 465, "3 · REAJUSTE EN o", 18, weight="700"),
            text(54, 499, "Red nueva con las épocas elegidas y todos los objetivos conocidos hasta o.", 17),
            text(54, 535, "Se vuelven a estimar las medianas y las escalas con ese entrenamiento.", 17),
            text(920, 465, "4 · EVALUACIÓN", 18, weight="700"),
            text(920, 499, "Mismos pares que", 17), text(920, 532, "persistencia y Ridge", 17),
            arrow(880, 501, 898, 501),
            text(34, 614, "El proceso se repite mes a mes con una ventana expansiva y un historial inicial de 120 meses.", 18, weight="700"),
            text(34, 645, "Los objetivos ausentes no se rellenan. Los rezagos se construyen sólo con el mes de referencia y meses anteriores.", 16, "#526984"),
            '</g></svg>']
    return "\n".join(out)


def main():
    docs = ROOT / "docs"
    assets = ROOT / "hf_model/assets"
    docs.mkdir(exist_ok=True)
    assets.mkdir(exist_ok=True)
    for name, content in [("red_neuronal.svg", architecture()), ("validacion_temporal.svg", validation())]:
        (docs / name).write_text(content, encoding="utf-8")
        shutil.copy2(docs / name, assets / name)
    print("Diagramas generados en docs/ y hf_model/assets/")


if __name__ == "__main__":
    main()
