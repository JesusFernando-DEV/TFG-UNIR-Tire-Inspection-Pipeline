import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# ============================================================
# Funciones auxiliares que apoyan al bloque principal
# ============================================================

def draw_lines(img, lines, color=(0, 0, 255), thickness=2, limits=2000):
    """
    Esta función dibuja sobre la imagen las líneas detectadas
    por el algoritmo de Hough clásico.
    """
    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + limits * (-b)), int(y0 + limits * a))
            pt2 = (int(x0 - limits * (-b)), int(y0 - limits * a))

            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def clean_lines(lines, rho_tolerance, theta_tolerance):
    """
    Esta función elimina líneas similares entre sí dentro del conjunto
    detectado por HoughLines, agrupando aquellas cuyos valores de rho
    y theta sean muy próximos.
    """

    def normalize_rho_theta(rho, theta):
        if rho < 0:
            rho = -rho
            theta = theta + np.pi
        return rho, theta

    def angular_diff(theta1, theta2):
        d = abs(theta1 - theta2)
        return min(d, 2 * np.pi - d)

    def eq_lines(line_1, line_2, rho_tol, theta_tol):
        rho_1, theta_1 = line_1[0]
        rho_2, theta_2 = line_2[0]

        rho_1, theta_1 = normalize_rho_theta(rho_1, theta_1)
        rho_2, theta_2 = normalize_rho_theta(rho_2, theta_2)

        return (abs(rho_1 - rho_2) < rho_tol) and (angular_diff(theta_1, theta_2) < theta_tol)

    cleaned_lines = []

    if lines is not None and len(lines) > 0:
        cleaned_lines.append(lines[0])

        for current_line in lines[1:]:
            equal = False
            for clean_line in cleaned_lines:
                if eq_lines(current_line, clean_line, rho_tolerance, theta_tolerance):
                    equal = True
                    break

            if not equal:
                cleaned_lines.append(current_line)

    return cleaned_lines


def clasificar_lineas_flanco(cleaned_lines):
    """
    Esta función clasifica las líneas detectadas y simplificadas
    en dos grupos aproximados: horizontales y verticales.
    """
    horizontales = []
    verticales = []

    tol_deg = 20

    for line in cleaned_lines:
        rho, theta = line[0]
        theta_deg = np.degrees(theta)

        if abs(theta_deg - 90) <= tol_deg:
            horizontales.append(line)
        elif theta_deg <= tol_deg or abs(theta_deg - 180) <= tol_deg:
            verticales.append(line)

    print(f"Rectas horizontales detectadas: {len(horizontales)}")
    print(f"Rectas verticales detectadas: {len(verticales)}")

    return horizontales, verticales


def obtener_bbox_hough(img_shape, horizontales, verticales):
    """
    Esta función intenta estimar una región de interés rectangular
    a partir de las líneas horizontales y verticales detectadas.
    """
    h, w = img_shape[:2]

    xs = []
    ys = []

    for line in verticales:
        rho, theta = line[0]
        a = np.cos(theta)
        if abs(a) > 1e-6:
            x = rho / a
            xs.append(int(x))

    for line in horizontales:
        rho, theta = line[0]
        b = np.sin(theta)
        if abs(b) > 1e-6:
            y = rho / b
            ys.append(int(y))

    xs_valid = [x for x in xs if 0 <= x < w]
    ys_valid = [y for y in ys if 0 <= y < h]

    if len(xs_valid) >= 2 and len(ys_valid) >= 2:
        x1 = max(0, min(xs_valid))
        x2 = min(w - 1, max(xs_valid))
        y1 = max(0, min(ys_valid))
        y2 = min(h - 1, max(ys_valid))

        if (x2 - x1) > 40 and (y2 - y1) > 20:
            return (x1, y1, x2, y2)

    return None


def obtener_roi_respaldo(img):
    """
    Esta función devuelve una región de interés de respaldo
    cuando Hough no delimita adecuadamente la zona del DOT.
    """
    h, w = img.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    y1 = int(h * 0.35)
    y2 = int(h * 0.75)

    return (x1, y1, x2, y2)


def preprocesar_roi_para_ocr(roi_bgr):
    """
    Esta función prepara la región de interés para mejorar el OCR.
    """
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    roi_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(roi_gray)

    roi_blur = cv2.GaussianBlur(roi_clahe, (5, 5), 0)

    roi_bin = cv2.adaptiveThreshold(
        roi_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    roi_bin_inv = cv2.bitwise_not(roi_bin)

    kernel = np.ones((2, 2), np.uint8)
    roi_morph = cv2.morphologyEx(roi_bin_inv, cv2.MORPH_CLOSE, kernel)

    return roi_gray, roi_clahe, roi_bin, roi_morph


def estimar_angulo_deskew(img_binaria):
    """
    Esta función estima el ángulo de inclinación principal del texto
    a partir de los píxeles activos de la imagen binaria.
    """
    coords = np.column_stack(np.where(img_binaria > 0))

    if len(coords) < 20:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    return angle


def aplicar_deskew(img, angle):
    """
    Esta función corrige la inclinación de una imagen a partir
    del ángulo estimado previamente.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def extraer_mejor_candidato_texto(roi_morph, roi_bgr):
    """
    Esta función localiza contornos candidatos a contener texto,
    filtrando regiones demasiado pequeñas o incompatibles con
    caracteres del código DOT.
    """
    contours, _ = cv2.findContours(roi_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    vis = roi_bgr.copy()

    h_roi, w_roi = roi_morph.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)

        if area < 80:
            continue

        if h < 12 or w < 8:
            continue

        if h > int(h_roi * 0.8) or w > int(w_roi * 0.95):
            continue

        if aspect_ratio < 0.2 or aspect_ratio > 12:
            continue

        candidatos.append((x, y, w, h))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(candidatos) == 0:
        return roi_bgr.copy(), vis

    candidatos = sorted(candidatos, key=lambda r: (r[1], r[0]))

    x_min = min([c[0] for c in candidatos])
    y_min = min([c[1] for c in candidatos])
    x_max = max([c[0] + c[2] for c in candidatos])
    y_max = max([c[1] + c[3] for c in candidatos])

    margen_x = 8
    margen_y = 8

    x1 = max(0, x_min - margen_x)
    y1 = max(0, y_min - margen_y)
    x2 = min(roi_bgr.shape[1], x_max + margen_x)
    y2 = min(roi_bgr.shape[0], y_max + margen_y)

    mejor_roi = roi_bgr[y1:y2, x1:x2].copy()

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return mejor_roi, vis

def preparar_variantes_ocr(roi_bgr):
    """
    Genera varias versiones de la ROI para mejorar la robustez del OCR.
    """
    variantes = []

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Reescalado para dar más tamaño a los caracteres
    gray_big = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # Filtro que preserva bordes mejor que un blur gaussiano simple
    bilateral = cv2.bilateralFilter(gray_big, 9, 75, 75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(bilateral)

    # Otsu binaria normal
    _, otsu_bin = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu binaria invertida
    _, otsu_inv = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptativa normal
    adap_bin = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8
    )

    # Adaptativa invertida
    adap_inv = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 8
    )

    kernel = np.ones((2, 2), np.uint8)

    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernel)
    otsu_inv = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel)
    adap_bin = cv2.morphologyEx(adap_bin, cv2.MORPH_CLOSE, kernel)
    adap_inv = cv2.morphologyEx(adap_inv, cv2.MORPH_CLOSE, kernel)

    variantes.append(("gray_big", gray_big))
    variantes.append(("clahe", clahe))
    variantes.append(("otsu_bin", otsu_bin))
    variantes.append(("otsu_inv", otsu_inv))
    variantes.append(("adap_bin", adap_bin))
    variantes.append(("adap_inv", adap_inv))

    return variantes


def validar_dot(cadena):
    """
    Devuelve un DOT válido de 4 dígitos si encuentra alguno
    compatible con semana/año.
    """
    coincidencias = re.findall(r'\d{4}', cadena)

    coincidencias_validas = []
    for c in coincidencias:
        semana = int(c[:2])
        if 1 <= semana <= 53:
            coincidencias_validas.append(c)

    if len(coincidencias_validas) > 0:
        return coincidencias_validas[-1]

    return None


def extraer_texto_dot_multiple(roi_bgr):
    """
    Prueba varias estrategias de OCR sobre distintas versiones
    de la misma ROI y devuelve la mejor coincidencia encontrada.
    """
    variantes = preparar_variantes_ocr(roi_bgr)

    h, w = roi_bgr.shape[:2]

    x_inicio = int(w * 0.60)
    roi_derecha = roi_bgr[:, x_inicio:w]
    roi_derecha = cv2.resize(roi_derecha, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    roi_derecha_gray = cv2.cvtColor(roi_derecha, cv2.COLOR_BGR2GRAY)

    roi_derecha_bin_inv = cv2.adaptiveThreshold(
        roi_derecha_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 11
    )

    roi_derecha_bin = cv2.adaptiveThreshold(
        roi_derecha_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    variantes_derecha = [
        ("derecha_gray", roi_derecha_gray),
        ("derecha_bin", roi_derecha_bin),
        ("derecha_bin_inv", roi_derecha_bin_inv),
    ]

    configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=DOT0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=DOT0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=DOT0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    ]

    mejor_texto = ""
    mejor_dot = None
    mejor_variante = None

    for nombre, img in variantes:
        img_borde = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        for cfg in configs:
            texto = pytesseract.image_to_string(img_borde, config=cfg)
            texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
            dot = validar_dot(texto_limpio)

            if dot is not None:
                return texto_limpio, dot, nombre

            if len(texto_limpio) > len(mejor_texto):
                mejor_texto = texto_limpio
                mejor_variante = nombre
                mejor_dot = dot

    for nombre, img in variantes_derecha:
        img_borde = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        h2, w2 = img_borde.shape[:2]

        subrois_candidatas = [
            ("r2b", int(w2 * 0.48), int(h2 * 0.28), w2, int(h2 * 0.85)),
            ("r2c", int(w2 * 0.50), int(h2 * 0.22), w2, int(h2 * 0.82)),
            ("r2", int(w2 * 0.55), int(h2 * 0.35), w2, int(h2 * 0.80)),
            ("r1", int(w2 * 0.55), int(h2 * 0.20), w2, int(h2 * 0.65)),
            ("r3", int(w2 * 0.45), int(h2 * 0.30), w2, int(h2 * 0.78)),
            ("r4", int(w2 * 0.60), int(h2 * 0.15), w2, int(h2 * 0.60))
        ]

        for subroi_nombre, x1, y1, x2, y2 in subrois_candidatas:
            img_0222_base = img_borde[y1:y2, x1:x2].copy()
            img_0222_proc = cv2.resize(img_0222_base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(rf"C:\Users\Chus\Downloads\TFE\debug_{subroi_nombre}_base.png", img_0222_base)
            cv2.imwrite(rf"C:\Users\Chus\Downloads\TFE\debug_{subroi_nombre}_proc.png", img_0222_proc)

            print(f"[SUBROI] {subroi_nombre} base shape={img_0222_base.shape} proc shape={img_0222_proc.shape}")

            for psm in [11, 8, 7]:
                data = pytesseract.image_to_data(
                    img_0222_proc,
                    config=f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789',
                    output_type=Output.DICT
                )

                candidatos = []

                for i in range(len(data["text"])):
                    txt = data["text"][i].strip()
                    conf = data["conf"][i]

                    try:
                        conf = float(conf)
                    except:
                        conf = -1

                    txt_limpio = re.sub(r'[^0-9]', '', txt)

                    left = data["left"][i]
                    top = data["top"][i]
                    width = data["width"][i]
                    height = data["height"][i]

                    alto_ocr, ancho_ocr = img_0222_proc.shape[:2]
                    
                    aspecto = width / float(height) if height > 0 else 0

                    if (
                        txt_limpio
                        and left > ancho_ocr * 0.05
                        and 12 <= width <= ancho_ocr * 0.18
                        and 30 <= height <= alto_ocr * 0.35
                        and 0.08 <= aspecto <= 1.2
                        and conf >= 20
                    ):
                        candidatos.append((
                            left,
                            txt_limpio,
                            conf,
                            top,
                            width,
                            height
                        ))

                if not candidatos:
                    texto_fallback = pytesseract.image_to_string(
                        img_0222_proc,
                        config=f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
                    )
                    texto_fallback = re.sub(r'[^0-9]', '', texto_fallback)

                    print(f"[OCR derecha string] variante={nombre} subroi={subroi_nombre} psm={psm} texto={texto_fallback}")

                    if len(texto_fallback) == 4 and texto_fallback.isdigit():
                        return texto_fallback, validar_dot(texto_fallback), f"derecha_string_{subroi_nombre}_psm{psm}_" + nombre
                    
                    if len(texto_fallback) >= 2 and len(texto_fallback) > len(mejor_texto):
                        mejor_texto = texto_fallback
                        mejor_variante = f"derecha_string_{subroi_nombre}_psm{psm}_" + nombre
                        mejor_dot = validar_dot(texto_fallback)

                print(f"[OCR derecha data] variante={nombre} subroi={subroi_nombre} psm={psm} candidatos={candidatos}")

                if candidatos:
                    candidatos = sorted(candidatos, key=lambda x: x[0])
                    texto_limpio = ''.join([c[1] for c in candidatos])
                    dot = validar_dot(texto_limpio)

                    if dot is not None:
                        return texto_limpio, dot, f"derecha_data_{subroi_nombre}_psm{psm}_" + nombre

                    if len(texto_limpio) == 4 and texto_limpio.isdigit():
                        return texto_limpio, validar_dot(texto_limpio), f"derecha_data_{subroi_nombre}_psm{psm}_" + nombre
                    
                    if len(texto_limpio) >= 2 and len(texto_limpio) > len(mejor_texto):
                        mejor_texto = texto_limpio
                        mejor_variante = f"derecha_data_{subroi_nombre}_psm{psm}_" + nombre
                        mejor_dot = validar_dot(texto_limpio)

    return mejor_texto, mejor_dot, mejor_variante

def evaluar_caducidad_dot(dot):
    """
    Esta función interpreta los 4 dígitos finales del DOT
    y evalúa una antigüedad aproximada.
    """
    if dot is None:
        return "No se ha podido determinar", None

    semana = int(dot[:2])
    anio = int(dot[2:])
    anio_completo = 2000 + anio

    fecha_actual = datetime.now()
    anio_actual = fecha_actual.year
    semana_actual = fecha_actual.isocalendar()[1]

    antiguedad = (anio_actual - anio_completo) + ((semana_actual - semana) / 52.0)

    if antiguedad > 5:
        estado = "Neumático potencialmente caducado o envejecido"
    else:
        estado = "Neumático dentro del intervalo temporal definido"

    return estado, antiguedad


def escribir_resultado_final(img_final, dot, estado):
    """
    Esta función escribe sobre la imagen final el resultado
    de la detección DOT.
    """
    if dot is not None:
        texto1 = f"DOT detectado: {dot}"
        texto2 = f"Estado: {estado}"
        color = (0, 255, 0)
    else:
        texto1 = "DOT detectado: NO"
        texto2 = "Estado: No evaluable"
        color = (0, 0, 255)

    cv2.putText(img_final, texto1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img_final, texto2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def redimensionar_si_muy_grande(img_color, max_lado=2500):
    h, w = img_color.shape[:2]
    lado_mayor = max(h, w)

    if lado_mayor <= max_lado:
        return img_color

    escala = max_lado / lado_mayor
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)

    img_redim = cv2.resize(img_color, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    return img_redim

def ejecutar_pipeline_dot(image_path):
    """
    Esta función ejecuta el pipeline completo de lectura del código DOT
    sobre la ruta de imagen que se le facilita.
    """
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_color is None:
        raise ValueError("No se pudo cargar la imagen")

    print("Tamaño original:", img_color.shape[:2])

    img_color = redimensionar_si_muy_grande(img_color, max_lado=2500)

    print("Tamaño ajustado:", img_color.shape[:2])

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(img_gray)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 1.0)
    img_canny = cv2.Canny(img_blur, 50, 150)

    lines = cv2.HoughLines(img_canny, 1, np.pi / 180, 110)

    img_lines = img_color.copy()
    draw_lines(img_lines, lines, color=(0, 0, 255), thickness=2)

    cleaned_lines = clean_lines(lines, rho_tolerance=45, theta_tolerance=12 * np.pi / 180)
    horizontales, verticales = clasificar_lineas_flanco(cleaned_lines)

    img_classified = img_color.copy()

    for line in horizontales:
        draw_lines(img_classified, [line], color=(0, 0, 255), thickness=2)

    for line in verticales:
        draw_lines(img_classified, [line], color=(255, 0, 0), thickness=2)

    bbox = obtener_bbox_hough(img_color.shape, horizontales, verticales)

    if bbox is None:
        bbox = obtener_roi_respaldo(img_color)

    x1, y1, x2, y2 = bbox
    roi_bgr = img_color[y1:y2, x1:x2].copy()

    img_roi = img_color.copy()
    cv2.rectangle(img_roi, (x1, y1), (x2, y2), (0, 255, 255), 2)

    roi_gray, roi_clahe, roi_bin, roi_morph = preprocesar_roi_para_ocr(roi_bgr)

    angle = estimar_angulo_deskew(roi_morph)
    roi_deskew = aplicar_deskew(roi_bgr, angle)

    roi_gray_2, roi_clahe_2, roi_bin_2, roi_morph_2 = preprocesar_roi_para_ocr(roi_deskew)

    roi_texto, roi_contornos = extraer_mejor_candidato_texto(roi_morph_2, roi_deskew)

    texto_limpio, dot_detectado, variante_ocr = extraer_texto_dot_multiple(roi_texto)

    estado_dot, antiguedad = evaluar_caducidad_dot(dot_detectado)

    img_final = img_roi.copy()
    escribir_resultado_final(img_final, dot_detectado, estado_dot)

    resultado = {
        "img_original": img_color,
        "img_gray": img_gray,
        "img_clahe": img_clahe,
        "img_blur": img_blur,
        "img_canny": img_canny,
        "img_lines": img_lines,
        "img_classified": img_classified,
        "img_roi": img_roi,
        "roi_gray": roi_gray,
        "roi_clahe": roi_clahe,
        "roi_bin": roi_bin,
        "roi_morph": roi_morph,
        "roi_deskew": roi_deskew,
        "roi_contornos": roi_contornos,
        "roi_texto": roi_texto,
        #"roi_texto_morph": roi_texto_morph,
        "img_final": img_final,
        "dot_detectado": dot_detectado,
        "texto_limpio": texto_limpio,
        "estado_dot": estado_dot,
        "antiguedad": antiguedad,
        "angulo_deskew": angle,
        "variante_ocr": variante_ocr
    }

    return resultado


# ============================================================
# Interfaz gráfica simple con Tkinter
# ============================================================

class AplicacionDOT:
    def __init__(self, root):
        self.root = root
        self.root.title("Inspección de neumáticos - Lectura código DOT")
        self.root.geometry("1100x750")

        self.label_titulo = tk.Label(
            root,
            text="Pipeline DOT - rama:feature-DOT-funcion",
            font=("Arial", 16, "bold")
        )
        self.label_titulo.pack(pady=10)

        self.frame_botones = tk.Frame(root)
        self.frame_botones.pack(pady=10)

        self.btn_cargar = tk.Button(
            self.frame_botones,
            text="Cargar imagen",
            command=self.cargar_imagen,
            width=20
        )
        self.btn_cargar.grid(row=0, column=0, padx=10)

        self.btn_salir = tk.Button(
            self.frame_botones,
            text="Salir",
            command=root.quit,
            width=20
        )
        self.btn_salir.grid(row=0, column=1, padx=10)

        self.label_resultado = tk.Label(
            root,
            text="Aquí se mostrará el resultado del análisis DOT",
            font=("Arial", 12),
            justify="left"
        )
        self.label_resultado.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        try:
            resultado = ejecutar_pipeline_dot(file_path)

            img_mostrar = resultado["img_final"].copy()
            img_mostrar = cv2.cvtColor(img_mostrar, cv2.COLOR_BGR2RGB)

            h, w = img_mostrar.shape[:2]
            max_w = 900
            max_h = 450
            scale = min(max_w / w, max_h / h, 1.0)

            new_w = int(w * scale)
            new_h = int(h * scale)

            img_mostrar = cv2.resize(img_mostrar, (new_w, new_h))
            img_pil = Image.fromarray(img_mostrar)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk

            dot = resultado["dot_detectado"]
            texto = resultado["texto_limpio"]
            estado = resultado["estado_dot"]
            angulo = resultado["angulo_deskew"]
            variante = resultado["variante_ocr"]

            if resultado["antiguedad"] is not None:
                antiguedad_txt = f"{resultado['antiguedad']:.2f} años"
            else:
                antiguedad_txt = "No calculable"

            texto_resultado = (
                f"Texto OCR limpio: {texto}\n"
                f"Variante OCR útil: {variante}\n"
                f"DOT detectado: {dot}\n"
                f"Ángulo de deskew aplicado: {angulo:.2f} grados\n"
                f"Antigüedad aproximada: {antiguedad_txt}\n"
                f"Estado final: {estado}"
            )

            self.label_resultado.configure(text=texto_resultado)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"{type(e).__name__}: {e}")


# ============================================================
# Bloque principal de ejecución de la aplicación
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionDOT(root)
    root.mainloop()
