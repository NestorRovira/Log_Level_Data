# -*- coding: utf-8 -*-
import sys
import os
import time
import logging
import re
from bisect import bisect_left
import pandas as pd

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------
# Conjuntos / estructuras
# ----------------------------
block_set = {
    "DoStatement", "WhileStatement", "SynchronizedStatement",
    "IfStatement", "SwitchStatement", "TryStatement",
    "EnhancedForStatement", "ForStatement", "MethodDeclaration",
    "CatchClause", "Block", "SwitchCase"
}
syntactic_filter_set = {
    "Block", "SimpleName", "SimpleType", "QualifiedName", "ParameterizedType",
    "PrimitiveType", "SingleVariableDeclaration", "ArrayType", "TypeLiteral"
}

# Diccionarios globales (se llenan en tiempo de ejecución)
block_dict = {}                 # location (method path) -> [líneas candidatas]
target_dict = {}                # (class, start, end) -> [tipos AST]
methods_dict = {}               # location (method path) -> primera línea del método
methods_lines = {}              # class -> [líneas de comienzo de métodos]
target_dict_logged = {}
level_dict_logged = {}
message_dict_logged = {}
target_dict_nonlogged = {}

# Índices rápidos
ast_by_class = {}               # class -> [(begin, astType), ...] (ordenado)
ast_begin_by_class = {}         # class -> [begin, ...] (para bisect)
logs_by_class = {}              # class -> [(line, level_id, message, rawlog)]
logline_set = set()             # {(class, line)} para saltar nodos-log O(1)

# ----------------------------
# Regex precompiladas
# ----------------------------
rx_ast_type = re.compile(r'<type>([^<]+)</type>')
rx_ast_method = re.compile(r'<method>([^<]+)</method>')
rx_ast_begin = re.compile(r'<begin>([^<]+)</begin>')
rx_ast_end = re.compile(r'<end>([^<]+)</end>')
rx_ast_name = re.compile(r'<name>(.*?)</name>')

rx_log_callsite = re.compile(r'<callsite>([^<]+)</callsite>')
rx_log_level = re.compile(r'<level>([^<]+)</level>')
rx_log_line = re.compile(r'<line>([^<]+)</line>')
rx_log_const = re.compile(r'<constant>([^<]+)</constant>')

# ----------------------------
# Utilidades
# ----------------------------
_gcn_cache = {}
def get_classname(method):
    """Devuelve pkg.Class.java a partir de un callsite/locación con cache."""
    if method in _gcn_cache:
        return _gcn_cache[method]
    fullpath = method.split('.')
    # ...pkg.Class.method  -> pkg.Class.java
    class_name = fullpath[-3] + '.' + fullpath[-2] + '.java'
    _gcn_cache[method] = class_name
    return class_name

def ensure_exists(path, kind="file"):
    if kind == "file" and not os.path.isfile(path):
        logging.error(f"Fichero no encontrado: {path}")
        return False
    if kind == "dir" and not os.path.isdir(path):
        logging.error(f"Carpeta no encontrada: {path}")
        return False
    return True

def read_logs(filename):
    path = os.path.join('original_logs/', f'logs-{filename}.txt')
    logging.info(f"Leyendo logs desde: {path}")
    if not ensure_exists(path, "file"):
        sys.exit(1)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    logging.info(f"Total líneas de log leídas: {len(lines):,}")
    return lines

def read_AST_file(filename):
    path = os.path.join('AST/', f'AST-{filename}.txt')
    logging.info(f"Leyendo AST desde: {path}")
    if not ensure_exists(path, "file"):
        sys.exit(1)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    logging.info(f"Total líneas AST leídas: {len(lines):,}")
    return lines

# ----------------------------
# Parseadores
# ----------------------------
def parse_ASTlines(ASTlines):
    """Devuelve lista de nodos [astType, location, begin, end, content]."""
    out = []
    t0 = time.time()
    append = out.append
    for idx, astline in enumerate(ASTlines):
        try:
            astType = rx_ast_type.findall(astline)[0]
            location = rx_ast_method.findall(astline)[0]
            begin = rx_ast_begin.findall(astline)[0]
            end = rx_ast_end.findall(astline)[0]
            content = rx_ast_name.findall(astline)[0]
            append([astType, location, begin, end, content])
        except Exception:
            # Saltamos líneas que no encajan
            continue
    logging.info(f"AST parseado: {len(out):,} nodos válidos en {time.time()-t0:.2f}s")
    return out

def level_to_id(level_txt):
    if level_txt == 'trace': return 0
    if level_txt == 'debug': return 1
    if level_txt == 'info':  return 2
    if level_txt == 'warn':  return 3
    if level_txt == 'error': return 4
    return 5  # fatal/otros

def id_to_level(level_id):
    return {0:"trace",1:"debug",2:"info",3:"warn",4:"error",5:"fatal"}.get(level_id,"unknown")

def parse_Loglines(Loglines):
    """Devuelve lista de logs [level, line, content, callsite] y llena índices por clase."""
    out = []
    t0 = time.time()
    for idx, logline in enumerate(Loglines):
        try:
            callsite = rx_log_callsite.findall(logline)[0]
            level = rx_log_level.findall(logline)[0]
            line = int(rx_log_line.findall(logline)[0])
            content_m = rx_log_const.findall(logline)
            content = content_m[0] if content_m else 'No message'
            out.append([level, line, content, callsite])
        except Exception:
            continue

    # Índices por clase (optimiza etiquetado) y set de líneas de log
    for level, line, content, callsite in out:
        cls = get_classname(callsite)
        logs_by_class.setdefault(cls, []).append((line, level_to_id(level), content, logline))
        logline_set.add((cls, line))

    # Asegurar orden por línea dentro de cada clase (para barridos eficientes)
    for cls, lst in logs_by_class.items():
        lst.sort(key=lambda x: x[0])

    logging.info(f"Logs parseados: {len(out):,} entradas válidas en {time.time()-t0:.2f}s")
    return out

# ----------------------------
# Helpers de lógica
# ----------------------------
def not_level_guard(string):
    s = string.lower()
    return not ("enabled" in s and ("info" in s or "debug" in s or "trace" in s))

def get_methods_dict(node):
    """Registra la primera línea del método para cada location."""
    if node[1] in methods_dict:
        if int(methods_dict[node[1]]) > int(node[2]):
            methods_dict[node[1]] = node[2]
    else:
        methods_dict[node[1]] = node[2]

def get_methods_lines(methods_dict):
    for key, value in methods_dict.items():
        class_name = get_classname(key)
        methods_lines.setdefault(class_name, []).append(int(value))
    for key in methods_lines:
        methods_lines[key].sort()

def get_method_start_line_for_AST(class_name, start_line):
    """Devuelve el comienzo del método que contiene start_line, o start_line si no hay info."""
    method_start_line = int(start_line)
    memory_line = 1
    if class_name in methods_lines and methods_lines[class_name]:
        for v in methods_lines[class_name]:
            if int(v) >= int(start_line):
                return int(memory_line)
            else:
                memory_line = int(v)
        return int(memory_line)
    return int(method_start_line)

def build_ast_index(ASTlists):
    """Crea índices por clase para acceso O(log N) + O(k) con bisect."""
    # Filtramos ya los tipos puramente sintácticos para no procesarlos más tarde
    for astType, location, begin, end, content in ASTlists:
        cls = get_classname(location)
        b = int(begin)
        # Guardamos solo lo que podría llegar al 'Values' (evita filtrar luego)
        if astType not in syntactic_filter_set:
            ast_by_class.setdefault(cls, []).append((b, astType, location))
    # Orden por línea de inicio para cada clase y tener lista paralela de begins para bisect
    for cls, lst in ast_by_class.items():
        lst.sort(key=lambda x: x[0])
        ast_begin_by_class[cls] = [x[0] for x in lst]

def label_blocks_fast():
    """Etiqueta bloques usando logs_by_class indexado (rápido)."""
    logging.info("Etiquetando bloques con niveles de log…")
    n_logged, n_non = 0, 0
    t0 = time.time()
    for (key_class, key_start, key_end), value in target_dict.items():
        best_level = -1
        best_msg = '-'
        # Recorremos solo los logs de esa clase
        for line, lvl_id, msg, raw in logs_by_class.get(key_class, []):
            if key_start <= line <= key_end:
                if lvl_id > best_level:
                    best_level, best_msg = lvl_id, msg
        if best_level >= 0:
            target_dict_logged[(key_class, key_start, key_end)] = value
            level_dict_logged[(key_class, key_start, key_end)] = id_to_level(best_level)
            message_dict_logged[(key_class, key_start, key_end)] = best_msg
            n_logged += 1
        else:
            target_dict_nonlogged[(key_class, key_start, key_end)] = value
            n_non += 1
    logging.info(f"Bloques etiquetados. Con log: {n_logged:,} | Sin log: {n_non:,} | Tiempo: {time.time()-t0:.2f}s")

def tuplekey_to_str(key_tuple):
    cls, s, e = key_tuple
    return f"<class>{cls}</class><start>{s}</start><end>{e}</end>"

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error("Uso: python block_processing.py <dataset>\nEjemplo: python block_processing.py cassandra")
        sys.exit(1)

    dataset = sys.argv[1]
    logging.info(f"=== Inicio block_processing (optimizado) para dataset: {dataset} ===")

    # Carpetas base
    os.makedirs('blocks', exist_ok=True)

    # 1) Lectura
    ASTlines = read_AST_file(dataset)
    loglines_raw = read_logs(dataset)

    # 2) Parseo
    ASTlists = parse_ASTlines(ASTlines)
    parse_Loglines(loglines_raw)  # llena logs_by_class y logline_set

    # 3) Catálogo de métodos
    logging.info("Catalogando métodos…")
    for astlist in ASTlists:
        get_methods_dict(astlist)
    get_methods_lines(methods_dict)
    logging.info(f"Métodos catalogados: {len(methods_dict):,} | Clases con métodos: {len(methods_lines):,}")

    # 4) Recolección de límites de bloques (idéntico a versión original)
    logging.info("Recolectando límites de bloques…")
    t0 = time.time()
    n_candidates = 0
    for astType, location, begin, end, content in ASTlists:
        if astType in block_set and not_level_guard(content[:40]):
            n_candidates += 1
            lst = block_dict.setdefault(location, [])
            b = int(begin); e = int(end)
            if b not in lst: lst.append(b)
            if e not in lst: lst.append(e)
    for key in block_dict:
        block_dict[key].sort()
    logging.info(f"Límites de bloques recogidos: {n_candidates:,} | Métodos con límites: {len(block_dict):,} | {time.time()-t0:.2f}s")

    # 5) Construcción de keys (tuplas) (class, start, end) -> []
    logging.info("Construyendo rangos de bloque…")
    t0 = time.time()
    for location, value in block_dict.items():
        cls = get_classname(location)
        for i in range(0, len(value) - 1):
            start = value[i]
            end = value[i+1] - 1
            target_dict[(cls, start, end)] = []
    logging.info(f"Bloques construidos: {len(target_dict):,} | {time.time()-t0:.2f}s")

    # 6) Índice AST por clase (para acelerar población de contenido)
    logging.info("Indexando AST por clase…")
    t0 = time.time()
    build_ast_index(ASTlists)
    logging.info(f"Clases indexadas: {len(ast_by_class):,} | {time.time()-t0:.2f}s")

    # 7) Poblar contenido de cada bloque usando bisect (rápido)
    logging.info("Poblando contenido sintáctico por bloque (optimizado)…")
    t0 = time.time()
    total_blocks = len(target_dict)
    report_every = 100
    filled = 0

    for (cls, start_line, end_line), value in target_dict.items():
        # Ajuste del inicio de método
        m_start_line = get_method_start_line_for_AST(cls, start_line)
        if m_start_line is None or m_start_line == 1:
            m_start_line = start_line

        begins = ast_begin_by_class.get(cls)
        nodes = ast_by_class.get(cls)
        if not begins or not nodes:
            filled += 1
            if filled % report_every == 0 or filled == total_blocks:
                elapsed = time.time() - t0
                speed = filled / elapsed if elapsed > 0 else 0
                eta = (total_blocks - filled) / speed if speed > 0 else 0
                pct = 100 * filled / total_blocks
                logging.info(f"  Bloques: {filled:,}/{total_blocks:,} ({pct:5.1f}%) | Vel: {speed:6.1f} blk/s | ETA: {eta/60:6.1f} min")
            continue

        # Comenzar a partir del primer nodo cuya línea >= m_start_line
        i = bisect_left(begins, int(m_start_line))
        # Avanzar hasta que la línea supere end_line
        end_int = int(end_line)
        add = value.append
        while i < len(nodes):
            b, astType, _loc = nodes[i]
            if b > end_int:
                break
            # Saltar si esa línea es un log real
            if (cls, b) not in logline_set:
                add(astType)
            i += 1

        filled += 1
        if filled % report_every == 0 or filled == total_blocks:
            elapsed = time.time() - t0
            speed = filled / elapsed if elapsed > 0 else 0
            eta = (total_blocks - filled) / speed if speed > 0 else 0
            pct = 100 * filled / total_blocks
            logging.info(f"  Bloques: {filled:,}/{total_blocks:,} ({pct:5.1f}%) | Vel: {speed:6.1f} blk/s | ETA: {eta/60:6.1f} min")

    logging.info(f"Contenido poblado para {filled:,} bloques en {(time.time()-t0)/60:.2f} min")

    # 8) Etiquetado de bloques (rápido con índice)
    label_blocks_fast()

    # 9) Salida CSV (formato original de 'Key')
    header_logged = ['Key', 'Values', 'Level', 'Message']
    rows = []
    for key_tuple, vals in target_dict_logged.items():
        kstr = tuplekey_to_str(key_tuple)
        rows.append([kstr, vals, level_dict_logged[key_tuple], message_dict_logged[key_tuple]])

    out_path = os.path.join('blocks', f'logged_syn_{dataset}.csv')
    pd.DataFrame(rows, columns=header_logged).to_csv(out_path, index=False, encoding='utf-8')
    logging.info(f"CSV escrito: {out_path} ({len(rows):,} filas).")

    logging.info(f"Bloques SIN log (no guardados): {len(target_dict_nonlogged):,}")
    logging.info(f"=== Fin block_processing (optimizado) para dataset: {dataset} ===")
