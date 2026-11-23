"""Spanish long-form passages and Spanish number normalization for the es-longform report.

PASSAGES are four public-domain Spanish texts, 100-180 words each, frozen verbatim from es.wikisource
(the title and revision id are in `source`; `sha256` is over `text`). Long-form means a minute or more
of speech per passage, read chunk by chunk (scoring/longform_chunker.py), so the report measures
chunking, number and ordinal reading, archaic register and dialogue rather than one short prompt.

es_number_words() / spell_digits() spell digit groups in Spanish. They are applied to the Whisper
HYPOTHESIS before WER: Whisper writes "72" for a spoken "setenta y dos", and scoring/wer.py
normalizes numbers for English only. Standard library only.
"""

import re

PASSAGES = [
    {
        'id': 'pd_galdos',
        'source': "Benito Pérez Galdós, Marianela (1878), cap. I. es.wikisource 'Marianela/I' revid 1674905. Accented preposition 'á' normalized to 'a'.",
        'license': 'Public domain (author died 1920)',
        'sha256': 'b867eedb3c0faab8f9d6b382c0860d0f3f235ef1d155b0799e8b61e0a59b5c6d',
        'text': 'Se puso el sol. Tras el breve crepúsculo vino tranquila y oscura la noche, en cuyo negro seno murieron poco a poco los últimos rumores de la tierra soñolienta, y el viajero siguió adelante en su camino, apresurando su paso a medida que avanzaba el de la noche. Iba por angosta vereda, de esas qué sobre el césped traza el constante pisar de hombres y brutos, y subia sin cansancio por un cerro en cuyas vertientes se alzaban pintorescos grupos de guinderos, hayas y robles. (Ya se ve que estamos en el Norte de España.) Era un hombre de mediana edad, de complexion récia, de buena talla, ancho de espaldas, resuelto de ademanes, firme de andadur, basto de facciones, de mirar osado y vivo, ligero a pesar de su regular obesidad, y (dígase de una vez aunque sea prematuro) excelente persona por doquiera que se le mirara.',
    },
    {
        'id': 'pd_becquer',
        'source': "Gustavo Adolfo Bécquer, Los ojos verdes (Obras, 1885), legend body from '—Herido va el ciervo'. es.wikisource 'Los ojos verdes' revid 1010752. 'á' normalized to 'a'.",
        'license': 'Public domain (author died 1870)',
        'sha256': '2df1e7a15256dc1ac0e2180a3e8444c43ba1fbc4a73a82a8952f53a588b89489',
        'text': '—Herido va el ciervo... herido va; no hay duda. Se ve el rastro de la sangre entre las zarzas del monte, y al saltar uno de esos lentiscos han flaqueado sus piernas... Nuestro joven señor comienza por donde otros acaban... en cuarenta años de montero no he visto mejor golpe... ¡Pero por San Saturio, patrón de Soria! cortadle el paso por esas carrascas, azuzad los perros, soplad en esas trompas hasta echar los hígados, y hundidle a los corceles una cuarta de hierro en los ijares: ¿no véis que se dirige hacia la fuente de los Álamos, y si la salva antes de morir podemos darle por perdido? Las cuencas del Moncayo repitieron de eco en eco el bramido de las trompas, el latir de la jauría desencadenada y las voces de los pajes resonaron con nueva furia, y el confuso tropel de hombres, caballos y perros se dirigió al punto que Iñigo, el montero mayor de los marqueses de Almenar, señalara como el más a propósito para cortarle el paso a la res. Pero todo fué inútil.',
    },
    {
        'id': 'pd_cervantes',
        'source': "Miguel de Cervantes, El ingenioso hidalgo Don Quijote de la Mancha, ed. 1905, Tomo I, cap. I. es.wikisource revid 1420662. Drop-cap 'E' restored; 'á'/'ó' normalized to 'a'/'o'.",
        'license': 'Public domain (author died 1616)',
        'sha256': '5c4e32ce6049a7b5747f9a0bee5eb63aa1afeb45500f1014267e183b3f291dfc',
        'text': 'En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mismo, y los días de entre semana se honraba con su vellorí de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera.',
    },
    {
        'id': 'pd_cadiz',
        'source': "Constitución política de la Monarquía española (Cádiz, 1812), Título I, arts. 1-5. es.wikisource 'Constitución española de 1812' revid 1455980. 'Art. N.' written out as 'Artículo <ordinal>.'; the Wikisource '[editar]' link removed and the heading 'CAPÍTULO II' written out as 'Capítulo segundo.'.",
        'license': 'Public domain (1812 government text)',
        'sha256': '0c77267dacf32cad4348eb7782f24a1114448b9f5f0719bf45aac67a10f72a01',
        'text': 'Artículo primero. La Nación española es la reunión de todos los españoles de ambos hemisferios. Artículo segundo. La Nación española es libre e independiente, y no es ni puede ser patrimonio de ninguna familia ni persona. Artículo tercero. La soberanía reside esencialmente en la Nación, y por lo mismo pertenece a ésta exclusivamente el derecho de establecer sus leyes fundamentales. Artículo cuarto. La Nación está obligada a conservar y proteger por leyes sabias y justas la libertad civil, la propiedad y los demás derechos legítimos de todos los individuos que la componen. Capítulo segundo: De los españoles. Artículo quinto. Son españoles: Primero. Todos los hombres libres nacidos y avecindados en los dominios de las Españas, y los hijos de éstos.',
    },
]

PASSAGE_BY_ID = {p["id"]: p for p in PASSAGES}

_UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
          "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
          "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis",
          "veintisiete", "veintiocho", "veintinueve"]
_TENS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta",
         90: "noventa"}
_HUND = {100: "ciento", 200: "doscientos", 300: "trescientos", 400: "cuatrocientos", 500: "quinientos",
         600: "seiscientos", 700: "setecientos", 800: "ochocientos", 900: "novecientos"}
_DIGITS = _UNITS[:10]
MAX_SPELLED = 10 ** 12  # below this, cardinal words ("novecientos mil millones"); at or above, digit by digit


def _below_1000(n):
    if n == 100:
        return "cien"
    if n < 30:
        return _UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TENS[t * 10] + ("" if u == 0 else " y " + _UNITS[u])
    h, r = divmod(n, 100)
    return _HUND[h * 100] + ("" if r == 0 else " " + _below_1000(r))


def _apocope(words):
    if words.endswith("veintiuno"):
        return words[:-len("veintiuno")] + "veintiún"
    if words.endswith("uno"):
        return words[:-3] + "un"
    return words


def _below_million(n):
    if n < 1000:
        return _below_1000(n)
    th, r = divmod(n, 1000)
    head = "mil" if th == 1 else _apocope(_below_1000(th)) + " mil"
    return head + ("" if r == 0 else " " + _below_1000(r))


def es_int_words(n):
    """Spanish cardinal for any n >= 0 (long scale: 10**9 is 'mil millones'). From MAX_SPELLED up, the digits
    are read one by one, so no hypothesis can make the scorer raise."""
    if n < 0:
        raise ValueError(n)
    if n >= MAX_SPELLED:
        return " ".join(_DIGITS[int(d)] for d in str(n))
    if n < 1_000_000:
        return _below_million(n)
    mi, r = divmod(n, 1_000_000)
    head = "un millón" if mi == 1 else _apocope(_below_million(mi)) + " millones"
    return head + ("" if r == 0 else " " + _below_million(r))


def es_number_words(tok):
    m = re.fullmatch(r"(\d+)(?:[.,](\d+))?", tok)
    if not m:
        raise ValueError(tok)
    whole = es_int_words(int(m.group(1)))
    if m.group(2) is None:
        return whole
    frac = m.group(2)
    zeros = len(frac) - len(frac.lstrip("0"))
    rest = frac.lstrip("0")
    parts = ["cero"] * zeros + ([es_int_words(int(rest))] if rest else [])
    return f"{whole} coma {' '.join(parts)}"


_ORD_1_9 = ["", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno"]
_ORD_TENS = ["", "décimo", "vigésimo", "trigésimo", "cuadragésimo", "quincuagésimo", "sexagésimo",
             "septuagésimo", "octogésimo", "nonagésimo"]
_ORD_SPECIAL = {11: "undécimo", 12: "duodécimo", 13: "decimotercero", 14: "decimocuarto", 15: "decimoquinto",
                16: "decimosexto", 17: "decimoséptimo", 18: "decimoctavo", 19: "decimonoveno"}


def es_ordinal_words(n, feminine=False):
    """'4º' -> 'cuarto', '4ª' -> 'cuarta' for 1-99; other values fall back to the cardinal."""
    if not 1 <= n <= 99:
        return es_int_words(n)
    if n in _ORD_SPECIAL:
        w = _ORD_SPECIAL[n]
    elif n < 10:
        w = _ORD_1_9[n]
    else:
        t, u = divmod(n, 10)
        w = _ORD_TENS[t] + ("" if u == 0 else " " + _ORD_1_9[u])
    if feminine:
        w = " ".join(x[:-1] + "a" if x.endswith("o") else x for x in w.split())
    return w


def spell_digits(text):
    """Spell digit groups in Spanish: '72 vergas' -> 'setenta y dos vergas', 'Artículo 4º' -> 'Artículo cuarto'."""
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+,\d+\b", lambda m: m.group(0).replace(".", ""), text)  # 1.200,50
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\.\d+\b", lambda m: m.group(0).replace(",", ""), text)  # 1,200.50
    text = re.sub(r"\d{1,3}(?:[.,]\d{3})+(?![.,]?\d)",
                  lambda m: re.sub(r"[.,]", "", m.group(0)), text)  # 1.000.000 / 1,500 -> 1000000 / 1500
    text = re.sub(r"(\d+)\.?\s*([ºª])",
                  lambda m: " " + es_ordinal_words(int(m.group(1)), m.group(2) == "ª") + " ", text)
    text = re.sub(r"(?<=\d)\s*%", " por ciento", text)
    text = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", " por ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)  # AC12 -> AC 12
    return re.sub(r"\d+(?:[.,]\d+)?", lambda m: es_number_words(m.group(0)), text)


def join_letter_runs(text):
    """'A C' / 'P R' (spaced diagram labels) -> 'AC' / 'PR', as Whisper writes them."""
    return re.sub(r"\b[A-Z](?: [A-Z])+\b", lambda m: m.group(0).replace(" ", ""), text)
