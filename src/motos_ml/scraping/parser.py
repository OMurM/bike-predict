import logging
from typing import Optional

from motos_ml.dto import MotoDTO

logger = logging.getLogger(__name__)

# Mapeamos los tipos de la API al vocabulario normalizado del DTO
TIPO_MAP = {
    "NAKED": "naked",
    "TRAIL": "trail",
    "SCOOTER": "scooter",
    "SPORT": "deportiva",
    "SUPERMOTARD": "supermotard",
    "CUSTOM": "custom",
    "TOURING": "touring",
    "THREE_WHEELER": "tres_ruedas",
    "MAXI_SCOOTER": "maxi_scooter",
    "CLASSIC": "classic",
    "OFF_ROAD": "enduro",
}


def parse_motorbike(raw: dict) -> Optional[MotoDTO]:
    """
    Convierte un dict crudo de la API de mundimoto en un MotoDTO.
    Devuelve None si faltan campos obligatorios o el DTO no pasa is_valid().
    """
    try:
        tipo_raw = raw.get("motorbike_type", "")
        tipo = TIPO_MAP.get(tipo_raw, "otro")

        moto = MotoDTO(
            marca=raw["brand"],
            modelo=raw["model"],
            anio=int(raw["year"]),
            km=int(raw["kms"]),
            tipo=tipo,
            precio=float(raw["price"]),
            cilindrada_cc=int(raw["cc"]) if raw.get("cc") else None,
            potencia_cv=None,
            ubicacion=None,
            url_anuncio=f"https://mundimoto.com/es/moto/{raw['id']}?flow=SALE",
            descripcion=None,
            distintivo_ambiental=raw.get("emission_type"),
            num_plazas=int(raw["num_seats"]) if raw.get("num_seats") else None,
            num_llaves=int(raw["num_keys"]) if raw.get("num_keys") else None,
            iva_deducible=bool(raw.get("deductible_vat")) if raw.get("deductible_vat") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Error parseando moto id=%s: %s", raw.get("id", "?"), e)
        return None

    if not moto.is_valid():
        logger.debug("MotoDTO inválido: %s %s", raw.get("brand"), raw.get("model"))
        return None

    return moto


def parse_moto_ocasion(raw: dict) -> Optional[MotoDTO]:
    try:
        url = raw.get("url", "")
        parts = url.strip("/").split("/")
        slug = parts[-1] if parts else ""
        marca = slug.split("-")[0].upper() if "-" in slug else "DESCONOCIDA"

        # Extraer CC de string ej: "471 CC"
        cc_str = raw.get("cilindrada", "").upper().replace("CC", "").strip()
        cc = int(cc_str) if cc_str.isdigit() else None
        
        # Extraer CV de string ej: "47 CV"
        cv_str = raw.get("potencia máxima", "").upper().replace("CV", "").strip()
        cv = int(cv_str) if cv_str.isdigit() else None

        tipo_raw = raw.get("tipo_moto_url", "").upper()
        tipo = TIPO_MAP.get(tipo_raw, "otro")

        # Algunas fichas tienen "año" o "ao" (por codificación HTML en BeautifulSoup, aunque forzamos utf-8)
        anio_str = raw.get("año", raw.get("ao", raw.get("ano", "0")))
        
        moto = MotoDTO(
            marca=marca,
            modelo=raw.get("modelo", "Desconocido"),
            anio=int(anio_str),
            km=int(raw.get("kilómetros", raw.get("kilmetros", "0"))),
            tipo=tipo,
            precio=float(raw.get("precio", 0.0)),
            cilindrada_cc=cc,
            potencia_cv=cv,
            ubicacion=None,
            url_anuncio=url,
            descripcion=None,
            origen="moto-ocasion"
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Error parseando moto ocasion url=%s: %s", raw.get("url", "?"), e)
        return None

    if not moto.is_valid():
        logger.debug("MotoDTO inválido moto-ocasion: %s", raw.get("url"))
        return None

    return moto


def parse_all(raw_list: list[dict]) -> list[MotoDTO]:
    """
    Parsea una lista de dicts crudos. Descarta silenciosamente los inválidos.
    """
    result = []
    for raw in raw_list:
        moto = parse_motorbike(raw)
        if moto:
            result.append(moto)

    discarded = len(raw_list) - len(result)
    if discarded > 0:
        logger.warning("%d motos descartadas por parseo inválido.", discarded)

    return result