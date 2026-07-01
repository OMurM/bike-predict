from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class MotoDTO:
    marca: str
    modelo: str
    anio: int
    km: int
    tipo: str
    precio: float
    cilindrada_cc: Optional[int] = None
    potencia_cv: Optional[int] = None
    ubicacion: Optional[str] = None
    url_anuncio: Optional[str] = None
    descripcion: Optional[str] = None
    origen: str = "mundimoto"  # mundimoto o moto-ocasion
    distintivo_ambiental: Optional[str] = None
    num_plazas: Optional[int] = None
    num_llaves: Optional[int] = None
    iva_deducible: Optional[bool] = None
    ingestion_date: Optional[date] = field(default_factory=date.today)

    def is_valid(self) -> bool:
        return (
            bool(self.marca)
            and bool(self.modelo)
            and self.anio > 1980
            and self.km >= 0
            and self.precio > 0
        )