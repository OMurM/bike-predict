# databricks-motos-ml

Pipeline de datos y modelo de ML para predicción de precios de motos de segunda mano.

## Descripción

Proyecto de práctica con Databricks que implementa:

- **Scraping diario** de anuncios de motos desde una web de ocasión
- **Pipeline Bronze → Silver → Gold** con Delta Lake
- **Modelo de regresión** para predecir el precio a partir de características básicas (marca, tipo, año, km, cilindrada)
- **Orquestación diaria** con Databricks Workflows / Jobs

## Arquitectura
