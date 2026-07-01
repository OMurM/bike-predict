import time
import logging
from typing import Optional, List
import requests
from bs4 import BeautifulSoup

from motos_ml.config import ScraperConfig

logger = logging.getLogger(__name__)

BASE_URL = "https://www.moto-ocasion.com/motos-de-ocasion"

def extract_moto_links_from_page(page_num: int, config: ScraperConfig) -> List[str]:
    url = f"{BASE_URL}/page/{page_num}/" if page_num > 1 else f"{BASE_URL}/"
    headers = {"User-Agent": config.user_agent}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 404:
            return []  # Llegamos al final de las páginas
        response.raise_for_status()
        time.sleep(config.request_delay_seconds)
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if href and 'moto-ocasion.com/motos-de-ocasion/' in href and len(href.split('/')) > 5:
                if href not in links:
                    links.append(href)
        return links
    except requests.RequestException as e:
        logger.warning(f"Error fetching moto-ocasion list page={page_num}: {e}")
        return []

def fetch_moto_detail(url: str, config: ScraperConfig) -> Optional[dict]:
    headers = {"User-Agent": config.user_agent}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        time.sleep(config.request_delay_seconds)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Encontrar la tabla de especificaciones
        specs = {}
        for tr in soup.find_all('tr'):
            th = tr.find('th')
            td = tr.find('td')
            if th and td:
                specs[th.text.strip().lower()] = td.text.strip()
        
        # Encontrar el precio
        price_elem = soup.select_one('.price .amount bdi')
        price_str = price_elem.text.strip() if price_elem else "0"
        # Limpiar precio (ej: 6.990,00&nbsp;€ -> 6990.00)
        price_str = price_str.replace('€', '').replace('&nbsp;', '').replace('.', '').replace(',', '.').strip()
        try:
            price = float(price_str)
        except ValueError:
            price = 0.0

        # Para el origen y el url:
        specs['url'] = url
        specs['precio'] = price
        
        # Tipo de moto (se suele poder deducir del breadcrumb o de la url)
        # ej: /motos-de-ocasion/trail/honda... -> trail
        parts = url.split('/')
        tipo = parts[-3] if len(parts) >= 3 else "otro"
        specs['tipo_moto_url'] = tipo

        return specs
    except requests.RequestException as e:
        logger.warning(f"Error fetching moto-ocasion detail {url}: {e}")
        return None

def fetch_all_moto_ocasion(config: ScraperConfig) -> List[dict]:
    all_raw = []
    # Limitar a pocas motos por ahora para no saturar durante el desarrollo o si tarda mucho
    limit = min(500, config.max_pages * 20) # Reducir un poco el límite si max_pages es pequeño
    
    page = 1
    links_to_fetch = []
    
    while len(links_to_fetch) < limit:
        links = extract_moto_links_from_page(page, config)
        if not links:
            break
        links_to_fetch.extend(links)
        page += 1
        
    links_to_fetch = links_to_fetch[:limit]
    
    for i, link in enumerate(links_to_fetch):
        detail = fetch_moto_detail(link, config)
        if detail:
            all_raw.append(detail)
            
        if (i + 1) % 10 == 0:
            logger.info(f"Moto-Ocasion: scraped {i+1}/{len(links_to_fetch)}")

    logger.info(f"Total Moto-Ocasion scrapeadas: {len(all_raw)}")
    return all_raw
