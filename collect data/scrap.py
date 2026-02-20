import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import sys
import re
import json
import logging
import subprocess
from contextlib import contextmanager
from dotenv import load_dotenv
from urllib.parse import urljoin
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


def _setup_logger():
    level_name = (os.getenv('LOG_LEVEL') or 'INFO').strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger('riyasewana_scraper')
    logger.setLevel(level)

    # Avoid duplicate handlers if the file is imported/re-run in interactive contexts
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Use stdout so we can silence stderr for noisy Chromium logs
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    log_file = (os.getenv('LOG_FILE') or '').strip()
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


logger = _setup_logger()


def _env_bool(name: str, default: str = '0') -> bool:
    return (os.getenv(name) or default).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


@contextmanager
def _redirect_stderr_to_devnull(enabled: bool):
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        # If we can't access the fd, do nothing.
        yield
        return

    with open(os.devnull, 'w') as devnull:
        old_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            try:
                os.dup2(old_stderr_fd, stderr_fd)
            finally:
                os.close(old_stderr_fd)

# 1. Define the target URL from environment variable
# URL should be like: https://riyasewana.com
base_url = (os.getenv('URL') or 'https://riyasewana.com').rstrip('/')
# Optional override if you want to scrape a different category/brand.
# Examples: /search/cars , /search/cars/toyota
start_path = os.getenv('START_PATH') or '/search/cars/toyota'
start_url = start_path if start_path.lower().startswith('http') else f'{base_url}{start_path}'

# Optional controls
max_pages_env = os.getenv('MAX_PAGES')
max_pages = int(max_pages_env) if max_pages_env and max_pages_env.isdigit() else None
request_delay = float(os.getenv('REQUEST_DELAY') or '1')

max_items_env = os.getenv('MAX_ITEMS')
max_items = int(max_items_env) if max_items_env and max_items_env.isdigit() else None

# Fetch each ad detail page (more fields like Contact/Make/Model/Details)
scrape_details = (os.getenv('SCRAPE_DETAILS') or '0').strip() == '1'

# Suppress noisy Chromium/Edge logs like:
# [pid:tid:...:ERROR:chrome\browser\task_manager\providers\fallback_task_provider.cc:126] ...
silence_browser_logs = _env_bool('SILENCE_BROWSER_LOGS', '1')

# If the site blocks Python requests (403), use a real browser engine.
# Set USE_SELENIUM=1 in .env to enable.
use_selenium = (os.getenv('USE_SELENIUM') or '0').strip() == '1'
browser = (os.getenv('BROWSER') or 'edge').strip().lower()  # edge | chrome
headless = (os.getenv('HEADLESS') or '1').strip() == '1'

logger.info(
    'Config | start_url=%s | max_pages=%s | max_items=%s | delay=%ss | details=%s | selenium=%s | browser=%s | headless=%s',
    start_url,
    str(max_pages),
    str(max_items),
    str(request_delay),
    str(scrape_details),
    str(use_selenium),
    browser,
    str(headless),
)

# 2. Add a User-Agent header so the website knows we are a standard browser, not a malicious bot
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    # Avoid Brotli here unless you have brotli installed; gzip/deflate are safe.
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': base_url + '/',
}

# Use a session to reuse connections (faster + more reliable)
session = requests.Session()
session.headers.update(headers)

# Warm up the session (some sites set cookies on the home page)
try:
    t0 = time.perf_counter()
    r = session.get(base_url + '/', timeout=30)
    logger.debug('Warm-up GET %s -> %s in %.2fs', r.url, r.status_code, time.perf_counter() - t0)
except requests.RequestException:
    logger.debug('Warm-up request failed (continuing).', exc_info=True)


def create_selenium_driver():
    from selenium import webdriver

    logger.info('Starting Selenium driver (%s, headless=%s)...', browser, headless)

    if browser == 'chrome':
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService

        options = ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1365,900')
        options.add_argument('--log-level=3')
        try:
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
        except Exception:
            pass
        options.add_argument(f'--user-agent={headers["User-Agent"]}')
        # --silent reduces driver/browser startup chatter in some environments
        service = ChromeService(
            log_output=subprocess.DEVNULL,
            service_args=['--silent', '--log-level=OFF'],
        )
        with _redirect_stderr_to_devnull(silence_browser_logs):
            return webdriver.Chrome(service=service, options=options)

    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    options = EdgeOptions()
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1365,900')
    options.add_argument('--log-level=3')
    try:
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
    except Exception:
        pass
    options.add_argument(f'--user-agent={headers["User-Agent"]}')
    service = EdgeService(log_output=subprocess.DEVNULL)
    with _redirect_stderr_to_devnull(silence_browser_logs):
        return webdriver.Edge(service=service, options=options)


def fetch_html_with_driver(driver, page_url: str):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver.get(page_url)
    WebDriverWait(driver, 25).until(
        EC.any_of(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li.item')),
            EC.presence_of_element_located((By.CSS_SELECTOR, '#mbody')),
        )
    )
    return driver.page_source


def parse_listings(page_soup: BeautifulSoup, page_url: str):
    listings = []

    car_listings = page_soup.find_all('li', class_='item')
    for car in car_listings:
        try:
            title_elem = car.find('h2', class_='more')
            a = title_elem.find('a') if title_elem else None

            title = a.get_text(strip=True) if a else None
            link = urljoin(page_url, a['href']) if a and a.get('href') else None

            boxtext = car.find('div', class_='boxtext')
            boxintxt_divs = boxtext.find_all('div', class_='boxintxt') if boxtext else []

            location = boxintxt_divs[0].get_text(strip=True) if len(boxintxt_divs) > 0 else None
            price = boxintxt_divs[1].get_text(strip=True) if len(boxintxt_divs) > 1 else None
            mileage = boxintxt_divs[2].get_text(strip=True) if len(boxintxt_divs) > 2 else None
            date = boxintxt_divs[3].get_text(strip=True) if len(boxintxt_divs) > 3 else None

            if title and price:
                listings.append({
                    'Title': title,
                    'Price': price,
                    'Mileage': mileage,
                    'Location': location,
                    'Date': date,
                    'URL': link,
                })
        except (AttributeError, IndexError, KeyError, TypeError):
            continue

    return listings


def find_next_page(page_soup: BeautifulSoup, page_url: str):
    for pagination in page_soup.select('div.pagination'):
        next_a = pagination.find('a', string=lambda s: s and s.strip().lower() == 'next')
        if next_a and next_a.get('href'):
            return urljoin(page_url, next_a['href'])
    return None


def parse_posted_info(text: str):
    # Example:
    # "Posted by Amarasinghe on 2026-02-12 12:57 pm, Colombo"
    raw = (text or '').strip()
    posted_by = None
    posted_on = None
    posted_time = None
    posted_location = None

    if raw.lower().startswith('posted by '):
        raw2 = raw[len('posted by '):]
        if ' on ' in raw2:
            posted_by, rest = raw2.split(' on ', 1)
            posted_by = posted_by.strip() or None

            if ',' in rest:
                rest, posted_location = rest.rsplit(',', 1)
                posted_location = posted_location.strip() or None

            rest = rest.strip()
            parts = rest.split()
            if parts:
                posted_on = parts[0].strip() or None
                posted_time = ' '.join(parts[1:]).strip() or None

    return {
        'PostedBy': posted_by,
        'PostedOn': posted_on,
        'PostedTime': posted_time,
        'PostedLocation': posted_location,
        'PostedInfoRaw': raw or None,
    }


def parse_detail_page(detail_soup: BeautifulSoup, detail_url: str):
    data = {
        'DetailURL': detail_url,
    }

    h1 = detail_soup.find('h1')
    if h1:
        data['AdTitle'] = h1.get_text(strip=True)

    # Posted by line is typically the first h2
    h2 = detail_soup.find('h2')
    if h2:
        data.update(parse_posted_info(h2.get_text(' ', strip=True)))

    main_img = detail_soup.select_one('#main-image')
    if main_img and main_img.get('src'):
        data['MainImageURL'] = urljoin(detail_url, main_img.get('src'))

    thumb_imgs = detail_soup.select('#thumbs img')
    image_urls = []
    for img in thumb_imgs:
        alt = (img.get('alt') or '').strip()
        if alt.startswith('http'):
            image_urls.append(alt)
    if image_urls:
        # Store as JSON list inside a single CSV cell
        data['ImageURLs'] = json.dumps(list(dict.fromkeys(image_urls)))

    label_map = {
        'Contact': 'Contact',
        'Price': 'DetailPrice',
        'Make': 'Make',
        'Model': 'Model',
        'YOM': 'YOM',
        'Mileage (km)': 'MileageKM',
        'Gear': 'Gear',
        'Fuel Type': 'FuelType',
        'Options': 'Options',
        'Engine (cc)': 'EngineCC',
        'Details': 'Details',
    }

    for p in detail_soup.select('table.moret p.moreh'):
        label = p.get_text(' ', strip=True)
        key = label_map.get(label)
        if not key:
            continue

        td = p.find_parent('td')
        value_td = td.find_next_sibling('td') if td else None
        if not value_td:
            continue

        if key == 'Details':
            value = value_td.get_text('\n', strip=True)
        else:
            value = value_td.get_text(' ', strip=True)

        data[key] = value

    # Normalize contact to digits only if present
    if data.get('Contact'):
        digits = re.sub(r'\D+', '', data['Contact'])
        if digits:
            data['ContactDigits'] = digits

    # Views
    for div in detail_soup.select('div.last'):
        t = div.get_text(' ', strip=True)
        if 'views' in t.lower():
            m = re.search(r'(\d+)\s*Views', t, flags=re.IGNORECASE)
            if m:
                data['Views'] = int(m.group(1))
            data['ViewsRaw'] = t
            break

    return data


# 3-7. Crawl pages through pagination
dataset = []
seen_keys = set()
interrupted = False
reached_limit = False

current_url = start_url
page_num = 1

driver = None
if use_selenium:
    try:
        driver = create_selenium_driver()
    except Exception as e:
        logger.error('Failed to start Selenium browser: %s', e)
        logger.error('Tip: set BROWSER=edge or BROWSER=chrome in .env')
        raise

try:
    while current_url:
        if max_items is not None and len(dataset) >= max_items:
            logger.info('Reached MAX_ITEMS=%s; stopping.', max_items)
            reached_limit = True
            break

        logger.info('Fetching page %s: %s', page_num, current_url)
        html = None
        status_code = None
        fetch_t0 = time.perf_counter()

        if use_selenium:
            try:
                html = fetch_html_with_driver(driver, current_url)
                status_code = 200
            except Exception as e:
                logger.error('Selenium fetch failed on page %s: %s', page_num, e)
                logger.error('Stopping.')
                break
        else:
            try:
                response = session.get(current_url, timeout=30)
                status_code = response.status_code
                html = response.text
            except requests.RequestException as e:
                logger.error('Request failed on page %s: %s', page_num, e)
                break

        logger.info('Page %s status=%s in %.2fs', page_num, status_code, time.perf_counter() - fetch_t0)
        if status_code == 403 and not use_selenium:
            logger.error('Got 403 Forbidden (likely bot protection).')
            snippet = (html or '')[:300].replace('\n', ' ')
            if snippet:
                logger.error('Response snippet: %s', snippet)
            logger.error('Tip: set USE_SELENIUM=1 in .env to fetch with a real browser.')
            break
        if status_code != 200:
            logger.error('Non-200 response; stopping.')
            break

        soup = BeautifulSoup(html, 'html.parser')

        car_listings_count = len(soup.find_all('li', class_='item'))
        logger.info('Listings found on page %s: %s', page_num, car_listings_count)
        if car_listings_count == 0:
            title_tag = soup.find('title')
            page_title = title_tag.get_text(strip=True) if title_tag else '(no title)'
            logger.error('No listings found. Page title: %s', page_title)
            logger.error('Stopping to avoid looping on unexpected HTML (blocked/changed layout).')
            break

        page_listings = parse_listings(soup, current_url)
        added_this_page = 0
        for idx, item in enumerate(page_listings, start=1):
            if max_items is not None and len(dataset) >= max_items:
                reached_limit = True
                break

            key = item.get('URL') or (item.get('Title'), item.get('Price'), item.get('Location'))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if scrape_details and item.get('URL'):
                detail_url = item.get('URL')
                try:
                    detail_t0 = time.perf_counter()
                    logger.debug('Detail fetch %s/%s (page %s): %s', idx, len(page_listings), page_num, detail_url)
                    if use_selenium and driver is not None:
                        detail_html = fetch_html_with_driver(driver, detail_url)
                    else:
                        detail_resp = session.get(detail_url, timeout=30)
                        if detail_resp.status_code != 200:
                            raise RuntimeError(f"detail status {detail_resp.status_code}")
                        detail_html = detail_resp.text

                    detail_soup = BeautifulSoup(detail_html, 'html.parser')
                    item.update(parse_detail_page(detail_soup, detail_url))
                    logger.debug('Detail parsed in %.2fs | make=%s model=%s yom=%s',
                                 time.perf_counter() - detail_t0,
                                 item.get('Make'), item.get('Model'), item.get('YOM'))
                except Exception as e:
                    item['DetailError'] = str(e)
                    logger.warning('Detail fetch/parse failed: %s | %s', detail_url, e)

            dataset.append(item)
            added_this_page += 1

        logger.info('Rows added from page %s: %s (total: %s)', page_num, added_this_page, len(dataset))

        if reached_limit:
            logger.info('Reached MAX_ITEMS=%s; stopping.', max_items)
            break

        next_url = find_next_page(soup, current_url)
        if not next_url:
            logger.info('No Next link found; finished.')
            break

        page_num += 1
        if max_pages is not None and page_num > max_pages:
            logger.info('Reached MAX_PAGES=%s; stopping.', max_pages)
            break

        current_url = next_url
        logger.debug('Next page url: %s', current_url)
        if request_delay > 0:
            time.sleep(request_delay)
except KeyboardInterrupt:
    logger.warning('Interrupted by user (Ctrl+C). Saving collected data...')
    interrupted = True
finally:
    if driver is not None:
        driver.quit()

# 8. Convert the list of dictionaries into a Tabular Pandas DataFrame
df = pd.DataFrame(dataset)

# 9. Save the tabular dataset to a CSV file for your ML model
output_path = Path('./data/sri_lankan_vehicles.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
logger.info('Successfully scraped %s vehicles and saved to %s', len(df), str(output_path))

if interrupted:
    sys.exit(0)