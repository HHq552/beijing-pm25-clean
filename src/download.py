# -*- coding: utf-8 -*-
"""数据获取：从公开数据源下载北京空气质量站点数据与 ERA5 气象数据。

用法：
    python src/download.py --start 2015-01-01 --end 2025-12-31
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

RAW_DIR = Path('data/raw')
AIR_URL = 'https://quotsoft.net/air/data/beijing_all_{ymd}.csv'
WEATHER_URL = (
    'https://archive-api.open-meteo.com/v1/archive'
    '?latitude=39.9042&longitude=116.4074'
    '&start_date={start}&end_date={end}'
    '&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean'
    '&timezone=Asia%2FShanghai'
)


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_air(ymd: str) -> str:
    """下载单个日文件；已存在且非空则跳过（支持断点续传）。"""
    path = RAW_DIR / f'beijing_all_{ymd}.csv'
    if path.exists() and path.stat().st_size > 1000:
        return 'skip'
    try:
        urllib.request.urlretrieve(AIR_URL.format(ymd=ymd), path)
        return 'ok'
    except Exception:
        return 'fail'


def download_air(start: date, end: date, workers: int = 10):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dates = [d.strftime('%Y%m%d') for d in daterange(start, end)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_air, dates))
    ok = results.count('ok')
    skip = results.count('skip')
    fail = results.count('fail')
    print(f'空气质量文件：下载 {ok}，跳过 {skip}，失败 {fail}')
    return {'ok': ok, 'skip': skip, 'fail': fail, 'total': len(dates)}


def download_weather(start: date, end: date):
    url = WEATHER_URL.format(start=start.isoformat(), end=end.isoformat())
    with urllib.request.urlopen(url, timeout=90) as r:
        payload = json.loads(r.read().decode('utf-8'))['daily']
    rows = list(zip(payload['time'], payload['temperature_2m_mean'],
                    payload['precipitation_sum'], payload['wind_speed_10m_max'],
                    payload['relative_humidity_2m_mean']))
    out = Path('data/weather_beijing_daily.csv')
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8-sig') as f:
        f.write('date,temp,precip,wind,rh\n')
        for row in rows:
            f.write(','.join('' if v is None else str(v) for v in row) + '\n')
    print(f'气象数据：{len(rows)} 天 -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2025-12-31')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    start, end = args.start.split('-'), args.end.split('-')
    download_air(date(*map(int, start)), date(*map(int, end)), args.workers)
    download_weather(date(*map(int, start)), date(*map(int, end)))


if __name__ == '__main__':
    main()
