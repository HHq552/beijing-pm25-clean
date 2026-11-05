# -*- coding: utf-8 -*-
"""数据清洗主流程：原始站点-小时宽表 -> 站点/小时/日三级质控 -> 城市日度面板。

清洗规则
--------
1. 单元格：剔除空值、负值、超过物理上限的读数（PM2.5>1000、PM10>2000、AQI>500）
2. 小时：某整点有效站点数 >= 60%（21/35）才计算城市小时均值
3. 日：当日有效小时数 >= 18（75%）才计算日均值；源站缺失日不插值

用法：
    python src/clean.py --input data/raw_beijing_site_hour_2015_2025.csv.gz
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_STATION = 35
STATION_RATIO = 0.6
MIN_HOURS = 18
LIMITS = {'PM2.5': 1000.0, 'PM10': 2000.0, 'AQI': 500.0}
OUTPUT_COLS = ['date', 'pm25', 'pm10', 'aqi', 'temp', 'precip', 'wind', 'rh',
               'heating', 'ratio', 'ln_pm25', 'year', 'month', 'year_c']


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'date': str, 'hour': int, 'type': str})
    stations = [c for c in df.columns if c.startswith('站点')]
    df[stations] = df[stations].apply(pd.to_numeric, errors='coerce')
    print(f'读取原始数据：{len(df):,} 行 × {len(stations)} 站点，'
          f'{df["date"].nunique()} 天，指标 {sorted(df["type"].unique())}')
    return df


def cell_quality(df: pd.DataFrame, stations: list) -> dict:
    """单元格级质量统计（空值/负值/超限值）。"""
    cells = len(df) * len(stations)
    empty = int(df[stations].isna().sum().sum())
    negative = int((df[stations] < 0).sum().sum())
    over = int(sum((df.loc[df['type'] == t, stations] > lim).sum().sum()
                   for t, lim in LIMITS.items()))
    print(f'单元格总数 {cells:,}；空值 {empty:,}（{empty / cells:.2%}）；'
          f'负值 {negative}；超限 {over}（{over / cells:.3%}）')
    return {'raw_cells': cells, 'empty_cells': empty,
            'negative_values': negative, 'over_limit': over}


def aggregate_daily(df: pd.DataFrame, stations: list):
    """站点-小时 -> 城市小时 -> 城市日均，返回日度表与质控计数。"""
    rows, invalid_hours, hours_total, dropped = [], 0, 0, 0
    for ymd, g in df.groupby('date', sort=True):
        day = {}
        for typ in ('PM2.5', 'PM10', 'AQI'):
            arr = g[g['type'] == typ].set_index('hour')[stations]
            hourly = []
            for h in range(24):
                hours_total += 1
                if h not in arr.index:
                    invalid_hours += 1
                    continue
                v = arr.loc[h]
                valid = v[(v >= 0) & (v <= LIMITS[typ])].dropna()
                if len(valid) >= math.ceil(STATION_RATIO * N_STATION):
                    hourly.append(valid.mean())
                else:
                    invalid_hours += 1
            if len(hourly) >= MIN_HOURS:
                day[typ.lower().replace('.', '')] = round(float(np.mean(hourly)), 2)
        if 'pm25' not in day:
            dropped += 1
        rows.append({'date': f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}', **day})

    out = pd.DataFrame(rows)
    out['date'] = pd.to_datetime(out['date'])
    print(f'小时组合 {hours_total:,}，覆盖不足剔除 {invalid_hours:,}'
          f'（{invalid_hours / hours_total:.2%}）；日样本剔除 {dropped}')
    return out, {'hours_total': hours_total, 'invalid_hours': invalid_hours,
                 'days_dropped': dropped}


def build_panel(air: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    weather = pd.read_csv(weather_path)
    weather['date'] = pd.to_datetime(weather['date'])
    df = (air.merge(weather, on='date', how='left')
              .sort_values('date').reset_index(drop=True))
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['ln_pm25'] = np.log(df['pm25'])
    df['ratio'] = df['pm25'] / df['pm10']
    # 采暖季：11-15 至次年 03-15
    df['heating'] = (((df['month'] == 11) & (df['date'].dt.day >= 15)) |
                     (df['month'] == 12) | (df['month'] <= 2) |
                     ((df['month'] == 3) & (df['date'].dt.day <= 15))).astype(int)
    df['year_c'] = df['year'] - df['year'].min()
    return df[OUTPUT_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/raw_beijing_site_hour_2015_2025.csv.gz')
    ap.add_argument('--weather', default='data/weather_beijing_daily_2015_2025.csv')
    ap.add_argument('--output', default='output/beijing_pm25_clean_daily.csv')
    ap.add_argument('--stats', default='output/cleaning_stats.json')
    args = ap.parse_args()

    src, out = Path(args.input), Path(args.output)
    if not src.exists():
        sys.exit(f'输入文件不存在：{src}\n请先运行 src/download.py 并合并原始日文件。')
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw(src)
    stations = [c for c in raw.columns if c.startswith('站点')]
    stats = cell_quality(raw, stations)

    air, agg = aggregate_daily(raw, stations)
    stats.update(agg)

    panel = build_panel(air, Path(args.weather))
    valid = int(panel['pm25'].notna().sum())
    expect = (panel['date'].max() - panel['date'].min()).days + 1
    stats.update({'clean_days': valid, 'expected_days': expect,
                  'clean_coverage': round(valid / expect, 4),
                  'mean_station': float(len(stations)),
                  'weather_days': int(panel['wind'].notna().sum())})

    panel.to_csv(out, index=False, encoding='utf-8-sig')
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                encoding='utf-8')
    print(f'清洗完成：{valid}/{expect} 天（{valid / expect:.2%}）-> {out}')
    print(f'质控报告 -> {args.stats}')


if __name__ == '__main__':
    main()
