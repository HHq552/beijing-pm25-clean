# -*- coding: utf-8 -*-
"""统计分析：描述性统计、可视化、气象校正回归与归因分解。

用法：
    python src/analyze.py
"""
import json
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

FIG = Path('figures')
OUT = Path('output')
FIG.mkdir(exist_ok=True)


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = {'pm25': 'PM2.5 (μg/m³)', 'pm10': 'PM10 (μg/m³)', 'aqi': 'AQI',
              'ratio': 'PM2.5/PM10', 'temp': '气温 (℃)', 'wind': '风速 (km/h)',
              'rh': '相对湿度 (%)', 'precip': '降水 (mm)'}
    return pd.DataFrame({
        '变量': labels.values(),
        '样本量': [int(df[k].notna().sum()) for k in labels],
        '均值': [df[k].mean() for k in labels],
        '标准差': [df[k].std() for k in labels],
        '最小值': [df[k].min() for k in labels],
        '中位数': [df[k].median() for k in labels],
        'P95': [df[k].quantile(.95) for k in labels],
        '最大值': [df[k].max() for k in labels],
    })


def annual_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=['pm25'])
    g = sub.groupby('year')
    tbl = pd.DataFrame({
        'n': g.size(),
        'pm25_mean': g['pm25'].mean(),
        'pm25_med': g['pm25'].median(),
        'pm25_p95': g['pm25'].apply(lambda s: np.percentile(s, 95)),
        'heavy': g['pm25'].apply(lambda s: int((s > 150).sum())),
        'attain': g['pm25'].apply(lambda s: float((s <= 75).mean() * 100)),
    }).reset_index()
    return tbl


def fit_models(df: pd.DataFrame):
    reg = df.dropna(subset=['ln_pm25', 'temp', 'wind', 'rh', 'precip']).copy()
    mon = pd.get_dummies(reg['month'], prefix='m', drop_first=True, dtype=float)
    X1 = sm.add_constant(pd.concat([reg[['year_c']], mon], axis=1))
    X2 = sm.add_constant(pd.concat([reg[['year_c', 'wind', 'temp', 'rh', 'precip']], mon], axis=1))
    m1 = sm.OLS(reg['ln_pm25'], X1).fit(cov_type='HAC', cov_kwds={'maxlags': 14})
    m2 = sm.OLS(reg['ln_pm25'], X2).fit(cov_type='HAC', cov_kwds={'maxlags': 14})
    return reg, m1, m2


def weather_counterfactual(reg: pd.DataFrame, m2) -> pd.DataFrame:
    wcols = ['wind', 'temp', 'rh', 'precip']
    ref = reg[reg.year == reg.year.min()][wcols].mean()
    rows = []
    for y, sub in reg.groupby('year'):
        wy = sub[wcols].mean()
        eff = sum(m2.params[k] * (wy[k] - ref[k]) for k in wcols)
        rows.append({'year': int(y), 'observed': sub['pm25'].mean(),
                     'adjusted': sub['pm25'].mean() / math.exp(eff),
                     'weather_effect': eff})
    return pd.DataFrame(rows), ref


def fig_timeseries(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(df['date'], df['pm25'], color='#9ec5d8', lw=0.55, alpha=0.75)
    roll = df.set_index('date')['pm25'].rolling(30, min_periods=20).mean()
    ax.plot(roll.index, roll.values, color='#1f4e79', lw=1.6)
    for y in range(df['year'].min(), df['year'].max() + 1):
        ax.axvspan(pd.Timestamp(f'{y}-11-15'), pd.Timestamp(f'{y + 1}-03-15'),
                   color='#d9d9d9', alpha=0.30, lw=0)
    ax.set_ylabel('PM2.5 (μg/m³)'), ax.set_xlabel('年份')
    ax.set_ylim(0, 520)
    ax.set_xlim(df['date'].min(), df['date'].max())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper right', frameon=False, fontsize=9,
              handles=[plt.Line2D([], [], color='#9ec5d8', lw=1, label='PM2.5 日均值'),
                       plt.Line2D([], [], color='#1f4e79', lw=1.6, label='30 日滑动平均'),
                       Patch(facecolor='#d9d9d9', alpha=0.5, label='采暖季')])
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(FIG / 'fig1_timeseries.png')
    plt.close(fig)


def fig_annual(yr):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(yr['year'], yr['pm25_mean'], color='#4c78a8', width=0.62)
    for x, v in zip(yr['year'], yr['pm25_mean']):
        ax.text(x, v + 1.3, f'{v:.1f}', ha='center', fontsize=7.5, color='#2f4b6e')
    ax.set_ylabel('年均 PM2.5 (μg/m³)'), ax.set_xlabel('年份')
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(yr['year'], yr['heavy'], color='#d1495b', marker='o', ms=3.5, lw=1.4)
    ax2.set_ylabel('重污染天数', color='#d1495b')
    ax2.set_ylim(0, 60)
    ax.set_xticks(yr['year'])
    ax.spines[['top']].set_visible(False)
    fig.savefig(FIG / 'fig2_annual.png')
    plt.close(fig)


def fig_seasonal(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    data = [df.loc[df.month == m, 'pm25'].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(data, widths=0.58, patch_artist=True, showfliers=False,
                    medianprops=dict(color='#1f4e79', lw=1.2))
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='#e8a33d' if i in (10, 11, 0, 1, 2) else '#a8c6dd',
                alpha=0.85, lw=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
    ax.set_ylabel('PM2.5 (μg/m³)'), ax.set_xlabel('月份')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(FIG / 'fig3_seasonal.png')
    plt.close(fig)


def fig_weather(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, var, lab in [(axes[0], 'wind', '日最大风速 (km/h)'),
                         (axes[1], 'rh', '日均相对湿度 (%)')]:
        ax.scatter(df[var], df['pm25'], s=3, alpha=0.18, color='#2f7f6f', edgecolors='none')
        binned = df.groupby(pd.qcut(df[var], 10, duplicates='drop'), observed=True).agg(
            x=(var, 'mean'), y=('pm25', 'mean'))
        ax.plot(binned['x'], binned['y'], color='#d1495b', lw=1.8, marker='o', ms=3.5)
        ax.set_xlabel(lab), ax.set_ylabel('PM2.5 (μg/m³)')
        ax.set_ylim(0, 380)
        r = df['pm25'].corr(df[var])
        ax.text(0.97, 0.94, f'r = {r:.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_meteorology.png')
    plt.close(fig)


def fig_adjusted(adj):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(adj['year'], adj['observed'], color='#4c78a8', marker='o', ms=4, lw=1.8,
            label='实际年均浓度')
    ax.plot(adj['year'], adj['adjusted'], color='#d1495b', marker='s', ms=3.6, lw=1.6,
            ls='--', label='按基年气象条件校正')
    ax.fill_between(adj['year'], adj['adjusted'], adj['observed'], color='#4c78a8', alpha=0.08)
    ax.set_xlabel('年份'), ax.set_ylabel('PM2.5 (μg/m³)')
    ax.set_xticks(adj['year'])
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(FIG / 'fig5_adjusted.png')
    plt.close(fig)


def main():
    df = pd.read_csv(OUT / 'beijing_pm25_clean_daily.csv', parse_dates=['date'])

    desc = descriptive_table(df)
    desc.to_csv(OUT / 'descriptive_stats.csv', index=False,
                encoding='utf-8-sig', float_format='%.2f')

    yr = annual_table(df)
    yr.to_csv(OUT / 'annual_summary.csv', index=False,
              encoding='utf-8-sig', float_format='%.2f')

    reg, m1, m2 = fit_models(df)
    coefs = pd.DataFrame({'变量': m2.params.index, '系数': m2.params.values,
                          '标准误': m2.bse.values, 't值': m2.tvalues.values,
                          'p值': m2.pvalues.values})
    coefs.to_csv(OUT / 'regression_results.csv', index=False,
                 encoding='utf-8-sig', float_format='%.4f')

    adj, ref = weather_counterfactual(reg, m2)
    adj.to_csv(OUT / 'weather_adjusted_annual.csv', index=False,
               encoding='utf-8-sig', float_format='%.2f')

    # 十年对数变化分解（两部分可加总）
    last = reg[reg.year == reg.year.max()]
    wcols = ['wind', 'temp', 'rh', 'precip']
    total = math.log(last['pm25'].mean() / reg[reg.year == reg.year.min()]['pm25'].mean())
    weather_part = sum(m2.params[k] * (last[k].mean() - ref[k]) for k in wcols)
    trend_part = total - weather_part

    summary = {
        'm1_year_coef': float(m1.params['year_c']), 'm1_r2': float(m1.rsquared),
        'm2_year_coef': float(m2.params['year_c']), 'm2_r2': float(m2.rsquared),
        'm2_n': int(m2.nobs),
        'observed_change_pct': float(math.exp(total) - 1),
        'trend_part': trend_part, 'weather_part': weather_part,
        'trend_share': trend_part / total, 'weather_share': weather_part / total,
        'corr': {k: float(df['pm25'].corr(df[k])) for k in wcols},
    }
    (OUT / 'analysis_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    fig_timeseries(df)
    fig_annual(yr)
    fig_seasonal(df)
    fig_weather(df)
    fig_adjusted(adj)

    print(f'年均浓度 {yr["pm25_mean"].iloc[0]:.1f} -> {yr["pm25_mean"].iloc[-1]:.1f} μg/m³'
          f'（{math.exp(total) - 1:.1%}）')
    print(f'趋势项 {m2.params["year_c"]:.4f}（R²={m2.rsquared:.3f}）；'
          f'十年降幅中趋势占 {trend_part / total:.1%}，气象占 {weather_part / total:.1%}')
    print('图表已输出至 figures/')


if __name__ == '__main__':
    main()
