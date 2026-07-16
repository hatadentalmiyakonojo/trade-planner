#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""株式トレードプランナー データ取得スクリプト

GitHub Actions (.github/workflows/daily.yml) から毎営業日実行される。
ローカル実行: .venv/bin/python scripts/fetch_data.py [--skip-info]

出力（リポジトリの data/ 配下）:
  meta.json     生成日時・取得品質・失敗銘柄
  market.json   指数・為替・セクターETF（JP17本+US11本）のOHLCV 約400営業日
  summary.json  全ユニバース銘柄の事前計算指標＋基礎ファンダ（スクリーナー用）
  ohlcv/*.json  個別銘柄の日足OHLCV（チャート表示時に遅延読込）

ユニバース（日経225構成・S&P100構成）は末尾の定数リスト。構成銘柄の入替は
年数回程度なので、年1回この定数を手動更新すれば十分（外部サイトへの依存を
なくして壊れにくくする方針）。
取得成功率が80%未満の場合は data/ を書き換えずに異常終了する（前日データ維持）。
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

SCHEMA = 1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FETCH_PERIOD = "800d"   # 暦日800日 ≒ 540営業日（SMA200を安定計算するための余裕）
KEEP_BARS = 400          # 保存する営業日数
BATCH_SIZE = 50
MIN_SUCCESS_RATE = 0.80

JST = timezone(timedelta(hours=9))

# ---- 指数・為替・ベンチマーク・セクターETF ----------------------------------
# (ticker, 表示名, 市場, 種別)  kind: index/bench/sector/fx
SERIES_DEFS = [
  ("^N225",  "日経平均株価",   "JP", "index"),
  ("1306.T", "TOPIX連動ETF",  "JP", "bench"),
  ("^GSPC",  "S&P500",        "US", "index"),
  ("^IXIC",  "NASDAQ総合",    "US", "index"),
  ("SPY",    "SPDR S&P500",   "US", "bench"),
  ("JPY=X",  "米ドル/円",      "FX", "fx"),
  # TOPIX-17 セクターETF（NEXT FUNDS）
  ("1617.T", "食品",                     "JP", "sector"),
  ("1618.T", "エネルギー資源",           "JP", "sector"),
  ("1619.T", "建設・資材",               "JP", "sector"),
  ("1620.T", "素材・化学",               "JP", "sector"),
  ("1621.T", "医薬品",                   "JP", "sector"),
  ("1622.T", "自動車・輸送機",           "JP", "sector"),
  ("1623.T", "鉄鋼・非鉄",               "JP", "sector"),
  ("1624.T", "機械",                     "JP", "sector"),
  ("1625.T", "電機・精密",               "JP", "sector"),
  ("1626.T", "情報通信・サービスその他", "JP", "sector"),
  ("1627.T", "電力・ガス",               "JP", "sector"),
  ("1628.T", "運輸・物流",               "JP", "sector"),
  ("1629.T", "商社・卸売",               "JP", "sector"),
  ("1630.T", "小売",                     "JP", "sector"),
  ("1631.T", "銀行",                     "JP", "sector"),
  ("1632.T", "金融（除く銀行）",         "JP", "sector"),
  ("1633.T", "不動産",                   "JP", "sector"),
  # 米国 SPDR セクターETF
  ("XLK",  "情報技術",         "US", "sector"),
  ("XLV",  "ヘルスケア",       "US", "sector"),
  ("XLF",  "金融",             "US", "sector"),
  ("XLY",  "一般消費財",       "US", "sector"),
  ("XLP",  "生活必需品",       "US", "sector"),
  ("XLE",  "エネルギー",       "US", "sector"),
  ("XLI",  "資本財",           "US", "sector"),
  ("XLB",  "素材",             "US", "sector"),
  ("XLU",  "公益",             "US", "sector"),
  ("XLRE", "不動産",           "US", "sector"),
  ("XLC",  "通信サービス",     "US", "sector"),
]


# ---- ユーティリティ ----------------------------------------------------------

def log(msg):
  print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)


def rnd(x, nd):
  if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
    return None
  return round(float(x), nd)


def clean_frame(df):
  """終値NaN行（未確定の当日・休場日）を除去し、時系列順に整える。"""
  if df is None or df.empty:
    return None
  df = df.dropna(subset=["Close"]).sort_index()
  if df.empty:
    return None
  return df


def download_batch(tickers, retries=3):
  """yf.downloadをリトライ付きで実行。戻り値: {ticker: DataFrame}"""
  for attempt in range(retries):
    try:
      raw = yf.download(tickers, period=FETCH_PERIOD, interval="1d",
                        group_by="ticker", auto_adjust=False,
                        threads=True, progress=False)
      out = {}
      for t in tickers:
        try:
          df = raw[t] if len(tickers) > 1 else raw
          df = clean_frame(df)
          if df is not None and len(df) >= 30:
            out[t] = df
        except (KeyError, TypeError):
          pass
      missing = [t for t in tickers if t not in out]
      if missing and attempt < retries - 1:
        # 失敗分だけ再取得（'database is locked' 等の一時故障対策）
        time.sleep(10 * (attempt + 1))
        tickers = missing
        merged = out
        raw2 = download_batch(missing, retries=1)
        merged.update(raw2)
        return merged
      return out
    except Exception as e:
      log(f"  batch error ({attempt+1}/{retries}): {e}")
      time.sleep(10 * (attempt + 1))
  return {}


# ---- 指標計算（pandas Series in/out） -----------------------------------------

def wilder_ema(s, n):
  return s.ewm(alpha=1.0 / n, adjust=False).mean()


def calc_indicators(df):
  """summary.json 用の最新値指標を dict で返す。"""
  c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
  n = len(c)
  sma25 = c.rolling(25).mean()
  sma75 = c.rolling(75).mean()
  sma200 = c.rolling(200).mean()
  # RSI14 (Wilder)
  diff = c.diff()
  gain = wilder_ema(diff.clip(lower=0), 14)
  loss = wilder_ema((-diff).clip(lower=0), 14)
  rsi = (100 - 100 / (1 + gain / loss)).where(loss > 0, 100.0)
  # ATR14 (Wilder)
  tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
  atr = wilder_ema(tr, 14)
  # 出来高・売買代金
  vol_r = v.rolling(5).mean() / v.rolling(20).mean().replace(0, float("nan"))
  turnover = c * v
  up_mask = (c > o).tail(20)
  t20 = turnover.tail(20)
  up_vol = float(t20[up_mask].sum() / t20.sum()) if t20.sum() > 0 else None
  # GC継続日数（SMA25 > SMA75 の連続日数）
  gc = (sma25 > sma75)
  gc_days = 0
  for val in gc.iloc[::-1]:
    if val:
      gc_days += 1
    else:
      break
  # 株式分割等の異常変動検出（未調整値運用のため警告フラグ）
  jumps = c.pct_change().abs().tail(300)
  split_warn = bool((jumps > 0.40).any())
  hi52 = float(h.tail(252).max())
  lo52 = float(l.tail(252).min())
  roc = lambda nn: float(c.iloc[-1] / c.iloc[-1 - nn] - 1) if n > nn else None
  s200_slope = None
  if n >= 220 and not math.isnan(sma200.iloc[-1]) and not math.isnan(sma200.iloc[-21]):
    s200_slope = float(sma200.iloc[-1] - sma200.iloc[-21])
  return {
    "c": float(c.iloc[-1]),
    "sma25": rnd(sma25.iloc[-1], 2), "sma75": rnd(sma75.iloc[-1], 2),
    "sma200": rnd(sma200.iloc[-1], 2), "s200up": (s200_slope or 0) > 0,
    "rsi": rnd(rsi.iloc[-1], 1), "atr": rnd(atr.iloc[-1], 2),
    "roc5": rnd(roc(5), 4), "roc20": rnd(roc(20), 4), "roc60": rnd(roc(60), 4),
    "volR": rnd(vol_r.iloc[-1], 2), "upVol": rnd(up_vol, 3),
    "hi52": rnd(hi52, 2), "lo52": rnd(lo52, 2),
    "gcDays": gc_days, "turn20": rnd(turnover.tail(20).mean(), 0),
    "splitWarn": split_warn,
  }


def frame_to_columnar(df, price_nd):
  df = df.tail(KEEP_BARS)
  dates = [int(d.strftime("%Y%m%d")) for d in df.index]
  col = lambda name, nd: [rnd(x, nd) for x in df[name].tolist()]
  return {
    "d": dates,
    "o": col("Open", price_nd), "h": col("High", price_nd),
    "l": col("Low", price_nd), "c": col("Close", price_nd),
    "v": [int(x) if not math.isnan(x) else 0 for x in df["Volume"].tolist()],
  }


def dump_json(path, obj):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


# ---- メイン -------------------------------------------------------------------

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--skip-info", action="store_true",
                  help="ファンダ(info)取得を省略（デバッグ用・高速）")
  args = ap.parse_args()

  t0 = time.time()
  new_dir = DATA_DIR + ".new"
  if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
  os.makedirs(os.path.join(new_dir, "ohlcv"))

  # 1) 指数・セクターETF
  log(f"series: {len(SERIES_DEFS)}本 取得開始")
  series_tickers = [t for t, _, _, _ in SERIES_DEFS]
  series_data = download_batch(series_tickers)
  series_out = {}
  for t, name, mkt, kind in SERIES_DEFS:
    if t in series_data:
      nd = 3 if kind == "fx" else (1 if mkt == "JP" else 2)
      series_out[t] = {"name": name, "mkt": mkt, "kind": kind,
                       **frame_to_columnar(series_data[t], nd)}
  series_fail = [t for t in series_tickers if t not in series_out]
  log(f"series: {len(series_out)}/{len(series_tickers)} 成功 失敗={series_fail}")

  usdjpy = None
  if "JPY=X" in series_out:
    usdjpy = series_out["JPY=X"]["c"][-1]

  # 2) 個別銘柄 OHLCV
  universe = ([(t, n, s, "JP") for t, n, s in UNIVERSE_JP] +
              [(t, n, s, "US") for t, n, s in UNIVERSE_US])
  log(f"universe: {len(universe)}銘柄 取得開始")
  stock_frames = {}
  tickers_all = [u[0] for u in universe]
  for i in range(0, len(tickers_all), BATCH_SIZE):
    batch = tickers_all[i:i + BATCH_SIZE]
    got = download_batch(batch)
    stock_frames.update(got)
    log(f"  {i + len(batch)}/{len(tickers_all)} 累計成功 {len(stock_frames)}")
    time.sleep(3)

  ok_rate = len(stock_frames) / len(tickers_all)
  failed = [t for t in tickers_all if t not in stock_frames]
  log(f"OHLCV成功率 {ok_rate:.1%} 失敗{len(failed)}銘柄: {failed[:10]}")
  if ok_rate < MIN_SUCCESS_RATE:
    log(f"成功率が{MIN_SUCCESS_RATE:.0%}未満のため中断（既存データを維持）")
    sys.exit(1)

  # 3) ファンダ（yfinance info）— 失敗しても続行して null 埋め
  funda = {}
  info_fail = 0
  if not args.skip_info:
    log("funda: info取得開始")
    for idx, t in enumerate(stock_frames.keys()):
      try:
        tk = yf.Ticker(t)
        info = tk.info
        dy = info.get("dividendYield")
        # 次回決算発表予定日（取れない銘柄はNoneのまま）
        earn = None
        try:
          cal = tk.calendar
          eds = cal.get("Earnings Date") if isinstance(cal, dict) else None
          if eds:
            earn = int(eds[0].strftime("%Y%m%d"))
        except Exception:
          pass
        funda[t] = {
          "per": rnd(info.get("trailingPE"), 1),
          "fper": rnd(info.get("forwardPE"), 1),
          "pbr": rnd(info.get("priceToBook"), 2),
          "dy": rnd(dy, 2),
          "mcap": info.get("marketCap"),
          "earn": earn,
        }
      except Exception:
        info_fail += 1
        funda[t] = {}
      if (idx + 1) % 50 == 0:
        log(f"  funda {idx + 1}/{len(stock_frames)} (fail {info_fail})")
      time.sleep(0.15)
    log(f"funda: 完了 fail={info_fail}")

  # 4) summary.json / ohlcv/*.json
  stocks = []
  for t, name, sec, mkt in universe:
    if t not in stock_frames:
      continue
    df = stock_frames[t]
    nd = 1 if mkt == "JP" else 2
    ind = calc_indicators(df)
    ind["c"] = rnd(ind["c"], nd)
    row = {"t": t, "name": name, "sec": sec, "mkt": mkt, **ind, **funda.get(t, {})}
    stocks.append(row)
    dump_json(os.path.join(new_dir, "ohlcv", f"{t}.json"),
              {"t": t, "name": name, "sec": sec, "mkt": mkt,
               **frame_to_columnar(df, nd)})

  generated = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
  dump_json(os.path.join(new_dir, "summary.json"),
            {"schema": SCHEMA, "generatedAt": generated, "usdjpy": usdjpy,
             "stocks": stocks})
  dump_json(os.path.join(new_dir, "market.json"),
            {"schema": SCHEMA, "generatedAt": generated, "series": series_out})
  dump_json(os.path.join(new_dir, "meta.json"),
            {"schema": SCHEMA, "generatedAt": generated, "usdjpy": usdjpy,
             "stockCount": len(stocks), "seriesCount": len(series_out),
             "okRate": round(ok_rate, 3), "failed": failed,
             "seriesFailed": series_fail, "infoFail": info_fail,
             "elapsedSec": round(time.time() - t0)})

  # 5) アトミックに差し替え
  if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
  os.rename(new_dir, DATA_DIR)
  log(f"完了: {len(stocks)}銘柄 / {len(series_out)}系列 / {time.time() - t0:.0f}秒")


# ---- ユニバース定義（年1回程度、手動更新） ------------------------------------
# 日経225構成銘柄（2026-07時点、JPX公表の銘柄名・TOPIX-17業種ETF対応）
# (ticker, 銘柄名, 所属セクターETF)
UNIVERSE_JP = [
  ("1332.T", "ニッスイ", "1617.T"),
  ("1605.T", "ＩＮＰＥＸ", "1618.T"),
  ("1721.T", "コムシスホールディングス", "1619.T"),
  ("1801.T", "大成建設", "1619.T"),
  ("1802.T", "大林組", "1619.T"),
  ("1803.T", "清水建設", "1619.T"),
  ("1808.T", "長谷工コーポレーション", "1619.T"),
  ("1812.T", "鹿島建設", "1619.T"),
  ("1925.T", "大和ハウス工業", "1619.T"),
  ("1928.T", "積水ハウス", "1619.T"),
  ("1963.T", "日揮ホールディングス", "1619.T"),
  ("2002.T", "日清製粉グループ本社", "1617.T"),
  ("2269.T", "明治ホールディングス", "1617.T"),
  ("2282.T", "日本ハム", "1617.T"),
  ("2413.T", "エムスリー", "1626.T"),
  ("2432.T", "ディー・エヌ・エー", "1626.T"),
  ("2501.T", "サッポロホールディングス", "1617.T"),
  ("2502.T", "アサヒグループホールディングス", "1617.T"),
  ("2503.T", "キリンホールディングス", "1617.T"),
  ("2768.T", "双日", "1629.T"),
  ("2801.T", "キッコーマン", "1617.T"),
  ("2802.T", "味の素", "1617.T"),
  ("2871.T", "ニチレイ", "1617.T"),
  ("2914.T", "日本たばこ産業", "1617.T"),
  ("3086.T", "Ｊ．フロント　リテイリング", "1630.T"),
  ("3092.T", "ＺＯＺＯ", "1630.T"),
  ("3099.T", "三越伊勢丹ホールディングス", "1630.T"),
  ("3289.T", "東急不動産ホールディングス", "1633.T"),
  ("3382.T", "セブン＆アイ・ホールディングス", "1630.T"),
  ("3401.T", "帝人", "1620.T"),
  ("3402.T", "東レ", "1620.T"),
  ("3405.T", "クラレ", "1620.T"),
  ("3407.T", "旭化成", "1620.T"),
  ("3436.T", "ＳＵＭＣＯ", "1619.T"),
  ("3659.T", "ネクソン", "1626.T"),
  ("3697.T", "ＳＨＩＦＴ", "1626.T"),
  ("3861.T", "王子ホールディングス", "1620.T"),
  ("4004.T", "レゾナック・ホールディングス", "1620.T"),
  ("4005.T", "住友化学", "1620.T"),
  ("4021.T", "日産化学", "1620.T"),
  ("4042.T", "東ソー", "1620.T"),
  ("4043.T", "トクヤマ", "1620.T"),
  ("4061.T", "デンカ", "1620.T"),
  ("4063.T", "信越化学工業", "1620.T"),
  ("4151.T", "協和キリン", "1621.T"),
  ("4183.T", "三井化学", "1620.T"),
  ("4188.T", "三菱ケミカルグループ", "1620.T"),
  ("4208.T", "ＵＢＥ", "1620.T"),
  ("4307.T", "野村総合研究所", "1626.T"),
  ("4324.T", "電通グループ", "1626.T"),
  ("4385.T", "メルカリ", "1626.T"),
  ("4452.T", "花王", "1620.T"),
  ("4502.T", "武田薬品工業", "1621.T"),
  ("4503.T", "アステラス製薬", "1621.T"),
  ("4506.T", "住友ファーマ", "1621.T"),
  ("4507.T", "塩野義製薬", "1621.T"),
  ("4519.T", "中外製薬", "1621.T"),
  ("4523.T", "エーザイ", "1621.T"),
  ("4543.T", "テルモ", "1625.T"),
  ("4568.T", "第一三共", "1621.T"),
  ("4578.T", "大塚ホールディングス", "1621.T"),
  ("4661.T", "オリエンタルランド", "1626.T"),
  ("4689.T", "ＬＩＮＥヤフー", "1626.T"),
  ("4704.T", "トレンドマイクロ", "1626.T"),
  ("4751.T", "サイバーエージェント", "1626.T"),
  ("4755.T", "楽天グループ", "1626.T"),
  ("4901.T", "富士フイルムホールディングス", "1620.T"),
  ("4902.T", "コニカミノルタ", "1625.T"),
  ("4911.T", "資生堂", "1620.T"),
  ("5019.T", "出光興産", "1618.T"),
  ("5020.T", "ＥＮＥＯＳホールディングス", "1618.T"),
  ("5101.T", "横浜ゴム", "1622.T"),
  ("5108.T", "ブリヂストン", "1622.T"),
  ("5201.T", "ＡＧＣ", "1619.T"),
  ("5214.T", "日本電気硝子", "1619.T"),
  ("5233.T", "太平洋セメント", "1619.T"),
  ("5301.T", "東海カーボン", "1619.T"),
  ("5332.T", "ＴＯＴＯ", "1619.T"),
  ("5333.T", "ＮＧＫ", "1619.T"),
  ("5401.T", "日本製鉄", "1623.T"),
  ("5406.T", "神戸製鋼所", "1623.T"),
  ("5411.T", "ＪＦＥホールディングス", "1623.T"),
  ("5631.T", "日本製鋼所", "1624.T"),
  ("5706.T", "三井金属", "1623.T"),
  ("5711.T", "三菱マテリアル", "1623.T"),
  ("5713.T", "住友金属鉱山", "1623.T"),
  ("5714.T", "ＤＯＷＡホールディングス", "1623.T"),
  ("5801.T", "古河電気工業", "1623.T"),
  ("5802.T", "住友電気工業", "1623.T"),
  ("5803.T", "フジクラ", "1623.T"),
  ("5831.T", "しずおかフィナンシャルグループ", "1631.T"),
  ("6098.T", "リクルートホールディングス", "1626.T"),
  ("6103.T", "オークマ", "1624.T"),
  ("6113.T", "アマダ", "1624.T"),
  ("6146.T", "ディスコ", "1624.T"),
  ("6178.T", "日本郵政", "1626.T"),
  ("6273.T", "ＳＭＣ", "1624.T"),
  ("6301.T", "小松製作所", "1624.T"),
  ("6302.T", "住友重機械工業", "1624.T"),
  ("6305.T", "日立建機", "1624.T"),
  ("6326.T", "クボタ", "1624.T"),
  ("6361.T", "荏原製作所", "1624.T"),
  ("6367.T", "ダイキン工業", "1624.T"),
  ("6471.T", "日本精工", "1624.T"),
  ("6472.T", "ＮＴＮ", "1624.T"),
  ("6473.T", "ジェイテクト", "1624.T"),
  ("6479.T", "ミネベアミツミ", "1625.T"),
  ("6501.T", "日立製作所", "1625.T"),
  ("6503.T", "三菱電機", "1625.T"),
  ("6504.T", "富士電機", "1625.T"),
  ("6506.T", "安川電機", "1625.T"),
  ("6526.T", "ソシオネクスト", "1625.T"),
  ("6532.T", "ベイカレント", "1626.T"),
  ("6594.T", "ニデック", "1625.T"),
  ("6645.T", "オムロン", "1625.T"),
  ("6701.T", "日本電気", "1625.T"),
  ("6702.T", "富士通", "1625.T"),
  ("6723.T", "ルネサスエレクトロニクス", "1625.T"),
  ("6724.T", "セイコーエプソン", "1625.T"),
  ("6752.T", "パナソニック　ホールディングス", "1625.T"),
  ("6753.T", "シャープ", "1625.T"),
  ("6758.T", "ソニーグループ", "1625.T"),
  ("6762.T", "ＴＤＫ", "1625.T"),
  ("6770.T", "アルプスアルパイン", "1625.T"),
  ("6841.T", "横河電機", "1625.T"),
  ("6857.T", "アドバンテスト", "1625.T"),
  ("6861.T", "キーエンス", "1625.T"),
  ("6902.T", "デンソー", "1622.T"),
  ("6920.T", "レーザーテック", "1625.T"),
  ("6954.T", "ファナック", "1625.T"),
  ("6963.T", "ローム", "1625.T"),
  ("6971.T", "京セラ", "1625.T"),
  ("6976.T", "太陽誘電", "1625.T"),
  ("6981.T", "村田製作所", "1625.T"),
  ("6988.T", "日東電工", "1620.T"),
  ("7004.T", "カナデビア", "1624.T"),
  ("7011.T", "三菱重工業", "1624.T"),
  ("7012.T", "川崎重工業", "1622.T"),
  ("7013.T", "ＩＨＩ", "1624.T"),
  ("7186.T", "横浜フィナンシャルグループ", "1631.T"),
  ("7201.T", "日産自動車", "1622.T"),
  ("7202.T", "いすゞ自動車", "1622.T"),
  ("7203.T", "トヨタ自動車", "1622.T"),
  ("7211.T", "三菱自動車工業", "1622.T"),
  ("7261.T", "マツダ", "1622.T"),
  ("7267.T", "本田技研工業", "1622.T"),
  ("7269.T", "スズキ", "1622.T"),
  ("7270.T", "ＳＵＢＡＲＵ", "1622.T"),
  ("7272.T", "ヤマハ発動機", "1622.T"),
  ("7453.T", "良品計画", "1630.T"),
  ("7532.T", "パン・パシフィック・インターナショナルホールディングス", "1630.T"),
  ("7731.T", "ニコン", "1625.T"),
  ("7733.T", "オリンパス", "1625.T"),
  ("7735.T", "ＳＣＲＥＥＮホールディングス", "1625.T"),
  ("7741.T", "ＨＯＹＡ", "1625.T"),
  ("7751.T", "キヤノン", "1625.T"),
  ("7752.T", "リコー", "1625.T"),
  ("7832.T", "バンダイナムコホールディングス", "1626.T"),
  ("7911.T", "ＴＯＰＰＡＮホールディングス", "1626.T"),
  ("7912.T", "大日本印刷", "1626.T"),
  ("7951.T", "ヤマハ", "1626.T"),
  ("7974.T", "任天堂", "1626.T"),
  ("8001.T", "伊藤忠商事", "1629.T"),
  ("8002.T", "丸紅", "1629.T"),
  ("8015.T", "豊田通商", "1629.T"),
  ("8031.T", "三井物産", "1629.T"),
  ("8035.T", "東京エレクトロン", "1625.T"),
  ("8053.T", "住友商事", "1629.T"),
  ("8058.T", "三菱商事", "1629.T"),
  ("8233.T", "高島屋", "1630.T"),
  ("8252.T", "丸井グループ", "1630.T"),
  ("8253.T", "クレディセゾン", "1632.T"),
  ("8267.T", "イオン", "1630.T"),
  ("8304.T", "あおぞら銀行", "1631.T"),
  ("8306.T", "三菱ＵＦＪフィナンシャル・グループ", "1631.T"),
  ("8308.T", "りそなホールディングス", "1631.T"),
  ("8309.T", "三井住友トラストグループ", "1631.T"),
  ("8316.T", "三井住友フィナンシャルグループ", "1631.T"),
  ("8331.T", "千葉銀行", "1631.T"),
  ("8354.T", "ふくおかフィナンシャルグループ", "1631.T"),
  ("8411.T", "みずほフィナンシャルグループ", "1631.T"),
  ("8591.T", "オリックス", "1632.T"),
  ("8601.T", "大和証券グループ本社", "1632.T"),
  ("8604.T", "野村ホールディングス", "1632.T"),
  ("8630.T", "ＳＯＭＰＯホールディングス", "1632.T"),
  ("8697.T", "日本取引所グループ", "1632.T"),
  ("8725.T", "ＭＳ＆ＡＤインシュアランスグループホールディングス", "1632.T"),
  ("8750.T", "第一ライフグループ", "1632.T"),
  ("8766.T", "東京海上ホールディングス", "1632.T"),
  ("8795.T", "Ｔ＆Ｄホールディングス", "1632.T"),
  ("8801.T", "三井不動産", "1633.T"),
  ("8802.T", "三菱地所", "1633.T"),
  ("8804.T", "東京建物", "1633.T"),
  ("8830.T", "住友不動産", "1633.T"),
  ("9001.T", "東武鉄道", "1628.T"),
  ("9005.T", "東急", "1628.T"),
  ("9007.T", "小田急電鉄", "1628.T"),
  ("9008.T", "京王電鉄", "1628.T"),
  ("9009.T", "京成電鉄", "1628.T"),
  ("9020.T", "東日本旅客鉄道", "1628.T"),
  ("9021.T", "西日本旅客鉄道", "1628.T"),
  ("9022.T", "東海旅客鉄道", "1628.T"),
  ("9064.T", "ヤマトホールディングス", "1628.T"),
  ("9101.T", "日本郵船", "1628.T"),
  ("9104.T", "商船三井", "1628.T"),
  ("9107.T", "川崎汽船", "1628.T"),
  ("9147.T", "ＮＩＰＰＯＮ　ＥＸＰＲＥＳＳホールディングス", "1628.T"),
  ("9201.T", "日本航空", "1628.T"),
  ("9202.T", "ＡＮＡホールディングス", "1628.T"),
  ("9432.T", "ＮＴＴ", "1626.T"),
  ("9433.T", "ＫＤＤＩ", "1626.T"),
  ("9434.T", "ソフトバンク", "1626.T"),
  ("9501.T", "東京電力ホールディングス", "1627.T"),
  ("9502.T", "中部電力", "1627.T"),
  ("9503.T", "関西電力", "1627.T"),
  ("9531.T", "東京瓦斯", "1627.T"),
  ("9532.T", "大阪瓦斯", "1627.T"),
  ("9602.T", "東宝", "1626.T"),
  ("9735.T", "セコム", "1626.T"),
  ("9766.T", "コナミグループ", "1626.T"),
  ("9843.T", "ニトリホールディングス", "1630.T"),
  ("9983.T", "ファーストリテイリング", "1630.T"),
  ("9984.T", "ソフトバンクグループ", "1626.T"),
]

# S&P100構成銘柄（2026-07時点、GICSセクター→SPDR ETF対応）
UNIVERSE_US = [
  ("AAPL", "Apple Inc.", "XLK"),
  ("ABBV", "AbbVie", "XLV"),
  ("ABT", "Abbott Laboratories", "XLV"),
  ("ACN", "Accenture", "XLK"),
  ("ADBE", "Adobe Inc.", "XLK"),
  ("AMAT", "Applied Materials", "XLK"),
  ("AMD", "Advanced Micro Devices", "XLK"),
  ("AMGN", "Amgen", "XLV"),
  ("AMT", "American Tower", "XLRE"),
  ("AMZN", "Amazon", "XLY"),
  ("AVGO", "Broadcom", "XLK"),
  ("AXP", "American Express", "XLF"),
  ("BA", "Boeing", "XLI"),
  ("BAC", "Bank of America", "XLF"),
  ("BKNG", "Booking Holdings", "XLY"),
  ("BLK", "BlackRock", "XLF"),
  ("BMY", "Bristol Myers Squibb", "XLV"),
  ("BNY", "BNY Mellon", "XLF"),
  ("BRK-B", "Berkshire Hathaway (Class B)", "XLF"),
  ("C", "Citigroup", "XLF"),
  ("CAT", "Caterpillar Inc.", "XLI"),
  ("CL", "Colgate-Palmolive", "XLP"),
  ("CMCSA", "Comcast", "XLC"),
  ("COF", "Capital One", "XLF"),
  ("COP", "ConocoPhillips", "XLE"),
  ("COST", "Costco", "XLP"),
  ("CRM", "Salesforce", "XLK"),
  ("CSCO", "Cisco", "XLK"),
  ("CVS", "CVS Health", "XLV"),
  ("CVX", "Chevron Corporation", "XLE"),
  ("DE", "Deere & Company", "XLI"),
  ("DHR", "Danaher Corporation", "XLV"),
  ("DIS", "Walt Disney Company (The)", "XLC"),
  ("DUK", "Duke Energy", "XLU"),
  ("EMR", "Emerson Electric", "XLI"),
  ("FDX", "FedEx", "XLI"),
  ("GD", "General Dynamics", "XLI"),
  ("GE", "GE Aerospace", "XLI"),
  ("GEV", "GE Vernova", "XLI"),
  ("GILD", "Gilead Sciences", "XLV"),
  ("GM", "General Motors", "XLY"),
  ("GOOG", "Alphabet Inc. (Class C)", "XLC"),
  ("GOOGL", "Alphabet Inc. (Class A)", "XLC"),
  ("GS", "Goldman Sachs", "XLF"),
  ("HD", "Home Depot", "XLY"),
  ("HONA", "Honeywell Aerospace", "XLI"),
  ("IBM", "IBM", "XLK"),
  ("INTC", "Intel", "XLK"),
  ("INTU", "Intuit", "XLK"),
  ("ISRG", "Intuitive Surgical", "XLV"),
  ("JNJ", "Johnson & Johnson", "XLV"),
  ("JPM", "JPMorgan Chase", "XLF"),
  ("KO", "Coca-Cola Company (The)", "XLP"),
  ("LIN", "Linde plc", "XLB"),
  ("LLY", "Eli Lilly and Company", "XLV"),
  ("LMT", "Lockheed Martin", "XLI"),
  ("LOW", "Lowe's", "XLY"),
  ("LRCX", "Lam Research", "XLK"),
  ("MA", "Mastercard", "XLF"),
  ("MCD", "McDonald's", "XLY"),
  ("MDLZ", "Mondelēz International", "XLP"),
  ("MDT", "Medtronic", "XLV"),
  ("META", "Meta Platforms", "XLC"),
  ("MMM", "3M", "XLI"),
  ("MO", "Altria", "XLP"),
  ("MRK", "Merck & Co.", "XLV"),
  ("MS", "Morgan Stanley", "XLF"),
  ("MSFT", "Microsoft", "XLK"),
  ("MU", "Micron Technology", "XLK"),
  ("NEE", "NextEra Energy", "XLU"),
  ("NFLX", "Netflix, Inc.", "XLC"),
  ("NKE", "Nike, Inc.", "XLY"),
  ("NOW", "ServiceNow", "XLK"),
  ("NVDA", "Nvidia", "XLK"),
  ("ORCL", "Oracle Corporation", "XLK"),
  ("PEP", "PepsiCo", "XLP"),
  ("PFE", "Pfizer", "XLV"),
  ("PG", "Procter & Gamble", "XLP"),
  ("PLTR", "Palantir Technologies", "XLK"),
  ("PM", "Philip Morris International", "XLP"),
  ("QCOM", "Qualcomm", "XLK"),
  ("RTX", "RTX Corporation", "XLI"),
  ("SBUX", "Starbucks", "XLY"),
  ("SCHW", "Charles Schwab Corporation", "XLF"),
  ("SO", "Southern Company", "XLU"),
  ("SPG", "Simon Property Group", "XLRE"),
  ("T", "AT&T", "XLC"),
  ("TMO", "Thermo Fisher Scientific", "XLV"),
  ("TMUS", "T-Mobile US", "XLC"),
  ("TSLA", "Tesla, Inc.", "XLY"),
  ("TXN", "Texas Instruments", "XLK"),
  ("UBER", "Uber", "XLI"),
  ("UNH", "UnitedHealth Group", "XLV"),
  ("UNP", "Union Pacific Corporation", "XLI"),
  ("UPS", "United Parcel Service", "XLI"),
  ("USB", "U.S. Bancorp", "XLF"),
  ("V", "Visa Inc.", "XLF"),
  ("VZ", "Verizon", "XLC"),
  ("WFC", "Wells Fargo", "XLF"),
  ("WMT", "Walmart", "XLP"),
  ("XOM", "ExxonMobil", "XLE"),
]

if __name__ == "__main__":
  main()
