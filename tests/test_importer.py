"""
Unit and integration tests for the TradingView Excel Importer.
"""

import os
import tempfile
import glob
from datetime import datetime, timedelta
import pandas as pd
import pytest

from portfolio.importer import (
    Trade,
    import_tradingview_files,
    parse_tradingview_file,
    normalize_column_name,
)

# Root directory of the sample files
SAMPLE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tradingview-xlsx-export-samples"
    )
)


@pytest.fixture
def mock_english_report_path():
    """Generates a temporary mock English TradingView Excel report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "My_Strategy_OKX_BTCUSDT.P_2026-07-13_abc12.xlsx")
        
        # 1. Properties
        df_prop = pd.DataFrame({
            "name": ["Symbol", "Trading range", "Initial capital", "Commission"],
            "value": ["OKX:BTCUSDT.P", "Jan 1, 2026 — Jul 13, 2026", "1000", "0.05"]
        })
        
        # 2. Performance Summary
        df_perf = pd.DataFrame({
            "Unnamed: 0": ["Initial capital", "Net profit"],
            "All USDT": [1000, 500],
            "All %": [0.0, 50.0]
        })
        
        # 3. Trades
        df_trades = pd.DataFrame({
            "Trade number": [1, 1, 2, 2],
            "Type": ["Exit long", "Entry long", "Exit short", "Entry short"],
            "Date and time": [
                "2026-01-02 12:00:00",
                "2026-01-01 12:00:00",
                "2026-01-05 12:00:00",
                "2026-01-04 12:00:00"
            ],
            "Signal": ["Close entry", "Long entry", "Close short", "Short entry"],
            "Price USDT": [105.0, 100.0, 95.0, 100.0],
            "Size (qty)": [10.0, 10.0, 5.0, 5.0],
            "Net PnL USDT": [50.0, 50.0, 25.0, 25.0],
            "Return %": [5.0, 5.0, 5.0, 5.0]
        })
        
        # 4. Orders
        df_orders = pd.DataFrame({
            "Order number": [1, 2, 3, 4],
            "Ticker": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "Type": ["Buy", "Sell", "Sell", "Buy"]
        })
        
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df_prop.to_excel(writer, sheet_name="Properties", index=False)
            df_perf.to_excel(writer, sheet_name="Performance", index=False)
            df_trades.to_excel(writer, sheet_name="Trades", index=False)
            df_orders.to_excel(writer, sheet_name="Orders", index=False)
            
        yield file_path


@pytest.fixture
def mock_chinese_report_path():
    """Generates a temporary mock Chinese TradingView Excel report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "My_Strategy_OKX_BTCUSDT.P_2026-07-13_abc12.xlsx")
        
        # 1. Properties
        df_prop = pd.DataFrame({
            "name": ["商品代码", "交易范围", "初始资金", "佣金"],
            "value": ["OKX:BTCUSDT.P", "2026年1月1日 — 2026年7月13日", "1000", "0.05"]
        })
        
        # 2. Performance Summary
        df_perf = pd.DataFrame({
            "Unnamed: 0": ["初始资本", "净利润"],
            "全部 USDT": [1000, 500],
            "全部 %": [0.0, 50.0]
        })
        
        # 3. Trades
        df_trades = pd.DataFrame({
            "交易编号": [1, 1, 2, 2],
            "类型": ["多头出场", "多头进场", "空头出场", "空头进场"],
            "日期和时间": [
                "2026-01-02 12:00:00",
                "2026-01-01 12:00:00",
                "2026-01-05 12:00:00",
                "2026-01-04 12:00:00"
            ],
            "信号": ["Close entry", "Long entry", "Close short", "Short entry"],
            "价格 USDT": [105.0, 100.0, 95.0, 100.0],
            "大小（数量）": [10.0, 10.0, 5.0, 5.0],
            "净损益 USDT": [50.0, 50.0, 25.0, 25.0],
            "回报 %": [5.0, 5.0, 5.0, 5.0]
        })
        
        # 4. Orders
        df_orders = pd.DataFrame({
            "订单编号": [1, 2, 3, 4],
            "Ticker": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "类型": ["买入", "卖出", "卖出", "买入"]
        })
        
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df_prop.to_excel(writer, sheet_name="属性", index=False)
            df_perf.to_excel(writer, sheet_name="表现", index=False)
            df_trades.to_excel(writer, sheet_name="交易", index=False)
            df_orders.to_excel(writer, sheet_name="订单", index=False)
            
        yield file_path


def test_normalize_column_name():
    """Verify column normalization logic."""
    assert normalize_column_name("Net PnL (USDT)") == "net pnl usdt"
    assert normalize_column_name("交易编号") == "交易编号"
    assert normalize_column_name("  Size (qty)  ") == "size qty"


def test_parse_mock_english_report(mock_english_report_path):
    """Test parsing of a simulated English report."""
    report = parse_tradingview_file(mock_english_report_path)
    
    assert report.strategy_name == "My_Strategy"
    assert len(report.trades) == 2
    
    # Trade 1
    t1 = report.trades[0]
    assert t1.trade_id == 1
    assert t1.side == "Long"
    assert t1.entry_price == 100.0
    assert t1.exit_price == 105.0
    assert t1.contracts == 10.0
    assert t1.position_value == 1000.0
    assert t1.profit == 50.0
    assert t1.profit_percent == 5.0
    assert t1.holding_time == timedelta(days=1)
    # Commission = (100 * 10 + 105 * 10) * 0.0005 = 2050 * 0.0005 = 1.025
    assert abs(t1.commission - 1.025) < 1e-6
    
    # Trade 2
    t2 = report.trades[1]
    assert t2.trade_id == 2
    assert t2.side == "Short"
    assert t2.entry_price == 100.0
    assert t2.exit_price == 95.0
    assert t2.contracts == 5.0
    assert t2.position_value == 500.0
    assert t2.profit == 25.0
    assert t2.profit_percent == 5.0
    assert t2.holding_time == timedelta(days=1)


def test_parse_mock_chinese_report(mock_chinese_report_path):
    """Test parsing of a simulated Chinese report."""
    report = parse_tradingview_file(mock_chinese_report_path)
    
    assert report.strategy_name == "My_Strategy"
    assert len(report.trades) == 2
    
    # Trade 1
    t1 = report.trades[0]
    assert t1.trade_id == 1
    assert t1.side == "Long"
    assert t1.entry_price == 100.0
    assert t1.exit_price == 105.0
    assert t1.contracts == 10.0
    assert t1.position_value == 1000.0
    assert t1.profit == 50.0
    assert t1.profit_percent == 5.0
    assert t1.holding_time == timedelta(days=1)


def test_import_tradingview_files_glob(mock_english_report_path):
    """Test importing using glob pattern and lists."""
    trades = import_tradingview_files(os.path.dirname(mock_english_report_path))
    assert len(trades) == 2
    
    trades_list = import_tradingview_files([mock_english_report_path])
    assert len(trades_list) == 2


def test_real_samples_integration():
    """Integration test that runs on the actual sample files in the workspace directory."""
    if not os.path.exists(SAMPLE_DIR):
        pytest.skip(f"Sample folder not found at {SAMPLE_DIR}")
        
    xlsx_files = glob.glob(os.path.join(SAMPLE_DIR, "*.xlsx"))
    xlsx_files = [
        f
        for f in xlsx_files
        if not os.path.basename(f).startswith("~$")
        and not os.path.basename(f).startswith(".~")
    ]
    
    if not xlsx_files:
        pytest.skip("No actual Excel reports found in sample directory")
        
    print(f"\nTesting actual integration on: {xlsx_files}")
    
    for f in xlsx_files:
        report = parse_tradingview_file(f)
        assert len(report.trades) > 0
        assert report.strategy_name in [
            "Zen_Kirin_1.0",
            "ZEN_Arion_1.0",
            "CTC_AZ-SuperTrend_2026.1",
            "CTC_Pixiu_v.5.1",
        ]
        
        # Verify Trade properties
        for t in report.trades:
            assert isinstance(t.strategy_name, str)
            assert isinstance(t.trade_id, int)
            assert isinstance(t.entry_time, datetime)
            assert isinstance(t.exit_time, datetime)
            assert t.side in ["Long", "Short"]
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.contracts > 0
            assert t.position_value > 0
            assert t.commission >= 0
            assert isinstance(t.holding_time, timedelta)
