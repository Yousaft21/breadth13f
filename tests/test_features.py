"""Tests for the point-in-time breadth-feature panel on a hand-computed case.

Two periods, two securities, with deliberately chosen holder sets and share
counts so every breadth statistic (Δbreadth, new/exit rates, net-buyer breadth,
HHI, totals) has a known closed-form value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.breadth import build_feature_panel

CUSIP_A = "AAAAAAAA1"
CUSIP_B = "BBBBBBBB2"
FIGI_A = "FIGIAAAA0001"
FIGI_B = "FIGIBBBB0002"


def _secmaster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cusip": [CUSIP_A, CUSIP_B],
            "composite_figi": [FIGI_A, FIGI_B],
            "ticker": ["AAA", "BBB"],
            "is_us_common_equity": [True, True],
        }
    )


def _holdings() -> pd.DataFrame:
    # (period, manager, cusip, shares)
    p_pm1 = pd.Timestamp("2020-03-31")
    p_p = pd.Timestamp("2020-06-30")
    acc_pm1 = pd.Timestamp("2020-05-01")   # <= formation(p-1)=2020-05-16
    acc_p = pd.Timestamp("2020-08-01")     # <= formation(p)=2020-08-15
    recs = [
        # --- period p-1: A held by {1,2,3} @100 each; B held by {1} @100
        (p_pm1, 1, CUSIP_A, 100, acc_pm1),
        (p_pm1, 2, CUSIP_A, 100, acc_pm1),
        (p_pm1, 3, CUSIP_A, 100, acc_pm1),
        (p_pm1, 1, CUSIP_B, 100, acc_pm1),
        # --- period p: A held by {2:150(up),3:50(down),4:100(new)}; 1 exits
        (p_p, 2, CUSIP_A, 150, acc_p),
        (p_p, 3, CUSIP_A, 50, acc_p),
        (p_p, 4, CUSIP_A, 100, acc_p),
        # --- period p: B held by {1:200(up), 2:100(new)}
        (p_p, 1, CUSIP_B, 200, acc_p),
        (p_p, 2, CUSIP_B, 100, acc_p),
    ]
    df = pd.DataFrame(recs, columns=["period_of_report", "manager_cik", "cusip", "shares", "acceptance_datetime"])
    df["put_call"] = None
    df["value"] = df["shares"] * 10
    df["accession_number"] = ["acc%d" % i for i in range(len(df))]
    df["manager_name"] = "MGR"
    return df


def test_breadth_panel_values():
    panel = build_feature_panel(_holdings(), _secmaster(), filing_lag_days=46)
    # Only period p (2020-06-30) yields features (needs a prior quarter).
    assert set(panel["period"].unique()) == {pd.Timestamp("2020-06-30")}
    assert set(panel["composite_figi"]) == {FIGI_A, FIGI_B}
    # n_active_managers for period p = distinct filers {1,2,3,4} = 4.
    assert panel["n_active_managers"].unique().tolist() == [4]

    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_holders_p"] == 3 and a["n_holders_pm1"] == 3
    assert a["delta_breadth"] == 0
    assert np.isclose(a["breadth_level"], 3 / 4)
    assert np.isclose(a["new_position_rate"], 1 / 3)   # manager 4 initiated
    assert np.isclose(a["exit_rate"], 1 / 3)           # manager 1 exited
    assert np.isclose(a["net_buyer_breadth"], 0.0)     # one up (m2), one down (m3)
    # HHI of shares [150,50,100] / 300 -> 0.25+0.0278+0.1111
    assert np.isclose(a["hhi"], (0.5**2 + (1/6)**2 + (1/3)**2))
    assert np.isclose(a["top5_share"], 1.0)
    assert np.isclose(a["delta_total_inst_shares_pct"], 0.0)  # 300 -> 300

    b = panel.set_index("composite_figi").loc[FIGI_B]
    assert b["n_holders_p"] == 2 and b["n_holders_pm1"] == 1
    assert b["delta_breadth"] == 1
    assert np.isclose(b["delta_breadth_pct"], 1.0)
    assert np.isclose(b["new_position_rate"], 0.5)     # manager 2 initiated
    assert np.isclose(b["exit_rate"], 0.0)
    assert np.isclose(b["net_buyer_breadth"], 0.5)     # m1 up, no downs, 2 holders
    assert np.isclose(b["delta_total_inst_shares_pct"], 2.0)  # 100 -> 300


def test_pit_excludes_late_filings():
    # A filing accepted AFTER the formation date must not enter the snapshot.
    holdings = _holdings()
    # Add a late manager-5 position in A for period p accepted 2020-09-01 (> 08-15).
    late = pd.DataFrame(
        [{
            "period_of_report": pd.Timestamp("2020-06-30"), "manager_cik": 5,
            "cusip": CUSIP_A, "shares": 999, "acceptance_datetime": pd.Timestamp("2020-09-01"),
            "put_call": None, "value": 9990, "accession_number": "late", "manager_name": "LATE",
        }]
    )
    holdings = pd.concat([holdings, late], ignore_index=True)
    panel = build_feature_panel(holdings, _secmaster(), filing_lag_days=46)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    # Manager 5 excluded -> still 3 holders, not 4.
    assert a["n_holders_p"] == 3


def test_options_excluded():
    holdings = _holdings()
    holdings.loc[holdings["cusip"] == CUSIP_A, "put_call"] = None
    # Turn one period-p A holding into an option line; it should be dropped.
    idx = holdings[(holdings["cusip"] == CUSIP_A) & (holdings["manager_cik"] == 4)].index
    holdings.loc[idx, "put_call"] = "Call"
    panel = build_feature_panel(holdings, _secmaster(), filing_lag_days=46)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_holders_p"] == 2  # manager 4's option line excluded
