"""Tests for the point-in-time breadth-feature panel on a hand-computed case.

Two periods, two securities, with deliberately chosen holder sets and share
counts so every breadth statistic (Δbreadth, new/exit rates, net-buyer breadth,
HHI, totals) has a known closed-form value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.breadth import build_feature_panel, filing_deadline, formation_date

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
    acc_pm1 = pd.Timestamp("2020-05-01")   # <= deadline(p-1)=2020-05-15
    acc_p = pd.Timestamp("2020-08-01")     # <= deadline(p)=2020-08-14
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
    panel = build_feature_panel(_holdings(), _secmaster(), deadline_days=45)
    # Only period p (2020-06-30) yields features (needs a prior quarter).
    assert set(panel["period"].unique()) == {pd.Timestamp("2020-06-30")}
    assert set(panel["composite_figi"]) == {FIGI_A, FIGI_B}
    # n_active_managers for period p = distinct filers {1,2,3,4} = 4.
    assert panel["n_active_managers"].unique().tolist() == [4]

    # Manager 4 first files in p, so only managers {1,2,3} continue; the change
    # features ignore manager 4's position.
    assert panel["n_continuing_managers"].unique().tolist() == [3]

    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_holders_p"] == 3 and a["n_holders_pm1"] == 3
    assert np.isclose(a["delta_breadth"], (2 - 3) / 3)   # continuing holders 3 -> 2, per 3 managers
    assert np.isclose(a["delta_breadth_pct"], -1 / 3)
    assert np.isclose(a["breadth_level"], 3 / 4)
    assert np.isclose(a["new_position_rate"], 0.0)     # manager 4 is a new filer, not an initiation
    assert np.isclose(a["exit_rate"], 1 / 3)           # manager 1 still files, so it exited
    assert np.isclose(a["net_buyer_breadth"], 0.0)     # one up (m2), one down (m3)
    # HHI of shares [150,50,100] / 300 -> 0.25+0.0278+0.1111 (all holders)
    assert np.isclose(a["hhi"], (0.5**2 + (1/6)**2 + (1/3)**2))
    assert np.isclose(a["top5_share"], 1.0)
    # Continuing shares 300 -> 200; HHI among continuing: [1/3]*3 -> [0.75, 0.25].
    assert np.isclose(a["delta_total_inst_shares_pct"], -1 / 3)
    assert np.isclose(a["delta_hhi"], (0.75**2 + 0.25**2) - 1 / 3)

    b = panel.set_index("composite_figi").loc[FIGI_B]
    assert b["n_holders_p"] == 2 and b["n_holders_pm1"] == 1
    assert np.isclose(b["delta_breadth"], 1 / 3)
    assert np.isclose(b["delta_breadth_pct"], 1.0)
    assert np.isclose(b["new_position_rate"], 0.5)     # manager 2 initiated
    assert np.isclose(b["exit_rate"], 0.0)
    assert np.isclose(b["net_buyer_breadth"], 0.5)     # m1 up, no downs, 2 holders
    assert np.isclose(b["delta_total_inst_shares_pct"], 2.0)  # 100 -> 300


def test_missing_filer_is_not_an_exit():
    # Manager 3 files nothing for p by the deadline: its A position must not count
    # as an exit, and A's change features use managers {1, 2} only.
    holdings = _holdings()
    holdings = holdings[~((holdings["manager_cik"] == 3)
                          & (holdings["period_of_report"] == pd.Timestamp("2020-06-30")))]
    panel = build_feature_panel(holdings, _secmaster(), deadline_days=45)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_continuing_managers"] == 2
    assert np.isclose(a["exit_rate"], 1 / 2)                 # manager 1 only
    assert np.isclose(a["delta_breadth"], (1 - 2) / 2)       # holders {2} vs {1, 2}
    assert np.isclose(a["delta_total_inst_shares_pct"], 150 / 200 - 1)


def test_deadline_rolls_to_next_business_day():
    # 45 days after 2020-06-30 is Friday 2020-08-14: no roll; trade Monday.
    assert filing_deadline(pd.Timestamp("2020-06-30")) == pd.Timestamp("2020-08-14")
    assert formation_date(pd.Timestamp("2020-06-30")) == pd.Timestamp("2020-08-17")
    # Sunday 2016-08-14 -> Monday 2016-08-15.
    assert filing_deadline(pd.Timestamp("2016-06-30")) == pd.Timestamp("2016-08-15")
    # Sunday 2016-02-14, then Presidents' Day -> Tuesday 2016-02-16.
    assert filing_deadline(pd.Timestamp("2015-12-31")) == pd.Timestamp("2016-02-16")
    assert formation_date(pd.Timestamp("2015-12-31")) == pd.Timestamp("2016-02-17")


def test_deadline_day_filing_included():
    # A filing accepted on the evening of the deadline day is in the snapshot;
    # one accepted the next morning is not.
    holdings = _holdings()
    extra = pd.DataFrame(
        [{
            "period_of_report": pd.Timestamp("2020-06-30"), "manager_cik": m,
            "cusip": CUSIP_A, "shares": 10, "acceptance_datetime": pd.Timestamp(ts),
            "put_call": None, "value": 100, "accession_number": f"x{m}", "manager_name": "M",
        } for m, ts in [(5, "2020-08-14 21:30:00"), (6, "2020-08-15 08:00:00")]]
    )
    panel = build_feature_panel(pd.concat([holdings, extra], ignore_index=True),
                                _secmaster(), deadline_days=45)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_holders_p"] == 4
    assert a["formation_date"] == pd.Timestamp("2020-08-17")


def test_pit_excludes_late_filings():
    # A filing accepted AFTER the formation date must not enter the snapshot.
    holdings = _holdings()
    # Add a late manager-5 position in A for period p accepted 2020-09-01 (> 08-14).
    late = pd.DataFrame(
        [{
            "period_of_report": pd.Timestamp("2020-06-30"), "manager_cik": 5,
            "cusip": CUSIP_A, "shares": 999, "acceptance_datetime": pd.Timestamp("2020-09-01"),
            "put_call": None, "value": 9990, "accession_number": "late", "manager_name": "LATE",
        }]
    )
    holdings = pd.concat([holdings, late], ignore_index=True)
    panel = build_feature_panel(holdings, _secmaster(), deadline_days=45)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    # Manager 5 excluded -> still 3 holders, not 4.
    assert a["n_holders_p"] == 3


def test_options_excluded():
    holdings = _holdings()
    holdings.loc[holdings["cusip"] == CUSIP_A, "put_call"] = None
    # Turn one period-p A holding into an option line; it should be dropped.
    idx = holdings[(holdings["cusip"] == CUSIP_A) & (holdings["manager_cik"] == 4)].index
    holdings.loc[idx, "put_call"] = "Call"
    panel = build_feature_panel(holdings, _secmaster(), deadline_days=45)
    a = panel.set_index("composite_figi").loc[FIGI_A]
    assert a["n_holders_p"] == 2  # manager 4's option line excluded
