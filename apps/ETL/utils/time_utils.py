"""
Time utility functions for ECAP period calculations
"""

from datetime import datetime
from typing import Tuple


def get_ecap_start_month(trnd_tm_prd_cd: str, trnd_tm_prd_end_mnth_nbr: int) -> int:
    """
    Compute start month (YYYYMM) for a given ECAP time period.

    Args:
        trnd_tm_prd_cd (str): One of ["R3", "R6", "R12", "YTD"]
        trnd_tm_prd_end_mnth_nbr (int): End month in YYYYMM format

    Returns:
        int: Start month in YYYYMM format
    """
    end_date = datetime.strptime(str(trnd_tm_prd_end_mnth_nbr), "%Y%m")

    def subtract_months(dt, months):
        year = dt.year
        month = dt.month - months

        while month <= 0:
            month += 12
            year -= 1

        return datetime(year, month, 1)

    if trnd_tm_prd_cd.startswith("R"):
        months = int(trnd_tm_prd_cd[1:])
        # standard rolling window (inclusive)
        start_date = subtract_months(end_date, months - 1)

    elif trnd_tm_prd_cd == "YTD":
        start_date = datetime(end_date.year, 1, 1)

    else:
        raise ValueError(f"Unsupported trnd_tm_prd_cd: {trnd_tm_prd_cd}")

    return int(start_date.strftime("%Y%m"))


def convert_current_ecap_time_to_previous_year(current_period_start: int,
                                               current_period_end: int) -> Tuple[int, int]:
    """
    Convert ECAP period (YYYYMM) to previous year period.

    Args:
        current_period_start (int): Start period in YYYYMM format
        current_period_end (int): End period in YYYYMM format

    Returns:
        tuple[int, int]: (previous_period_start, previous_period_end)
    """
    def shift_to_previous_year(period: int) -> int:
        year = period // 100
        month = period % 100

        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month in period: {period}")

        return (year - 1) * 100 + month

    previous_period_start = shift_to_previous_year(current_period_start)
    previous_period_end = shift_to_previous_year(current_period_end)

    return previous_period_start, previous_period_end
