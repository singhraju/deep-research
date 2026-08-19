
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, Any
from datetime import date
import calendar
from dateutil.relativedelta import relativedelta

class UserIntent(BaseModel):
    """
    Clean, validated representation of user intent.

    Handles a common LLM parsing failure mode where a full intent dict
    is accidentally assigned into `time_start`.
    """

    model_config = ConfigDict(extra="ignore")  # ignore unknown keys safely

    time_start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    time_end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    state: Optional[str] = Field(default=None, description="2-letter US state code")
    market: Optional[str] = Field(default=None, description="Business market, e.g. CAID/CARE")
    provider: Optional[str] = Field(default=None, description="Provider ID (RNDRG_PROV_ID)")
    num_months: int = Field(default=0, description="Number of months between start and end")
    
    # --- Field-level normalization ---
    @field_validator("time_start", "time_end", mode="before")
    @classmethod
    def validate_date(cls, v: Any) -> Optional[str]:
        # If dict accidentally appears here, defer to model_validator to unwrap it.
        if isinstance(v, dict):
            return v  # keep it for model-level handling
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            if DATE_RE.match(v):
                return v
        # Anything else -> None (don’t apply timestamp filter)
        return None

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().upper()
            if len(s) == 2 and s.isalpha():
                return s
        return None

    @field_validator("market", "provider", mode="before")
    @classmethod
    def strip_text(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return None

    @model_validator(mode="after")
    def validate_time_range(self) -> "UserIntent":
        """
        Optional: enforce start <= end when both exist.
        Works lexicographically for YYYY-MM-DD, but we’ll be explicit-safe.
        Sets end date to today's date if None.
        Sets start date to 12 months behind current date if None
        Sets num_months to the number of months
        """
        if not self.time_end:
            today = date.today()
            end_date = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
            object.__setattr__(self, "time_end", end_date.strftime("%Y-%m-%d"))
        else:
            end_date = date.fromisoformat(self.time_end)
            end_date = date(end_date.year, end_date.month, calendar.monthrange(end_date.year, end_date.month)[1])
            object.__setattr__(self, "time_end", end_date.strftime("%Y-%m-%d"))
        
        if not self.time_start:
            end_date = date.fromisoformat(self.time_end)
            start_date = (end_date - relativedelta(months=12)).replace(day=1)
            object.__setattr__(self, "time_start", start_date.strftime("%Y-%m-%d"))
        else:
            start_date = date.fromisoformat(self.time_start).replace(day=1)
            object.__setattr__(self, "time_start", start_date.strftime("%Y-%m-%d"))
        
        if self.time_start > self.time_end:
            object.__setattr__(self, "time_start", self.time_end)
            object.__setattr__(self, "time_end", self.time_start)
        
        start = date.fromisoformat(self.time_start)
        end = date.fromisoformat(self.time_end)
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        object.__setattr__(self, "num_months", months)
        
        return self
