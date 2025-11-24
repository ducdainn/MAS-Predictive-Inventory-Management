"""
System Date Configuration

Allows setting a "current date" for the system to use instead of real current date.
Useful when working with historical/future data.

Usage:
    from agent.system_date import get_system_date, set_system_date
    
    # Set system date
    set_system_date("2025-12-06")
    
    # Get system date (returns configured date or real current date)
    current_date = get_system_date()
"""

from datetime import datetime, date
from typing import Optional, Union
import os
import json

# Config file path
CONFIG_FILE = "system_date_config.json"

# Global variable
_SYSTEM_DATE: Optional[date] = None


def set_system_date(date_value: Union[str, date, datetime, None]):
    """
    Set the system date.
    
    Args:
        date_value: Date as string "YYYY-MM-DD", date object, datetime object, or None to use real current date
    
    Examples:
        set_system_date("2025-12-06")
        set_system_date(datetime(2025, 12, 6))
        set_system_date(None)  # Use real current date
    """
    global _SYSTEM_DATE
    
    if date_value is None:
        _SYSTEM_DATE = None
        print("✅ System date reset to real current date")
        return
    
    if isinstance(date_value, str):
        _SYSTEM_DATE = datetime.strptime(date_value, "%Y-%m-%d").date()
    elif isinstance(date_value, datetime):
        _SYSTEM_DATE = date_value.date()
    elif isinstance(date_value, date):
        _SYSTEM_DATE = date_value
    else:
        raise ValueError(f"Invalid date_value type: {type(date_value)}")
    
    print(f"✅ System date set to: {_SYSTEM_DATE}")
    
    # Save to config file
    save_system_date_config(_SYSTEM_DATE)


def get_system_date() -> date:
    """
    Get the system date (configured date or real current date).
    
    Returns:
        date: The system date
    """
    global _SYSTEM_DATE
    
    # Load from config if not set
    if _SYSTEM_DATE is None:
        _SYSTEM_DATE = load_system_date_config()
    
    # Return configured date or real current date
    return _SYSTEM_DATE if _SYSTEM_DATE else date.today()


def get_system_datetime() -> datetime:
    """
    Get the system datetime (configured date at 00:00:00 or real current datetime).
    
    Returns:
        datetime: The system datetime
    """
    system_date = get_system_date()
    
    # If using configured date, return at midnight
    if _SYSTEM_DATE:
        return datetime.combine(system_date, datetime.min.time())
    else:
        return datetime.now()


def is_using_custom_date() -> bool:
    """Check if using a custom system date."""
    return _SYSTEM_DATE is not None


def get_date_info() -> dict:
    """Get information about current date configuration."""
    system_date = get_system_date()
    real_date = date.today()
    
    return {
        'system_date': system_date.isoformat(),
        'real_date': real_date.isoformat(),
        'using_custom': is_using_custom_date(),
        'days_difference': (system_date - real_date).days
    }


def save_system_date_config(date_value: Optional[date]):
    """Save system date configuration to file."""
    config = {
        'system_date': date_value.isoformat() if date_value else None,
        'set_at': datetime.now().isoformat()
    }
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save system date config: {e}")


def load_system_date_config() -> Optional[date]:
    """Load system date configuration from file."""
    if not os.path.exists(CONFIG_FILE):
        return None
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        date_str = config.get('system_date')
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception as e:
        print(f"⚠️  Could not load system date config: {e}")
    
    return None


def auto_detect_and_set_system_date(db_manager):
    """
    Auto-detect the latest date in data and set as system date.
    
    Args:
        db_manager: DatabaseManager instance
    
    Returns:
        date: The detected latest date
    """
    try:
        # Get latest date from sales
        result = db_manager.execute_query("SELECT MAX(date) as max_date FROM sales")
        
        if not result.empty and result.iloc[0]['max_date']:
            latest_date = result.iloc[0]['max_date']
            
            # Convert to date if datetime
            if isinstance(latest_date, datetime):
                latest_date = latest_date.date()
            elif isinstance(latest_date, str):
                latest_date = datetime.strptime(latest_date[:10], "%Y-%m-%d").date()
            
            # Set as system date
            set_system_date(latest_date)
            
            print(f"✅ Auto-detected latest data date: {latest_date}")
            print(f"   System date set to match data")
            
            return latest_date
        else:
            print("⚠️  Could not detect latest date from data")
            return None
            
    except Exception as e:
        print(f"⚠️  Error auto-detecting system date: {e}")
        return None


# Initialize on import
_SYSTEM_DATE = load_system_date_config()

if _SYSTEM_DATE:
    print(f"ℹ️  System date configured: {_SYSTEM_DATE}")




