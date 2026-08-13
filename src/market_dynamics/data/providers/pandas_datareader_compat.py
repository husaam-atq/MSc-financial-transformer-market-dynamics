"""Compatibility bridge for pandas-datareader 0.10 under pandas 3.x."""

from __future__ import annotations

import inspect
from typing import Any


def import_datareader() -> Any:
    """Import pandas-datareader, adapting its legacy deprecate_kwarg call form.

    pandas-datareader 0.10 imports ``deprecate_kwarg(old, new)`` while pandas
    3.x changed the function to require a warning class first. The shim is
    applied only when that new signature is present and only for the import.
    """
    try:
        from pandas.util import _decorators

        original = _decorators.deprecate_kwarg
        parameters = list(inspect.signature(original).parameters)
        if parameters and parameters[0] == "klass":
            def compatible_deprecate_kwarg(*args: Any, **kwargs: Any) -> Any:
                if args and isinstance(args[0], str):
                    return original(FutureWarning, *args, **kwargs)
                return original(*args, **kwargs)
            _decorators.deprecate_kwarg = compatible_deprecate_kwarg
        from pandas_datareader import data
        return data
    except ImportError as exc:
        raise ImportError("pandas-datareader is required for this provider. Install it with `pip install pandas-datareader`.") from exc
