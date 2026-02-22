from pydantic import BaseModel


class LibraryInvestigatorReport(BaseModel):
    """Structured report from investigating library source code in site-packages."""

    packages_examined: str
    api_surface: str
    constructor_signatures: str
    key_methods: str
    usage_patterns: str
    caveats: str
