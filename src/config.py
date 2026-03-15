from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_outputs: Path
    geo: Path
    models: Path
    reports_figures: Path
    reports_tables: Path
    reports_tuning: Path

def get_paths(project_root: Path) -> Paths:
    return Paths(
        root=project_root,
        data_raw=project_root / "data" / "raw",
        data_interim=project_root / "data" / "interim",
        data_processed=project_root / "data" / "processed",
        data_outputs=project_root / "data" / "outputs",
        geo=project_root / "geo",
        models=project_root / "models",
        reports_figures=project_root / "reports" / "figures",
        reports_tables=project_root / "reports" / "tables",
        reports_tuning=project_root / "reports" / "tuning",
    )
