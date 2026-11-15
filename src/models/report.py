from pydantic import BaseModel, ConfigDict, Field

class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary:str = Field(min_length=1)
    evidence_ids: list[str]
    clause_ids:list[str]
    limitations:list[str] = Field(min_length=1)