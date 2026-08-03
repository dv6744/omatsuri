from pydantic import BaseModel, Field
from typing import List


class MatsuriRecord(BaseModel):
    id: str = Field(description="Unique identifier eg. himeji-kenka-matsuri-0034")
    name: str = Field(description="Name of matsuri in jp-en eg. 姫路けんか祭り-himeji kenka matsuri")
    City: str = Field(description="City location eg. Himeji-city")
    Prefecture: str = Field(description="Prefecture location eg. Hyogo")
    Date: str = Field(description="Estimated date matsuri is held eg. 14th October or second week of May")
    Description: str = Field(description="Detailed historical information of the matsuri event, traditions and relevant details")
    AnnualTurnOut: str = Field(description="Estimated annual turnout to the matsuri event eg. 12000")
    RelevanceRating: str = Field(description="relevance 1 - national level, 2 - prefectural level, 3 - city level, 4 - local neighbourhood")


class MatsuriDataset(BaseModel):
    matsuris: List[MatsuriRecord]
