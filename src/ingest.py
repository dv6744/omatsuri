import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.models import MatsuriDataset

load_dotenv("../")

MODEL = "gemini-3.1-flash-lite"

REGIONS = [
    "Hokkaido",
    "Tohoku (Aomori, Iwate, Miyagi, Akita, Yamagata, Fukushima)",
    "Kanto (Tokyo, Kanagawa, Saitama, Chiba, Ibaraki, Tochigi, Gunma)",
    "Chubu (Niigata, Toyama, Ishikawa, Fukui, Yamanashi, Nagano, Shizuoka, Aichi, Gifu)",
    "Kansai (Osaka, Kyoto, Hyogo, Nara, Wakayama, Shiga, Mie)",
    "Chugoku (Hiroshima, Okayama, Shimane, Tottori, Yamaguchi)",
    "Shikoku (Ehime, Kagawa, Kochi, Tokushima)",
    "Kyushu (Fukuoka, Saga, Nagasaki, Kumamoto, Oita, Miyazaki, Kagoshima, Okinawa)",
]


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=5,
    )


def llm_structured_gemini(client: OpenAI, user_prompt: str, model: str, model_class):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_class.__name__,
                "schema": model_class.model_json_schema(),
            },
        },
    )
    parsed = model_class(**json.loads(response.choices[0].message.content))
    return response, parsed, response.usage


def batch_generate_matsuris(
    client: OpenAI, model: str = MODEL, batch_size: int = 20, total: int = 100
) -> MatsuriDataset:

    all_records = []
    seen_ids: set[str] = set()

    for i in range(0, total, batch_size):
        region = REGIONS[i // batch_size % len(REGIONS)]
        prompt = f"""
            Generate {batch_size} unique Japanese matsuri events specifically from: {region}.
            Make sure to add detailed descriptions for each event, of about 3-4 sentences.
            Add estimates of annual turnout and relevance of the matsuri at national, prefectural, city or local neighbourhood level.
            Do NOT repeat any festival from a previous batch. Generate exactly {batch_size} records.
        """.strip()

        _, batch_dataset, _ = llm_structured_gemini(
            client=client, user_prompt=prompt, model=model, model_class=MatsuriDataset
        )

        for record in batch_dataset.matsuris:
            if record.id not in seen_ids:
                seen_ids.add(record.id)
                all_records.append(record)

        print(
            f"Batch {i // batch_size + 1} ({region.split(' ')[0]}): "
            f"got {len(batch_dataset.matsuris)} records — total so far: {len(all_records)}"
        )

        if len(all_records) >= total:
            break

    return MatsuriDataset(matsuris=all_records[:total])


def save_dataset(dataset: MatsuriDataset, path: str = "data/matsuris.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = [r.model_dump() for r in dataset.matsuris]  # record of each matsuri
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {path}")


def load_dataset(path: str = "data/matsuris.json") -> MatsuriDataset:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return MatsuriDataset(matsuris=records)


if __name__ == "__main__":
    client = build_client()
    dataset = batch_generate_matsuris(client, model=MODEL, batch_size=20, total=100)
    save_dataset(dataset)
