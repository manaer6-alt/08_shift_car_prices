
import numpy as np
import pandas as pd


MISSING_TOKENS = {
    "",
    "-",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
}


def normalize_text_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if column == "car_id":
            continue

        if (
            pd.api.types.is_object_dtype(result[column])
            or pd.api.types.is_string_dtype(result[column])
        ):
            cleaned = (
                result[column]
                .astype("string")
                .str.strip()
            )

            result[column] = cleaned.mask(
                cleaned.str.lower().isin(MISSING_TOKENS),
                pd.NA,
            )

    return result


def add_parsed_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Пробег_число"] = pd.to_numeric(
        result["Пробег"],
        errors="coerce",
    )

    result["Расход_л_на_100км"] = pd.to_numeric(
        result["Расход"]
        .astype("string")
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )

    engine = result["Двигатель"].astype("string")

    result["Двигатель_цилиндры"] = pd.to_numeric(
        engine.str.extract(r"(\d+)\s*cyl", expand=False),
        errors="coerce",
    )

    result["Двигатель_объём_л"] = pd.to_numeric(
        engine.str.extract(r"(\d+(?:\.\d+)?)\s*L\b", expand=False),
        errors="coerce",
    )

    result["Двери_число"] = pd.to_numeric(
        result["Двери"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    result["Кресла_число"] = pd.to_numeric(
        result["Количество кресел"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    result["Штат"] = (
        result["Локация"]
        .astype("string")
        .str.extract(r",\s*([A-Z]{2,3})$", expand=False)
    )

    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_text_missing_values(df)
    result = add_parsed_features(result)
    return result

def add_missingness_and_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    technical_columns = [
        "Пробег_число",
        "Двигатель_объём_л",
        "Двигатель_цилиндры",
        "Расход_л_на_100км",
    ]

    for column in technical_columns:
        result[f"{column}_пропущен"] = (
            result[column]
            .isna()
            .astype("int8")
        )

    fuel = (
        result["Топливо"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # fillna(False) нужен, потому что Топливо иногда отсутствует.
    is_ev = fuel.eq("ELECTRIC").fillna(False)

    result["Электромобиль"] = is_ev.astype("int8")

    is_non_ev = ~is_ev

    zero_consumption_non_ev = (
        result["Расход_л_на_100км"]
        .eq(0)
        .fillna(False)
        & is_non_ev
    )

    zero_engine_volume_non_ev = (
        result["Двигатель_объём_л"]
        .eq(0)
        .fillna(False)
        & is_non_ev
    )

    result["Нулевой_расход_у_не_EV"] = (
        zero_consumption_non_ev.astype("int8")
    )

    result["Нулевой_объём_у_не_EV"] = (
        zero_engine_volume_non_ev.astype("int8")
    )

    # Для не-EV ноль считаем технически некорректным значением.
    # Исходные колонки при этом сохраняются без изменений.
    result["Расход_л_на_100км_очищенный"] = (
        result["Расход_л_на_100км"]
        .mask(zero_consumption_non_ev, np.nan)
    )

    result["Двигатель_объём_л_очищенный"] = (
        result["Двигатель_объём_л"]
        .mask(zero_engine_volume_non_ev, np.nan)
    )

    result["Количество_пропусков_техданных"] = (
        result[
            [
                "Пробег_число_пропущен",
                "Двигатель_объём_л_пропущен",
                "Двигатель_цилиндры_пропущен",
                "Расход_л_на_100км_пропущен",
            ]
        ]
        .sum(axis=1)
        .astype("int8")
    )

    return result

def add_age_mileage_features(
    df: pd.DataFrame,
    reference_year: int = 2024,
) -> pd.DataFrame:
    result = df.copy()

    year = pd.to_numeric(
        result["Год выпуска"],
        errors="coerce",
    )

    mileage = pd.to_numeric(
        result["Пробег_число"],
        errors="coerce",
    )

    result["Возраст_авто"] = (
        reference_year - year
    ).clip(lower=0)

    age_for_ratio = result["Возраст_авто"].clip(lower=1)

    result["Лог_пробег"] = np.log1p(mileage)

    result["Пробег_на_год"] = (
        mileage / age_for_ratio
    )

    result["Лог_пробег_на_год"] = np.log1p(
        result["Пробег_на_год"]
    )

    return result

def add_age_mileage_ratio_features(
    df: pd.DataFrame,
    reference_year: int = 2024,
) -> pd.DataFrame:
    result = df.copy()

    year = pd.to_numeric(
        result["Год выпуска"],
        errors="coerce",
    )

    mileage = pd.to_numeric(
        result["Пробег_число"],
        errors="coerce",
    )

    age = (reference_year - year).clip(lower=1)

    # Новый смысловой признак: насколько интенсивно
    # автомобиль использовался относительно возраста.
    result["Пробег_на_год"] = mileage / age

    # Флаг для почти новых машин, где пробег особенно важен.
    result["Авто_до_года"] = (
        (reference_year - year) <= 1
    ).astype("int8")

    # Грубые возрастные категории могут помочь модели
    # проще разделять ценовые режимы.
    result["Годовой_сегмент"] = pd.cut(
        year,
        bins=[0, 2000, 2005, 2010, 2015, 2018, 2020, 2022, 2024],
        labels=[
            "до_2000",
            "2000_2005",
            "2005_2010",
            "2010_2015",
            "2015_2018",
            "2018_2020",
            "2020_2022",
            "2022_plus",
        ],
        include_lowest=True,
    ).astype("string")

    return result

def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    title = (
        result["Полное название"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # Убираем год только в начале строки:
    # "2019 TOYOTA HILUX SR (4X4)"
    # -> "TOYOTA HILUX SR (4X4)"
    title_without_year = title.str.replace(
        r"^\d{4}\s+",
        "",
        regex=True,
    )

    result["Название_без_года"] = title_without_year

    result["Название_число_слов"] = (
        title_without_year
        .str.findall(r"[A-Z0-9]+")
        .str.len()
    )

    result["Название_есть_4X4"] = (
        title_without_year
        .str.contains(r"\b4X4\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_AWD"] = (
        title_without_year
        .str.contains(r"\bAWD\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_TURBO"] = (
        title_without_year
        .str.contains(r"\bTURBO\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_SPORT"] = (
        title_without_year
        .str.contains(r"\bSPORT\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_HYBRID"] = (
        title_without_year
        .str.contains(r"\bHYBRID\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_GT"] = (
        title_without_year
        .str.contains(r"\bGT\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_LUXURY"] = (
        title_without_year
        .str.contains(r"\bLUXURY\b", regex=True, na=False)
        .astype("int8")
    )

    result["Название_есть_DIESEL"] = (
        title_without_year
        .str.contains(r"\bDIESEL\b", regex=True, na=False)
        .astype("int8")
    )

    return result


def add_title_hierarchy_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Добавляет иерархические и технические признаки
    из поля 'Полное название'.

    Функция включает все title-features из v4
    и дополняет их prefix-уровнями, мощностью и маркерами комплектаций.
    """

    result = add_title_features(df)

    title = (
        result["Полное название"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # Убираем год только в начале строки.
    title_without_year = title.str.replace(
        r"^\d{4}\s+",
        "",
        regex=True,
    )

    # Приводим пунктуацию к пробелам:
    # "(4X4)" -> "4X4"
    # "SS-V" -> "SS V"
    normalized_title = (
        title_without_year
        .str.replace(
            r"[^A-Z0-9]+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    result["Название_нормализованное_без_года"] = (
        normalized_title
        .mask(normalized_title.eq(""), pd.NA)
    )

    tokens = normalized_title.str.split()

    # Иерархия от широкого к узкому.
    result["Название_префикс_2"] = (
        tokens
        .str[:2]
        .str.join(" ")
        .mask(lambda x: x.eq(""), pd.NA)
        .astype("string")
    )

    result["Название_префикс_3"] = (
        tokens
        .str[:3]
        .str.join(" ")
        .mask(lambda x: x.eq(""), pd.NA)
        .astype("string")
    )

    result["Название_префикс_4"] = (
        tokens
        .str[:4]
        .str.join(" ")
        .mask(lambda x: x.eq(""), pd.NA)
        .astype("string")
    )

    # Мощность, если она указана в названии:
    # "(132KW)" -> 132
    result["Название_мощность_kw"] = pd.to_numeric(
        normalized_title.str.extract(
            r"\b(\d{2,3})\s*KW\b",
            expand=False,
        ),
        errors="coerce",
    ).astype(float)

    result["Название_есть_мощность_kw"] = (
        result["Название_мощность_kw"]
        .notna()
        .astype("int8")
    )

    # Один общий моторный маркер как категориальный признак.
    result["Название_моторный_маркер"] = (
        normalized_title
        .str.extract(
            r"\b("
            r"TDI|TSI|TFSI|CDI|HDI|D4D|"
            r"TD4|TD5|V6|V8|V10|V12|ECOBOOST"
            r")\b",
            expand=False,
        )
        .astype("string")
    )

    title_patterns = {
        "Название_есть_AMG": r"\bAMG\b",
        "Название_есть_M_SPORT": r"\bM\s+SPORT\b",
        "Название_есть_RS": r"\bRS\b",
        "Название_есть_S_LINE": r"\bS\s+LINE\b",
        "Название_есть_GTI": r"\bGTI\b",
        "Название_есть_HSE": r"\bHSE\b",
        "Название_есть_SR5": r"\bSR5\b",
        "Название_есть_GXL": r"\bGXL\b",
        "Название_есть_LIMITED": r"\bLIMITED\b",
        "Название_есть_PREMIUM": r"\bPREMIUM\b",
        "Название_есть_COMFORTLINE": r"\bCOMFORTLINE\b",
        "Название_есть_ASCENT": r"\bASCENT\b",
        "Название_есть_ACTIVE": r"\bACTIVE\b",
        "Название_есть_ELITE": r"\bELITE\b",
        "Название_есть_TDI": r"\bTDI\b",
        "Название_есть_TSI": r"\bTSI\b",
        "Название_есть_TFSI": r"\bTFSI\b",
        "Название_есть_CDI": r"\bCDI\b",
        "Название_есть_V6": r"\bV6\b",
        "Название_есть_V8": r"\bV8\b",
    }

    for column, pattern in title_patterns.items():
        result[column] = (
            normalized_title
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
            .astype("int8")
        )

    return result