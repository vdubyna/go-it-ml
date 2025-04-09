import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# --- ПОРОГИ ТА НАЛАШТУВАННЯ (ВАЖЛИВО НАЛАШТУВАТИ!) ---

# Поріг кардинальності для визначення "низької" (все, що нижче або рівне - кандидат на OHE)
PORIG_NISKA_KARD: int = 20

# Поріг кардинальності для визначення "дуже високої" (все, що вище - кандидат на Drop/Hash/Native)
# Можна також встановити як долю від загальної кількості рядків, наприклад 0.5
PORIG_DUJE_VISOKA_KARD: int = 1000

# Поріг унікальності, що вказує на можливий ID (доля унікальних від загальної кількості)
# Якщо > 95% значень унікальні, ймовірно, це ID
PORIG_ID_LIKENESS: float = 0.95

# Мінімальна "сила зв'язку" з цільовою змінною, щоб вважати фічу інформативною для TE
# Це стандартне відхилення середніх (для регресії) або пропорцій (для класифікації)
# Значення залежить від масштабу вашої цільової змінної (для регресії)!
# Можливо, варто нормалізувати target перед розрахунком або використовувати інші метрики.
PORIG_SYLA_ZVJAZKU_REG: float = 0.05 # Приклад для нормалізованої target або якщо ви знаєте масштаб
PORIG_SYLA_ZVJAZKU_CLASS: float = 0.05 # Приклад (стандартне відхилення пропорції класу)

# Поріг відсотку пропущених значень, вище якого рекомендується Drop
PORIG_MISSING_VALUES_PCT: float = 70.0

# --------------------------------------------------------

def calculate_relationship_strength(df: pd.DataFrame, feature: str, target: str) -> Optional[float]:
    """Розраховує спрощену міру зв'язку між категоріальною фічею та цільовою змінною."""
    if target not in df.columns:
        print(f"Попередження: Цільова змінна '{target}' не знайдена.")
        return None
    if df[target].isnull().all():
        print(f"Попередження: Цільова змінна '{target}' містить лише NaN.")
        return None

    try:
        if pd.api.types.is_numeric_dtype(df[target]):
            # Регресія: Стандартне відхилення середніх значень target по категоріях
            # Додаємо невелике значення до знаменника, щоб уникнути ділення на нуль для категорій з одним прикладом
            category_means = df.groupby(feature)[target].agg(lambda x: x.mean(skipna=True))
            if category_means.nunique() <= 1: # Якщо всі середні однакові
                return 0.0
            # Нормалізуємо std dev на загальний std dev цільової змінної для стабільності? (Опціонально)
            # return category_means.std(ddof=0) / df[target].std(ddof=0) if df[target].std(ddof=0) > 1e-6 else 0.0
            strength = category_means.std(ddof=0)
            print(f"  [Аналіз зв'язку {feature}-{target} (регр.)] StdDev середніх: {strength:.4f}")
            return strength

        else:
            # Класифікація: Стандартне відхилення пропорцій одного з класів по категоріях
            # Використовуємо перший клас як референс
            first_class = df[target].dropna().unique()[0]
            # Рахуємо частку першого класу для кожної категорії
            category_proportions = df.groupby(feature)[target].agg(lambda x: (x == first_class).mean())
            if category_proportions.nunique() <= 1: # Якщо всі пропорції однакові
                return 0.0
            strength = category_proportions.std(ddof=0)
            print(f"  [Аналіз зв'язку {feature}-{target} (клас.)] StdDev пропорцій класу '{first_class}': {strength:.4f}")
            return strength

    except Exception as e:
        print(f"Помилка при розрахунку зв'язку для '{feature}': {e}")
        return None


def suggest_encoding_strategies(
        df: pd.DataFrame,
        categorical_features: List[str],
        target_variable: Optional[str] = None,
        ordinal_features: Optional[List[str]] = None,
        id_like_threshold: float = PORIG_ID_LIKENESS,
        low_card_threshold: int = PORIG_NISKA_KARD,
        very_high_card_threshold: int = PORIG_DUJE_VISOKA_KARD,
        missing_values_threshold_pct: float = PORIG_MISSING_VALUES_PCT,
        relationship_threshold_reg: float = PORIG_SYLA_ZVJAZKU_REG,
        relationship_threshold_class: float = PORIG_SYLA_ZVJAZKU_CLASS
) -> Dict[str, str]:
    """
    Аналізує категоріальні фічі та пропонує стратегії кодування.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        categorical_features (List[str]): Список назв категоріальних стовпців.
        target_variable (Optional[str]): Назва цільової змінної (для аналізу зв'язку).
        ordinal_features (Optional[List[str]]): Список фіч, які є порядковими.
        id_like_threshold (float): Поріг унікальності для визначення ID-подібних фіч.
        low_card_threshold (int): Поріг кардинальності для OHE.
        very_high_card_threshold (int): Поріг кардинальності для Drop/Native.
        missing_values_threshold_pct (float): Поріг % пропущених значень для Drop.
        relationship_threshold_reg (float): Поріг сили зв'язку для регресії.
        relationship_threshold_class (float): Поріг сили зв'язку для класифікації.

    Returns:
        Dict[str, str]: Словник {назва_фічі: рекомендована_стратегія}.
    """
    suggestions: Dict[str, str] = {}
    ordinal_features_set = set(ordinal_features) if ordinal_features else set()
    n_rows = len(df)

    if n_rows == 0:
        print("Попередження: DataFrame порожній.")
        return {feature: "Error: DataFrame is empty" for feature in categorical_features}

    print(f"\n--- Аналіз та Рекомендації для Категоріальних Фіч ---")
    print(f"Поріг низької кардинальності (OHE): <= {low_card_threshold}")
    print(f"Поріг дуже високої кардинальності (Drop/Native): > {very_high_card_threshold}")
    print(f"Поріг ID-подібності: > {id_like_threshold*100:.1f}% унікальних")
    print(f"Поріг пропущених значень для Drop: > {missing_values_threshold_pct:.1f}%")
    if target_variable:
        print(f"Поріг сили зв'язку (Регресія): > {relationship_threshold_reg:.4f}")
        print(f"Поріг сили зв'язку (Класифікація): > {relationship_threshold_class:.4f}")
    else:
        print("Цільова змінна не надана, аналіз зв'язку проводитись не буде.")
    print("-" * 50)


    for feature in categorical_features:
        if feature not in df.columns:
            print(f"Попередження: Фіча '{feature}' не знайдена в DataFrame. Пропускаємо.")
            suggestions[feature] = "Not Found"
            continue

        print(f"\nАналіз фічі: '{feature}'")

        # 1. Перевірка на пропущені значення
        missing_pct = (df[feature].isnull().sum() / n_rows) * 100
        print(f"  Пропущено: {missing_pct:.2f}%")
        if missing_pct > missing_values_threshold_pct:
            print(f"  РЕКОМЕНДАЦІЯ: Drop (Дуже багато пропущених: > {missing_values_threshold_pct}%)")
            suggestions[feature] = f"Drop (High Missing %: {missing_pct:.2f}%)"
            continue

        # 2. Кардинальність
        # dropna=False важливо, щоб NaN вважався окремою категорією при аналізі
        n_unique = df[feature].nunique(dropna=False)
        print(f"  Кардинальність: {n_unique}")

        # 3. Перевірка на ID-подібність
        uniqueness_ratio = n_unique / n_rows
        if n_unique > 1 and uniqueness_ratio >= id_like_threshold:
            print(f"  РЕКОМЕНДАЦІЯ: Drop (ID-подібна фіча? {uniqueness_ratio*100:.1f}% унікальних)")
            suggestions[feature] = f"Drop (ID-like? {uniqueness_ratio*100:.1f}% unique)"
            continue
        if n_unique > very_high_card_threshold:
            print(f"  Кардинальність дуже висока (> {very_high_card_threshold}).")
            # Для дуже високої кардинальності без зв'язку з target, Drop є кандидатом
            # Але якщо зв'язок є, TE або XGBoost Native можуть бути кращими

        # 4. Перевірка, чи фіча порядкова
        if feature in ordinal_features_set:
            print(f"  Фіча позначена як порядкова.")
            # Навіть для порядкових, якщо кардинальність дуже мала, OHE може бути ок
            if n_unique <= low_card_threshold:
                print(f"  РЕКОМЕНДАЦІЯ: LabelEncode (Порядкова) або OHE (Низька кардинальність)")
                suggestions[feature] = "LabelEncode (Ordinal) / OHE"
            else:
                print(f"  РЕКОМЕНДАЦІЯ: LabelEncode (Порядкова)")
                suggestions[feature] = "LabelEncode (Ordinal)"
            continue # Переходимо до наступної фічі

        # 5. Основна логіка рекомендацій (для номінальних фіч)
        suggestion = "Н/Д" # Не визначено

        if n_unique <= low_card_threshold:
            suggestion = "OHE (One-Hot Encoding)"
            print(f"  РЕКОМЕНДАЦІЯ: {suggestion} (Низька кардинальність)")
        else: # Середня або висока кардинальність
            if target_variable:
                strength = calculate_relationship_strength(df, feature, target_variable)

                # Визначаємо поріг сили зв'язку в залежності від типу target
                relationship_threshold = None
                if strength is not None:
                    if pd.api.types.is_numeric_dtype(df[target_variable]):
                        relationship_threshold = relationship_threshold_reg
                    else:
                        relationship_threshold = relationship_threshold_class

                if strength is not None and relationship_threshold is not None and strength > relationship_threshold:
                    # Є зв'язок, кардинальність > low_threshold
                    suggestion = "TE (Target Encoding - Обережно!) / XGBoost-Native"
                    print(f"  РЕКОМЕНДАЦІЯ: {suggestion} (Висока кардинальність, є зв'язок з target)")
                elif strength is not None:
                    # Зв'язок слабкий або відсутній, кардинальність > low_threshold
                    if n_unique > very_high_card_threshold:
                        suggestion = "Drop (Дуже висока кардинальність, слабкий зв'язок?) / XGBoost-Native"
                        print(f"  РЕКОМЕНДАЦІЯ: {suggestion}")
                    else:
                        suggestion = "Drop (Слабкий зв'язок?) / XGBoost-Native / LabelEncode?"
                        print(f"  РЕКОМЕНДАЦІЯ: {suggestion} (Висока/середня кардинальність, слабкий зв'язок з target)")

                else: # Не вдалося розрахувати зв'язок або target не вказано
                    suggestion = "XGBoost-Native / TE (Перевірити!) / LabelEncode? / Hash?"
                    print(f"  РЕКОМЕНДАЦІЯ: {suggestion} (Висока/середня кардинальність, зв'язок з target не аналізувався/не розрахований)")
            else: # Target змінна не надана
                suggestion = "XGBoost-Native / LabelEncode? / Hash? / Drop?"
                print(f"  РЕКОМЕНДАЦІЯ: {suggestion} (Висока/середня кардинальність, target не вказано)")

        suggestions[feature] = suggestion

    print("-" * 50)
    print("--- Завершено ---")
    return suggestions

# --- Приклад Використання ---
if __name__ == "__main__":
    # Створюємо приклад DataFrame (використовуйте ваш реальний df)
    data = {
        'city': ['Kyiv', 'Lviv', 'Kyiv', 'Odesa', 'Lviv', 'Kyiv', 'Lviv', 'Kyiv', None, 'Dnipro', 'Kharkiv'] * 10,
        'product_type': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'D', 'A'] * 10,
        'user_status': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver', 'Gold', 'Silver', 'Gold', 'New', 'Bronze', 'Silver'] * 10,
        'country': ['Ukraine'] * 110, # Дуже низька кардинальність
        'user_id': [f'id_{i}' for i in range(110)], # ID-подібна фіча
        'feedback_score': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1] * 10, # Може бути порядковою
        'high_card_feat': [f'cat_{i % 55}' for i in range(110)], # Середня/висока кардинальність
        'feat_many_missing': [1 if i % 4 == 0 else None for i in range(110)], # Багато пропущених
        # Цільова змінна (приклад для регресії)
        # 'target_reg': np.random.rand(110) * 100 + [50 if c == 'Kyiv' else 30 for c in ['Kyiv', 'Lviv', 'Kyiv', 'Odesa', 'Lviv', 'Kyiv', 'Lviv', 'Kyiv', None, 'Dnipro', 'Kharkiv'] * 10] ,
        # Цільова змінна (приклад для бінарної класифікації)
        'target_class': [1 if c in ['Kyiv', 'Lviv'] and s=='Gold' else 0 for c, s in zip(['Kyiv', 'Lviv', 'Kyiv', 'Odesa', 'Lviv', 'Kyiv', 'Lviv', 'Kyiv', None, 'Dnipro', 'Kharkiv'] * 10, ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver', 'Gold', 'Silver', 'Gold', 'New', 'Bronze', 'Silver'] * 10)]
    }
    df_example = pd.DataFrame(data)
    df_example.loc[5:10, 'city'] = None # Додамо ще трохи пропусків у city

    # Список ваших категоріальних фіч
    categorical_cols = ['city', 'product_type', 'user_status', 'country', 'user_id', 'feedback_score', 'high_card_feat', 'feat_many_missing']

    # Список фіч, які ви вважаєте порядковими (якщо є)
    ordinal_cols = ['feedback_score'] # 'user_status' теж міг би бути, якщо є чіткий порядок

    # Назва вашої цільової змінної (або None, якщо її немає на цьому етапі)
    # target_col = 'target_reg'
    target_col = 'target_class'
    # target_col = None # Якщо аналізуємо без цільової змінної

    # Отримуємо рекомендації
    recommendations = suggest_encoding_strategies(
        df_example,
        categorical_cols,
        target_variable=target_col,
        ordinal_features=ordinal_cols
        # Можна передати власні пороги сюди, якщо потрібно:
        # low_card_threshold=15,
        # relationship_threshold_class=0.08
    )

    print("\n--- Підсумок Рекомендацій ---")
    for feature, suggestion in recommendations.items():
        print(f"{feature}: {suggestion}")