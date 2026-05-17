import logging
import pandas as pd

from controller.modules.results_aggregator import Aggregator
from controller.modules.state_manager import StateManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Test")

# ==========================================
# ОЖИДАНИЯ (Эталонные результаты)
# ==========================================
EXPECTED_RESULTS = {
    1: {"desc": "Идеальный (3 из 3)", "price": 100.0, "status": "DONE", "conf_contains": "good"}, # Или excellent, если обновил агрегатор
    2: {"desc": "Частичный (2 из 3)", "price": 200.0, "status": "DONE", "conf_contains": "good"},
    3: {"desc": "Хаос (все разные)",  "price": 301.0, "status": "DONE", "conf_contains": "uncertain"},
    4: {"desc": "Таймаут (спасен)",   "price": 400.0, "status": "DONE", "conf_contains": "good"},
    5: {"desc": "Мусор (отброшен)",   "price": None,  "status": "DROPPED", "conf_contains": None}
}

def create_mock_vision_result(price_card, processed=True):
    if not processed:
        return {"processed": False, "description": "blurry", "payload": None}
    return {"processed": True, "payload": {"price_card": price_card, "product_name": "Тест"}}

def run_synthetic_test():
    logger.info("🛠 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА...")

    state = StateManager(threshold=3, timeout=3)
    aggregator = Aggregator(target_frames_count=3, min_matches_required=2)

    timeline = [
        (1, [
            (1, create_mock_vision_result(100.0)),
            (2, create_mock_vision_result(200.0)),
            (3, create_mock_vision_result(301.0)),
            (4, create_mock_vision_result(400.0)),
            (5, create_mock_vision_result(None, processed=False)),
        ]),
        (2, [
            (1, create_mock_vision_result(100.0)),
            (2, create_mock_vision_result(200.0)),
            (3, create_mock_vision_result(302.0)),
            (4, create_mock_vision_result(400.0)),
            (5, create_mock_vision_result(None, processed=False)),
        ]),
        (3, [
            (1, create_mock_vision_result(100.0)),
            (2, create_mock_vision_result(999.0)), # Галлюцинация VLM
            (3, create_mock_vision_result(303.0)),
            # ID 4 пропал
            (5, create_mock_vision_result(None, processed=False)),
        ]),
        (4, []), (5, []), (6, []), (7, []) # Крутим время для таймаутов
    ]

    # Симуляция
    for frame_num, frame_data in timeline:
        for track_id, vision_res in frame_data:
            state.add_result(track_id, frame_num, vision_res)

        ready_ids = state.get_ready_to_close(frame_num)
        for r_id in ready_ids:
            raw_results = state.memory_state[r_id]["temp_results"]
            if raw_results:
                state.save_final(r_id, aggregator.vote(raw_results), status="DONE")
            else:
                state.save_final(r_id, None, status="DROPPED")

    # ==========================================
    # ПРОВЕРКА РЕЗУЛЬТАТОВ (Ожидание vs Реальность)
    # ==========================================
    print("\n" + "="*60)
    print("📊 ОТЧЕТ О ТЕСТИРОВАНИИ (ОЖИДАНИЕ vs РЕАЛЬНОСТЬ)")
    print("="*60)

    df = state.export_to_dataframe()
    all_passed = True

    for t_id, expected in EXPECTED_RESULTS.items():
        actual_state = state.memory_state.get(t_id, {})
        actual_status = actual_state.get("status")

        print(f"\nID {t_id} | Сценарий: {expected['desc']}")
        print(f"  Ожидаемый статус: {expected['status']}")
        print(f"  Фактический статус: {actual_status}")

        passed = (actual_status == expected['status'])

        if actual_status == "DONE" and passed:
            actual_price = df.loc[t_id, 'price_card'] if t_id in df.index else None
            actual_conf = df.loc[t_id, '_confidence'] if t_id in df.index else "N/A"

            print(f"  Ожидаемая цена: {expected['price']} | Факт: {actual_price}")
            print(f"  Ожидаемая уверенность (содержит): '{expected['conf_contains']}' | Факт: '{actual_conf}'")

            if actual_price != expected['price'] or expected['conf_contains'] not in actual_conf:
                passed = False

        if passed:
            print("  ✅ РЕЗУЛЬТАТ: [PASS]")
        else:
            print("  ❌ РЕЗУЛЬТАТ: [FAIL]")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("🏆 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! АРХИТЕКТУРА ИДЕАЛЬНА.")
    else:
        print("⚠️ ЕСТЬ ОШИБКИ. ПРОВЕРЬТЕ ЛОГИКУ АГРЕГАТОРА.")
    print("="*60)

if __name__ == "__main__":
    run_synthetic_test()