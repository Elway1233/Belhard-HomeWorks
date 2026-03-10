import cv2
import os
import torch
from datetime import datetime, timedelta

MODEL_PATH = 'runs/train/exp4/weights/best.pt'
#VIDEO_SOURCE = 0
VIDEO_SOURCE = 'http://192.168.0.109:8080/video'
SLEEP_CLASS_NAME = 'Sleeping'
SLEEP_THRESHOLD_SECONDS = 10

print("Загрузка модели...")
model = torch.hub.load('.', 'custom', path=MODEL_PATH, source='local')
model.conf = 0.25
print("Модель загружена!")

cap = cv2.VideoCapture(VIDEO_SOURCE)

is_sleeping = False
sleep_start_time = None
sleep_events = []

print("Начинаем мониторинг... (Нажмите 'q' на английской раскладке для выхода)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    detected_classes = results.pandas().xyxy[0]['name'].tolist()

    if SLEEP_CLASS_NAME in detected_classes:
        if not is_sleeping:
            is_sleeping = True
            sleep_start_time = datetime.now()
            timestamp_str = sleep_start_time.strftime('%Y-%m-%d_%H-%M-%S')
            photo_name = f"sleep_proofs/sleep_start_{timestamp_str}.jpg"
            frame_to_save = cv2.cvtColor(results.render()[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(photo_name, frame_to_save)
            print(f"[{sleep_start_time.strftime('%H:%M:%S')}] Сон начался. Фото сохранено: {photo_name}")
    else:
        if is_sleeping:
            is_sleeping = False
            sleep_end_time = datetime.now()
            sleep_duration = (sleep_end_time - sleep_start_time).total_seconds()
            if sleep_duration >= SLEEP_THRESHOLD_SECONDS:
                event = {
                    'start': sleep_start_time,
                    'end': sleep_end_time,
                    'duration': sleep_duration,
                    'photo': photo_name
                }
                sleep_events.append(event)
                print(f"Записан сон! Длительность: {int(sleep_duration)} сек.")

    drawn_frame = results.render()[0]
    final_frame = cv2.cvtColor(drawn_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Sleep Monitor', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if is_sleeping:
    sleep_end_time = datetime.now()
    sleep_duration = (sleep_end_time - sleep_start_time).total_seconds()
    if sleep_duration >= SLEEP_THRESHOLD_SECONDS:
        sleep_events.append({'start': sleep_start_time, 'end': sleep_end_time, 'duration': sleep_duration})

total_sleep_seconds = sum([event['duration'] for event in sleep_events])

if len(sleep_events) > 0:
    report_filename = f"sleep_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ О СНЕ\n\n")

        for i, event in enumerate(sleep_events, 1):
            start_str = event['start'].strftime('%H:%M:%S')
            end_str = event['end'].strftime('%H:%M:%S')

            dur_str = str(timedelta(seconds=int(event['duration'])))

            f.write(f"{i}. Уснул в {start_str}, проснулся в {end_str}. "
                    f"Сон: {dur_str}. Фото: {event['photo']}\n")

        total_str = str(timedelta(seconds=int(total_sleep_seconds)))
        f.write(f"\nИТОГО ОБЩЕЕ ВРЕМЯ СНА: {total_str}\n")

    print(f"\nОтчет успешно сохранен в файл: {report_filename}")
else:
    print("\nСобытий сна дольше 10 секунд не зафиксировано. Файл не создан.")