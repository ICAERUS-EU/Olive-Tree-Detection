from ultralytics import YOLO
import pandas as pd
import json

model_path = './yolov9e_80ep_formaug/weights/last.pt'
val_save_folder = './yolov9e_80ep_formaug/runs/detect/val'
val_name_folder = 'resultsBestTest'

# Load trainied model
model = YOLO(model_path)
print('Model loaded...')

# Validations
test = model.val(split="test", plots=True, save_json=True, project=val_save_folder,
                 name=val_name_folder)

for res in test.curves_results:

  df1 = pd.DataFrame(res[0], columns=[res[2]]).reset_index(drop=True)
  df2 = pd.DataFrame(res[1][0], columns=[res[3]]).reset_index(drop=True)
  results_df = pd.concat([df1, df2])

  results_df.to_csv(val_save_folder + '/' + val_name_folder + f'/{res[2]}-{res[3]}.csv')

results = {
    'fitness': test.fitness,
    'precision(B)': test.results_dict['metrics/precision(B)'],
    'recall(B)': test.results_dict['metrics/recall(B)'],
    'mAP50(B)': test.results_dict['metrics/mAP50(B)'],
    'mAP50-95(B)': test.results_dict['metrics/mAP50-95(B)'],
}

with open(val_save_folder + '/' + val_name_folder + '/results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)