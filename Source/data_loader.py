import pandas as pd 
import os 
import logging 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path


    def load_datasets(self):
        if not os.path.isdir(self.folder_path):
            logging.error(f"Papka Topilmadi: {self.folder_path}")
            return pd.DataFrame()
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        if not csv_files:
            logging.warning("Papka ichida hech qanday csv fayl yo`q")
            return pd.DataFrame()
        
        df_list = []


        for file in csv_files:
            file_path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = file 
                df_list.append(df)
                logging.info(f"Yuklandi: {file}")
            except Exception as e:
                logging.error(f"Xatolik '{file}' faylni yuklashda : {e}")

        try:
            full_df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Umumiy data yuklandi, Shakli: {full_df.shape}")
            return full_df
        except Exception as e:
            logging.error(f"Dataframe birlaashtirishda xatolik bo`ldi: {e}")
            return pd.DataFrame()
