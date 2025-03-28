from typing import Literal
import pandas as pd
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
import re
import unicodedata
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib


class Comment_Analyzer ():
    def __init__(
        self,
        path_to_comments: str,
        vectorizer_path: str,
        model_path: str,
    ):
        # Asigno variables de instancia.
        self.path_to_comments = path_to_comments
        self.raw_data_df = self.__validate_data(self.__load_data())

        # Limpio la data cruda.
        self.clean_data_df = self.__clean_data()

        # Cargo modelo y vectorizador para procesar la data y luego predecir.
        self.vectorizer_instance = self.__load_vectorizer(vectorizer_path)
        self.model_instance = self.__load_model(model_path)
        # Preprocesamos la data para luego predecir.
        self.processed_data_df = self.__preprocess_data()
        # Predecimos la data procesada.
        self.predictions = self.model_instance.predict(self.processed_data_df)

    def __load_data(self):
        # Este metodo cargaria los datos, en este caso es un csv, pero cualquier tipo de extraccion
        # desde una consulta a SQL a archivos alojados en alguna nube.
        return pd.read_csv(self.path_to_comments, low_memory=False)

    def return_raw_data(self):
        # Este metodo retornaria los datos crudos, en este caso un dataframe de pandas.
        return self.raw_data_df

    def __validate_data(self, raw_data):
        # Este metodo valida los datos, lo que se hara es filtrar nulos para el caso en el que el campo
        # 'content' sea nulo. y se imputara 'title' (si este llegara a ser nulo) con un string vacio.

        # Primero validaremos que el campo 'content' y 'title' esten presentes en el DF de entrada.
        if ('content' in raw_data.columns) and ('title' in raw_data.columns):
            raw_data['content'] = raw_data['content'].replace(
                '', np.nan)
            raw_data['title'] = raw_data['title'].fillna('')

            print(
                f"Cantidad de registros sin contenido = {(raw_data['content'].isnull()).sum()}")
            print(
                f"Cantidad de registros sin titulo = {(raw_data['title'] == '').sum()}")

            raw_data.dropna(subset=['content'], inplace=True)

            return raw_data
        else:
            raise Exception(
                'El campo content o title no se encuentra en el dataframe de entrada.')

    def __clean_data(self):
        # Este metodo validara los datos, que no sean nulo, una vez validado, se procedera a realizar la limpieza del texto
        # en cada comentario.
        # Se aplciara:
        # - Transformacion de texto a minuscula.
        # - Eliminacion de numeros, caracteres especiales, signos de puntuacion,
        #    espacios multiples, stopwords.
        # - Se filtraran palabras que no tengan sentido.
        # - Se aplicara tokenizacion y stemming.

        # Funcion interna para realizar Stemming.
        # Aaplicara Stemming a los commentarios, para reducir las palabras que componen a los
        # comentarios a su raiz etimologica.
        def stemming(text):
            stemmer = SnowballStemmer(language='spanish')
            # Stemming. Transformamos texto a lista ya que el stemmer trabaja sobre string crudos, no sobre
            # linguisticos como lemmatization.
            tokens = text.split()

            return [stemmer.stem(token) for token in tokens]

        # Funcion interna para realizar limpieza de texto.
        def clean_text(text):
            # Transformo texto a minusculas.
            text = str(text).lower()

            # Elimino tildes/acentos.
            text = unicodedata.normalize('NFKD', text).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            # Elimino numeros.
            text = re.sub(r'\d+', ' ', text)
            # Elimino signos de puntuacion.
            text = re.sub(r'[^\w\s]', ' ', text)
            # Hay algunos comentarios que tienen la nueva linea como '\n' especificamente
            # reemplazare '\n' -> ' '
            text = re.sub(r'\n', ' ', text)
            # Eliminamos espacios multiples.
            text = re.sub(r'\s+', ' ', text)

            # Stemmingzamos.
            tokens = stemming(text)

            text = ' '.join(tokens)

            # Vuelvo a eliminar tildes por si Lemmatizacion introdujo alguna.
            text = unicodedata.normalize('NFKD', text).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            return text

        # Para no tocar la referencia original, se crea una copia del dataframe.
        clean_data = self.raw_data_df.copy()

        # Una vez validados los dos campos, procedemos a crear el campo 'text_label' que contendra
        # el texto limpio y procesado.
        clean_data['text_label'] = clean_data['title'] + \
            ' ' + clean_data['content']

        # Aplicamos la limpieza de texto a cada comentario.
        clean_data['text_label'] = clean_data['text_label'].apply(clean_text)
        return clean_data

    def return_clean_data(self):
        # Este metodo retorna los datos procesados y limpios, listos para predecir.
        return self.clean_data_df

    def __load_model(self, model_path: str):
        # Este metodo cargara el modelo de clasificacion entrenado previamente.
        try:
            instance = joblib.load(model_path)
            print(f'Modelo cargado correctamente...')
            print(instance)
            return instance
        except Exception as e:
            raise Exception(
                f"No se pudo cargar el modelo desde '{model_path}'. Error: {e}")

    def __load_vectorizer(self, vectorizer_path: str):
        # Este metodo cargara el vectorizador entrenado previamente.
        try:
            instance = joblib.load(vectorizer_path)
            print(f'Vectorizador cargado correctamente...')
            return instance
        except Exception as e:
            raise Exception(
                f"No se pudo cargar el vectorizador desde '{vectorizer_path}'. Error: {e}")

    def __preprocess_data(self):
        # Este metodo preprocesara los datos, aplicando el vectorizador a los comentarios limpios.
        processed_data_df = self.clean_data_df.copy()

        word_count = self.vectorizer_instance.transform(
            processed_data_df['text_label']
        ).toarray()

        return pd.DataFrame(
            word_count, columns=self.vectorizer_instance.get_feature_names_out())

    def return_processed_data(self):
        # Este metodo retornara los datos procesados, listos para predecir.
        return self.processed_data_df

    def return_word_cloud(self, on_data: Literal['raw', 'clean', 'predicted'] = 'predicted'):
        def plot_word_cloud(text, title, generate_form_frequecies=False):
            if not generate_form_frequecies:
                wordcloud = WordCloud(width=900, height=900).generate(text)
            else:
                wordcloud = WordCloud(
                    width=1000, height=1000).generate_from_frequencies(text)

            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"WordCloud -  {title}")
            plt.show()

        if on_data == 'raw':
            text = ' '.join(
                self.raw_data_df['content'].str.lower().astype(str)
            )
            plot_word_cloud(text, 'Data Cruda')

        elif on_data == 'clean':
            text = ' '.join(
                self.clean_data_df['text_label'].astype(str)
            )
            plot_word_cloud(text, 'Data Limpia')

        elif on_data == 'predicted':
            predicted_data_df = self.processed_data_df.copy()
            predicted_data_df['target'] = self.predictions

            target_list = predicted_data_df['target'].sort_values(
            ).unique().tolist()

            for target_iter in target_list:
                mask = predicted_data_df['target'] == target_iter
                df_iter = predicted_data_df[mask].copy()
                word_freq = df_iter.drop(
                    columns=['target']).sum(axis=0).to_dict()

                plot_word_cloud(
                    word_freq, f'Clase: {target_iter}', generate_form_frequecies=True)
        else:
            raise Exception(
                'El parametro "on_data" debe ser "raw", "clean" o "predicted".')
