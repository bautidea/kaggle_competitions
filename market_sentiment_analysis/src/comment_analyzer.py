import pandas as pd
import numpy as np


class Comment_Analyzer ():
    def __init__(self, path_to_comments):
        self.path_to_comments = path_to_comments
        self.raw_data_df = self.__load_data()
        self.clean_data_df = self.__clean_data(self.raw_data_df)

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

    def __clean_data(self, raw_data):
        # Este metodo validara los datos, que no sean nulo, una vez validado, se procedera a realizar la limpieza del texto
        # en cada comentario.
        # Se aplciara:
        # - Transformacion de texto a minuscula.
        # - Eliminacion de numeros, caracteres especiales, signos de puntuacion,
        #    espacios multiples, stopwords.
        # - Se filtraran palabras que no tengan sentido.
        # - Se aplicara tokenizacion y stemming.

        # Para no tocar la referencia original, se crea una copia del dataframe.
        raw_data = raw_data.copy()
        # Validaremos los campos y los comentarios.
        raw_data = self.__validate_data(raw_data)

        return raw_data

    def return_clean_data(self):
        # Este metodo retorna los datos procesados y limpios, listos para predecir.
        return self.clean_data_df
