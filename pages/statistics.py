import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path


@st.cache_data
def load_dfs(files, columns):
    base_path = Path(__file__).resolve().parents[1] / "documents"

    dfs = []
    for file in files:
        path = base_path / file
        df = pd.read_csv(
            base_path / file, 
            header=0, 
            names=columns, 
            dtype={"intressent_id": "string"}, 
            encoding="utf-8"
        )
        frames.append(df)
    return dfs

anforande_files = [
    "anforande-202223.csv",
    "anforande-202324.csv",
    "anforande-202425.csv",
]
votering_files = [
    "votering-202223.csv",
    "votering-202324.csv",
    "votering-202425.csv",
    "votering-202526.csv",
]

anforande_columns = [
    'dok_id', 
    'dok_rm', 
    'dok_nummer', 
    'dok_datum', 
    'avsnittsrubrik', 
    'kammaraktivitet', 
    'anforande_nummer', 
    'talare', 
    'parti', 
    'intressent_id', 
    'rel_dok_id', 
    'replik'
]

votering_columns = [
    'rm',
    'beteckning',
    'votering_id',
    'punkt',
    'namn',
    'intressent_id',
    'parti',
    'valkrets',
    'rost',
    'avser',
    'banknummer',
    'kon',
    'fodd',
    'systemdatum'
]

anforande_dfs = load_dfs(anforande_files, anforande_columns)
votering_dfs = load_dfs(votering_files, votering_columns)

st.write(anforande_dfs)