# CVDL2024

## Overview

Questo progetto implementa un'architettura a due step basato su YOLOv10 e YOLOv8 per la detection e la classificazione di pescatori in immagini. Utilizza di default Grad-CAM per la detection e EigenCAM per la classificazione per visualizzare dove la rete si è maggiormente concentrata per prendere la sua decisione.

## Benchmark

La Tabella riassume i risultati in termini di Precision, Recall, mAP50, mAP50-95 e tempi di inferenza, confrontando i modelli N, S ed M di YOLOv8 e YOLOv10.

|          |            |   YOLOv8    |         |            |    YOLOv10  |         |
|:----------:|:------------:|:-------------:|:---------:|:------------:|:-------------:|:---------:|
|          | N          | S           | M       | N          | S           | M       |
| Precision| 0.762      | 0.798       | 0.815   | 0.721      | 0.8         | 0.806   |
| Recall   | 0.471      | 0.525       | 0.575   | 0.488      | 0.542       | 0.564   |
| mAP50    | 0.551      | 0.605       | 0.655   | 0.559      | 0.624       | 0.644   |
| mAP50-95 | 0.317      | 0.354       | 0.389   | 0.313      | 0.36        | 0.384   |
| Inference (ms) | 3.4   | 5.4         | 9.7     | 4.4        | 7.2         | 10.1    |


## Installazione

È consigliato utilizzare `Conda` per gestire l'ambiente virtuale e le dipendenze del progetto.

Creare e attivare un ambiente `Conda`:

```bash
conda create -n net2step python=3.9
conda activate net2step
```

Per prima cosa verificare di avere `git` installato e clonare la repository mediante il seguente comando:
```bash
git clone https://github.com/Emanuele1087650/CVDL2024
```

Successivamente, spostarsi all'interno della cartella di lavoro:
```bash
cd CVDL2024
```

Installare le dipendenze necessarie:
```bash
pip install -r requirements.txt
```

## Utilizzo

La funzione main accetta tre parametri opzionali per eseguire la detection e la classificazione a due step.

+ `<immagine>` o `<cartella di immagini>`: Specifica il percorso dell'immagine. In alternativa inserire il perocrso di una cartella di immagini.
+ `<Grad-Cam>`: Specifica se utilizzare Grad-Cam per la detection. Valori accettati: `True` o `False`.
+ `<EigenCAM>`: Specifica se utilizzare EigenCAM per la classificazione. Valori accettati: `True` o `False`.

Il comando per l'utilizzo specificando un'immagine è il seguente:

```bash
python3 net2step.py '<immagine>' <cam_detection> <cam_classification>
```

In alternativa, specificando una cartella di immagini, verrà fatta l'inferenza su tutte le immagini in cartella.
```bash
python3 net2step.py '<cartella di immagini>' <cam_detection> <cam_classification>
```

Se non si desidera provare su specifiche immagini, nella cartella di progetto sono presenti delle immagini di prova.
Con il comando seguente verrà fatta l'inferenza su tutte le immagini nella cartella `test`. I parametri `<Grad-Cam>` e `<EigenCAM>` sono impostati a `False`.
```bash
python3 net2step.py
```

Di seguito è riportato un esempio di utilizzo sulle immagini della cartella `test` attivando `<Grad-Cam>`:
```bash
python3 net2step.py 'test' True False
```

## Risultati

I risultati verranno salvati all'interno della cartella di progetto, in particolare nella cartella `Risultati`.
Ogni immagine processata avrà una propria sotto-cartella con il nome dell'immagine.

```css
Risultati/
└── NomeImmagine/
    ├── classificazione/ (opzionale)
        ├── eigen_cam0.jpg
    ├── cam_detection.jpg (opzionale)
    └── detection.jpg
```

